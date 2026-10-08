"""
ESPN Fantasy Football Dashboard
--------------------------------
Pulls live league data (standings, rosters, matchups, power rankings,
free agents, injuries) from a private ESPN Fantasy Football league
using the espn_api library, and displays it in a Streamlit dashboard.

Setup: see README.md
Run:   streamlit run app.py
"""

import html
import os
import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from espn_api.football import League
from scipy.optimize import linear_sum_assignment

load_dotenv()

st.set_page_config(page_title="ESPN Fantasy Dashboard", layout="wide")

# Injury statuses, after normalize_status() (ESPN's INJURY_RESERVE shows up
# as "INJURED RESERVE").
NON_ACTIVE_STATUSES = {
    "QUESTIONABLE",
    "DOUBTFUL",
    "OUT",
    "INJURED RESERVE",
    "SUSPENSION",
    "PUP",
}
# Players with these statuses can't play, so they're never recommended as pickups.
UNAVAILABLE_STATUSES = {"OUT", "INJURED RESERVE", "SUSPENSION"}
# Injury report sort order: least severe first, most severe last.
STATUS_SEVERITY = {
    "QUESTIONABLE": 1,
    "DOUBTFUL": 2,
    "OUT": 3,
    "SUSPENSION": 4,
    "PUP": 5,
    "INJURED RESERVE": 6,
}
HISTORY_MAX_SEASONS = 10  # how far back the all-time league records look

STATUS_COLORS = {
    "ACTIVE": "#2e9e4f",           # green
    "QUESTIONABLE": "#e0b400",     # yellow
    "DOUBTFUL": "#f28c28",         # orange
    "OUT": "#ff2b2b",              # bright red
    "INJURED RESERVE": "#9b1c1c",  # dark red / maroon
}
WIN_COLOR = "#2e9e4f"
LOSS_COLOR = "#e03131"


# ---------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------
# ttl forces Streamlit to drop the cached League and reconnect periodically,
# which is what actually pulls fresh rosters/waivers from ESPN. Without a
# ttl, cache_resource holds the same connection (and its stale data) forever.
LEAGUE_CACHE_TTL_SECONDS = 30 * 60  # 30 minutes
FREE_AGENT_LIST_SIZE = 100  # names shown on the Free Agents tab (was 50)
FREE_AGENTS_PER_POSITION = 50  # pool per position for the lineup optimizer


@st.cache_resource(show_spinner="Connecting to ESPN...", ttl=LEAGUE_CACHE_TTL_SECONDS)
def get_league(league_id: int, year: int, espn_s2: str, swid: str) -> League:
    return League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)


def load_credentials():
    """Pull credentials from .env, falling back to sidebar inputs."""
    st.sidebar.header("League Connection")

    league_id = st.sidebar.text_input(
        "League ID", value=os.getenv("LEAGUE_ID", "")
    )

    current_calendar_year = datetime.now().year
    year_options = list(range(2018, current_calendar_year + 2))
    default_year = int(os.getenv("LEAGUE_YEAR", current_calendar_year))
    default_index = (
        year_options.index(default_year) if default_year in year_options else len(year_options) - 1
    )
    year = dropdown("Season", year_options, container=st.sidebar, index=default_index)

    espn_s2 = st.sidebar.text_input(
        "espn_s2", value=os.getenv("ESPN_S2", ""), type="password"
    )
    swid = st.sidebar.text_input(
        "SWID", value=os.getenv("ESPN_SWID", ""), type="password"
    )

    st.sidebar.divider()
    if st.sidebar.button("🔄 Refresh Data Now"):
        get_league.clear()
        get_week_projection_lookup.clear()
        get_free_agent_pool.clear()
        st.rerun()
    st.sidebar.caption(
        "Data auto-refreshes every 30 min. Use the button above right "
        "after waivers process if you don't want to wait."
    )

    return league_id, year, espn_s2, swid


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------
def _safe_round(value, digits=1):
    """Round a stat value, tolerating None or missing attributes."""
    return round(value, digits) if isinstance(value, (int, float)) else None


# Lineup-slot display order, matching ESPN's team page:
# QB, RB, WR, TE, FLEX, DP, K, Bench, IR.
SLOT_ORDER = [
    "QB", "TQB", "RB", "WR", "TE",
    "RB/WR", "WR/TE", "RB/WR/TE", "OP",               # flex-type slots
    "DT", "DE", "LB", "DL", "CB", "S", "DB", "DP",    # IDP slots
    "D/ST", "K", "P", "HC",
    "BE", "IR",
]
SLOT_LABELS = {"RB/WR/TE": "FLEX", "BE": "Bench"}


def slot_rank(slot: str) -> int:
    return SLOT_ORDER.index(slot) if slot in SLOT_ORDER else len(SLOT_ORDER)


def dropdown(label, options, container=st, **kwargs):
    """Select-only dropdown: typing/filtering is switched off.
    filter_mode=None needs a recent Streamlit; on an older one we fall
    back to a normal selectbox instead of crashing (upgrade with
    `pip install -U streamlit` to get select-only behaviour)."""
    try:
        return container.selectbox(label, options, filter_mode=None, **kwargs)
    except TypeError:
        return container.selectbox(label, options, **kwargs)


def fit_height(df, row_px: int = 35) -> int:
    """Pixel height that shows every row (plus header) with no scrolling."""
    return (len(df) + 1) * row_px + 3


def normalize_status(raw) -> str:
    """ESPN injury status -> display label. INJURY_RESERVE becomes
    INJURED RESERVE; a missing status means ACTIVE."""
    text = str(raw).strip().upper().replace("_", " ") if raw else "ACTIVE"
    if text in ("INJURY RESERVE", "INJURED RESERVE"):
        return "INJURED RESERVE"
    return text or "ACTIVE"


def _status_css(value) -> str:
    color = STATUS_COLORS.get(value)
    return f"color: {color}; font-weight: 600" if color else ""


def _streak_css(value) -> str:
    text = str(value).upper()
    if text.startswith("WIN"):
        return f"color: {WIN_COLOR}; font-weight: 600"
    if text.startswith("LOSS"):
        return f"color: {LOSS_COLOR}; font-weight: 600"
    return ""


def styled(df, status_cols=(), streak_col=None):
    """A pandas Styler for st.dataframe: injury statuses and win/loss streaks
    are colored, numbers show one decimal and missing values show a dash."""
    sty = df.style.format(precision=1, na_rep="—")
    # Styler.map replaced applymap in pandas 2.1; support both.
    def apply_css(sty, func, cols):
        mapper = sty.map if hasattr(sty, "map") else sty.applymap
        return mapper(func, subset=cols)

    cols = [c for c in status_cols if c in df.columns]
    if cols:
        sty = apply_css(sty, _status_css, cols)
    if streak_col and streak_col in df.columns:
        sty = apply_css(sty, _streak_css, [streak_col])
    return sty


def get_season_proj_points(player):
    """Season-long projected total. Useful context, but NOT for weekly
    lineup decisions -- use get_week_proj_points for those."""
    proj = getattr(player, "projected_total_points", None)
    return proj if isinstance(proj, (int, float)) and proj else None


@st.cache_data(ttl=LEAGUE_CACHE_TTL_SECONDS, show_spinner="Loading weekly projections...")
def get_week_projection_lookup(_league, league_id: int, year: int, week: int) -> dict:
    """
    {playerId: projected points for that week} for every rostered player
    in the league, taken from the week's box scores (which carry ESPN's
    week-specific projection, bench players included). league_id/year are
    only cache keys; _league is excluded from hashing.
    """
    lookup = {}
    try:
        boxes = _league.box_scores(week=week)
    except Exception:
        return lookup
    for box in boxes:
        for lineup in (
            getattr(box, "home_lineup", []) or [],
            getattr(box, "away_lineup", []) or [],
        ):
            for p in lineup:
                proj = getattr(p, "projected_points", None)
                pid = getattr(p, "playerId", None)
                if pid is not None and isinstance(proj, (int, float)):
                    lookup[pid] = proj
    return lookup


def get_week_proj_points(player, week: int, lookup: dict | None = None):
    """
    Projected points for ONE week, or None if ESPN has no projection for
    that player/week. Deliberately never falls back to the season total
    (or season/17): a silent fallback would reintroduce the exact
    season-vs-week mix-up this function exists to prevent.

    Lookup order: the week's box-score map -> player.stats[week] ->
    the player's own week-specific `projected_points` (box-score players).
    """
    if lookup:
        val = lookup.get(getattr(player, "playerId", None))
        if isinstance(val, (int, float)):
            return val

    stats = getattr(player, "stats", None) or {}
    week_stats = stats.get(week) or stats.get(str(week)) or {}
    val = week_stats.get("projected_points")
    if isinstance(val, (int, float)):
        return val

    val = getattr(player, "projected_points", None)
    if isinstance(val, (int, float)):
        return val

    return None


def get_avg_points(player):
    return getattr(player, "avg_points", None) or getattr(player, "total_points", None)


# ---------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------
def build_standings_df(league: League) -> pd.DataFrame:
    rows = []
    for team in league.teams:
        rows.append(
            {
                "Team": team.team_name,
                "Owner": team.owners[0].get("firstName", "") if team.owners else "",
                "Wins": team.wins,
                "Losses": team.losses,
                "Ties": getattr(team, "ties", 0),
                "Points For": round(team.points_for, 1),
                "Points Against": round(team.points_against, 1),
                "Streak": f"{team.streak_type} {team.streak_length}"
                if getattr(team, "streak_type", None)
                else "",
            }
        )
    df = pd.DataFrame(rows).sort_values(
        by=["Wins", "Points For"], ascending=[False, False]
    )
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df.reset_index(drop=True)


# Red (fewest wins) -> green (most wins), with strong mid-tones so
# neighbouring win totals are easy to tell apart.
WIN_COLORSCALE = [
    [0.00, "#d62728"],
    [0.25, "#ff8c1a"],
    [0.50, "#f2d600"],
    [0.75, "#8cc63f"],
    [1.00, "#1a9850"],
]


def _streak_len(text) -> int:
    m = re.match(r"\s*WIN\s+(\d+)", str(text).upper())
    return int(m.group(1)) if m else 0


def standings_summary(df: pd.DataFrame) -> list:
    """Four headline numbers for the Standings tab."""
    lead = df.iloc[0]
    record = f'{int(lead["Wins"])}-{int(lead["Losses"])}' + (f'-{int(lead["Ties"])}' if lead["Ties"] else "")
    pts = df.loc[df["Points For"].idxmax()]
    best_def = df.loc[df["Points Against"].idxmin()]
    streaks = df["Streak"].map(_streak_len)
    if streaks.max() > 0:
        hot = df.loc[streaks.idxmax()]
        hot_item = ("Longest Win Streak", str(hot["Team"]), f"{int(streaks.max())} straight")
    else:
        hot_item = ("Longest Win Streak", "—", None)
    return [
        ("League Leader", str(lead["Team"]), record),
        ("Most Points", str(pts["Team"]), f'{pts["Points For"]:.1f} pts'),
        ("Fewest Points Against", str(best_def["Team"]), f'{best_def["Points Against"]:.1f} pts'),
        hot_item,
    ]


def power_summary(df: pd.DataFrame) -> list:
    top = df.iloc[0]
    sc = df.loc[df["Avg Score"].idxmax()]
    mov = df.loc[df["Avg MOV"].idxmax()]
    dom = df.loc[df["Dominance"].idxmax()]
    return [
        ("#1 Power Ranked", str(top["Team"]), f'score {top["Power Score"]:.2f}'),
        ("Best Avg Score", str(sc["Team"]), f'{sc["Avg Score"]:.0f} per game'),
        ("Best Avg Margin", str(mov["Team"]), f'{mov["Avg MOV"]:+.0f} per game'),
        ("Most Dominant", str(dom["Team"]), f'{dom["Dominance"]:.0f} dominance'),
    ]


def free_agent_summary(df: pd.DataFrame, week: int) -> list:
    col = f"Week {week} Proj"
    items = [("Free Agents Shown", str(len(df)), None)]
    proj = df[col].dropna() if col in df else pd.Series(dtype=float)
    if len(proj):
        best = df.loc[proj.idxmax()]
        items.append(("Top Projection", str(best["Player"]), f'{best[col]:.1f} pts · {best["Position"]}'))
    else:
        items.append(("Top Projection", "—", None))
    if "% Owned" in df and df["% Owned"].notna().any():
        own = df.loc[df["% Owned"].idxmax()]
        items.append(("Most Owned", str(own["Player"]), f'{own["% Owned"]:.0f}% owned'))
    else:
        items.append(("Most Owned", "—", None))
    flagged = int(df["Injury Status"].isin(NON_ACTIVE_STATUSES).sum()) if "Injury Status" in df else 0
    items.append(("Injury Flags", str(flagged), "of the list"))
    return items


def injury_summary(df: pd.DataFrame) -> list:
    by_team = df["Team"].value_counts()
    worst = by_team.index[0]
    out = int(df["Status"].isin(["OUT", "INJURED RESERVE"]).sum())
    return [
        ("Players Flagged", str(len(df)), None),
        ("Out / Injured Reserve", str(out), None),
        ("Teams Affected", str(df["Team"].nunique()), None),
        ("Hardest Hit", str(worst), f"{int(by_team.iloc[0])} flagged"),
    ]


def build_points_chart(standings_df: pd.DataFrame):
    """Horizontal bars: most points at the top, colored by wins
    (green = most wins, red = fewest)."""
    df = standings_df.sort_values("Points For", ascending=False, kind="stable")
    lo, hi = int(df["Wins"].min()), int(df["Wins"].max())
    if lo == hi:  # everyone tied (e.g. before the first game): avoid a flat scale
        lo, hi = lo - 1, hi + 1

    fig = px.bar(
        df,
        x="Points For",
        y="Team",
        orientation="h",
        color="Wins",
        color_continuous_scale=WIN_COLORSCALE,
        range_color=(lo, hi),
        category_orders={"Team": list(df["Team"])},
        hover_data={"Wins": True, "Losses": True, "Points For": ":.1f"},
        title="Points For by Team",
    )
    fig.update_traces(
        texttemplate="%{x:.1f}", textposition="outside", cliponaxis=False
    )
    fig.update_yaxes(title=None)  # px already lists the first category at the top
    fig.update_xaxes(range=[0, float(df["Points For"].max()) * 1.12 or 1])
    fig.update_layout(
        height=max(320, 42 * len(df) + 110),
        margin=dict(l=0, r=10, t=50, b=10),
        coloraxis_colorbar=dict(title="Wins", tickmode="linear", tick0=lo, dtick=1),
    )
    return style_fig(fig)


def build_roster_df(team, week: int, lookup: dict) -> pd.DataFrame:
    """Roster in ESPN's slot order (QB, RB, WR, TE, FLEX, DP, K, Bench, IR).
    Players keep ESPN's own order within a slot."""
    week_col = f"Week {week} Proj"
    rows = []
    for player in team.roster:
        rows.append(
            {
                "_slot_rank": slot_rank(player.lineupSlot),
                "Slot": SLOT_LABELS.get(player.lineupSlot, player.lineupSlot),
                "Player": player.name,
                "Position": player.position,
                "Pro Team": player.proTeam,
                week_col: _safe_round(get_week_proj_points(player, week, lookup)),
                "Avg Points": _safe_round(get_avg_points(player)),
                "Season Proj": _safe_round(get_season_proj_points(player)),
                "Injury Status": normalize_status(getattr(player, "injuryStatus", None)),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = (
            df.sort_values(by="_slot_rank", kind="stable")
            .drop(columns="_slot_rank")
            .reset_index(drop=True)
        )
    return df


GRAY = "#9aa0a6"


def build_matchup_rows(league: League, week: int) -> list:
    """One dict per game: each side has team, score and ESPN's projected score."""
    def side(team, score, projected):
        if team is None:
            return {"team": "BYE", "score": None, "proj": None}
        return {"team": team.team_name, "score": score, "proj": projected}

    return [
        {
            "away": side(box.away_team, box.away_score, box.away_projected),
            "home": side(box.home_team, box.home_score, box.home_projected),
        }
        for box in league.box_scores(week=week)
    ]


def matchup_summary(rows: list) -> list:
    """Headline numbers for a week's games (scores show a dash until played)."""
    sides = [s for r in rows for s in (r["away"], r["home"]) if s["score"] is not None]
    games = [r for r in rows if r["away"]["score"] is not None and r["home"]["score"] is not None]
    played = [r for r in games if r["away"]["score"] or r["home"]["score"]]
    items = [("Games", str(len(rows)), None)]
    if played:
        top = max((s for s in sides), key=lambda s: s["score"])
        close = min(played, key=lambda r: abs(r["away"]["score"] - r["home"]["score"]))
        items.append(("Highest Score", str(top["team"]), f'{top["score"]:.1f}'))
        items.append(("Closest Game", f'{close["away"]["team"]} vs {close["home"]["team"]}',
                      f'by {abs(close["away"]["score"] - close["home"]["score"]):.1f}'))
    else:
        items.append(("Highest Score", "—", "not played yet"))
        items.append(("Closest Game", "—", "not played yet"))
    projs = [s for s in sides if isinstance(s["proj"], (int, float))]
    if projs:
        hp = max(projs, key=lambda s: s["proj"])
        items.append(("Highest Projection", str(hp["team"]), f'{hp["proj"]:.1f}'))
    else:
        items.append(("Highest Projection", "—", None))
    return items


def matchups_html(rows: list, bar: bool = False) -> str:
    """Scoreboard cards. Each side reads "(projected) actual" with the projection
    in lighter gray to the left of the actual score; the leading side is
    highlighted. bar=True adds a projection-split bar under each game.
    Team names are escaped."""

    def side_row(tag, side, lead):
        if side["score"] is None:
            score_html = "—"
        else:
            proj = ""
            if isinstance(side["proj"], (int, float)):
                proj = (
                    f'<span style="color:{GRAY};font-weight:400;margin-right:0.6em;">'
                    f'({side["proj"]:.1f})</span>'
                )
            score_html = f'{proj}{side["score"]:.1f}'
        cls = "sb-row lead" if lead else "sb-row"
        return (
            f'<div class="{cls}"><span class="sb-team"><span class="sb-tag">{tag}</span>'
            f'{html.escape(str(side["team"]))}</span>'
            f'<span class="sb-score">{score_html}</span></div>'
        )

    cards = []
    for r in rows:
        a, h = r["away"], r["home"]
        lead = None
        if a["score"] is not None and h["score"] is not None and (a["score"] or h["score"]):
            if a["score"] > h["score"]:
                lead = "away"
            elif h["score"] > a["score"]:
                lead = "home"
        split = ""
        if bar and isinstance(a["proj"], (int, float)) and isinstance(h["proj"], (int, float)) \
                and (a["proj"] + h["proj"]) > 0:
            pa = a["proj"] / (a["proj"] + h["proj"]) * 100
            split = (
                '<div class="sb-split"><div class="bar-track" style="display:flex;">'
                f'<div style="width:{pa:.1f}%;background:{ACCENT};"></div>'
                f'<div style="width:{100 - pa:.1f}%;background:{GRAY};opacity:0.55;"></div></div>'
                f'<div class="sb-split-cap"><span>{pa:.0f}% projected share</span>'
                f"<span>{100 - pa:.0f}%</span></div></div>"
            )
        cards.append(
            '<div class="sb-card">'
            + side_row("AWAY", a, lead == "away")
            + side_row("HOME", h, lead == "home")
            + split
            + "</div>"
        )
    return f'<div class="sb-grid">{"".join(cards)}</div>'


POWER_WEIGHTS = {"dominance": 0.80, "avg_score": 0.15, "avg_mov": 0.05}

POWER_FOOTNOTE = (
    "**Formula:** Power Score = 0.80 × Dominance + 0.15 × Avg Score + 0.05 × Avg MOV"
    "\n\n"
    "**Dominance** = Direct Wins + Indirect Wins. *Direct Wins* are the matchups a "
    "team has won through the selected week. *Indirect Wins*: each time a team "
    "beat an opponent, that opponent's own win total is added."
    "\n\n"
    "**Avg Score** is average points scored per week and **Avg MOV** is average "
    "margin of victory (negative means the team loses by that much on average). "
    "The formula drops the decimals from both, so the whole numbers it actually "
    "uses are shown."
    "\n\n"
    "Computed by the espn_api library through the selected week; it may differ "
    "from the power rankings on ESPN's own site."
)


def build_power_rankings_df(league: League, week: int) -> pd.DataFrame:
    """
    Power rankings with every input of the formula as its own column.
    The Power Score itself comes straight from the library, and the
    columns are recomputed from the same raw data; df.attrs["formula_matches"]
    records whether those columns reproduce the library's score.
    """
    try:
        rankings = league.power_rankings(week=week)
    except Exception:
        return pd.DataFrame()

    wk = week if 0 < week <= league.current_week else league.current_week
    teams = sorted(league.teams, key=lambda t: t.team_id)
    n = len(teams)
    pos = {t.team_id: i for i, t in enumerate(teams)}

    wins = [[0] * n for _ in range(n)]  # wins[i][j] = times team i beat team j
    for i, team in enumerate(teams):
        for mov, opp in zip(team.mov[:wk], team.schedule[:wk]):
            if mov > 0:
                wins[i][pos[opp.team_id]] += 1
    direct = [sum(row) for row in wins]
    indirect = [
        sum(wins[i][k] * wins[k][j] for k in range(n) for j in range(n))
        for i in range(n)
    ]

    rows, matches = [], True
    for rank, (score, team) in enumerate(rankings, start=1):
        i = pos[team.team_id]
        dominance = direct[i] + indirect[i]
        avg_score = int(sum(team.scores[:wk]) / wk)
        avg_mov = int(sum(team.mov[:wk]) / wk)
        recomputed = (
            dominance * POWER_WEIGHTS["dominance"]
            + avg_score * POWER_WEIGHTS["avg_score"]
            + avg_mov * POWER_WEIGHTS["avg_mov"]
        )
        if abs(recomputed - float(score)) > 0.011:
            matches = False
        rows.append(
            {
                "Rank": rank,
                "Team": team.team_name,
                "Direct Wins": direct[i],
                "Indirect Wins": indirect[i],
                "Dominance": dominance,
                "Avg Score": avg_score,
                "Avg MOV": avg_mov,
                "Power Score": float(score),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["formula_matches"] = matches
    return df


# ---------------------------------------------------------------------
# Lineup optimizer
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """A player being considered for a starting slot (rostered or free agent)."""
    key: object
    name: str
    position: str
    eligible: frozenset          # lineup slots this player is allowed to fill
    proj: float                  # projection for the selected week (0.0 if none)
    has_proj: bool = True
    status: str = "ACTIVE"
    current_slot: object = None  # None for free agents
    percent_owned: object = None
    is_fa: bool = False

    @property
    def label(self) -> str:
        return f"{self.name} ({self.position})"


def _player_key(p):
    pid = getattr(p, "playerId", None)
    return pid if pid is not None else p.name


def _confidence(gap: float) -> str:
    if gap >= 4:
        return "🟢 Strong"
    if gap >= 1.5:
        return "🟡 Moderate"
    return "🟠 Close call"


def roster_candidates(team, week: int, lookup: dict) -> list:
    out = []
    for p in team.roster:
        proj = get_week_proj_points(p, week, lookup)
        out.append(
            Candidate(
                key=_player_key(p),
                name=p.name,
                position=p.position,
                eligible=frozenset(getattr(p, "eligibleSlots", None) or []),
                proj=proj if proj is not None else 0.0,
                has_proj=proj is not None,
                status=normalize_status(getattr(p, "injuryStatus", None)),
                current_slot=p.lineupSlot,
            )
        )
    return out


def starting_slots(league: League) -> list:
    """One entry per starting spot, e.g. ["QB", "RB", "RB", "WR", ..., "DP", "K"]."""
    counts = dict(getattr(league.settings, "position_slot_counts", None) or {})
    slots = []
    for name, n in counts.items():
        if name not in ("BE", "IR"):
            slots.extend([name] * int(n))
    return slots


_BIG = 1e6          # cost of an illegal slot/player pairing
_KEEP_BONUS = 1e-6  # tie-break: keep players already in the lineup


def best_lineup(slots: list, candidates: list, prefer=frozenset()) -> list:
    """
    Highest-scoring way to fill the starting slots, as [(slot, Candidate)].

    This is an assignment problem solved exactly (not greedily): a player can
    only fill a slot he is eligible for (a QB can never take an RB slot) and
    each player is used at most once. FLEX-style slots are handled correctly
    because eligibility is checked per slot. Slots nobody is eligible for stay
    empty. `prefer` lists keys of players to keep when projections tie, so the
    tool doesn't suggest pointless swaps.
    """
    if not slots or not candidates:
        return []
    cost = np.full((len(slots), len(candidates)), _BIG)
    for i, slot in enumerate(slots):
        for j, c in enumerate(candidates):
            if slot in c.eligible:
                cost[i, j] = -(c.proj + (_KEEP_BONUS if c.key in prefer else 0.0))
    rows, cols = linear_sum_assignment(cost)
    return [
        (slots[i], candidates[j])
        for i, j in zip(rows, cols)
        if cost[i, j] < _BIG / 2
    ]


def pair_swaps(ins: list, outs: list) -> list:
    """Match each player coming into the lineup with one going out, preferring
    the same position, then a shared lineup slot, so each row reads sensibly."""
    if not ins or not outs:
        return [(i, None) for i in ins] + [(None, o) for o in outs]
    score = np.zeros((len(ins), len(outs)))
    for a, i in enumerate(ins):
        for b, o in enumerate(outs):
            if i.position == o.position:
                score[a, b] = 100
            elif (i.eligible & o.eligible) - {"BE", "IR"}:
                score[a, b] = 10
    rows, cols = linear_sum_assignment(-score)
    matched = dict(zip(rows, cols))
    pairs = [(i, outs[matched[a]] if a in matched else None) for a, i in enumerate(ins)]
    pairs += [(None, o) for b, o in enumerate(outs) if b not in set(matched.values())]
    pairs.sort(key=lambda p: -((p[0].proj if p[0] else 0.0) - (p[1].proj if p[1] else 0.0)))
    return pairs


def free_agent_upgrades(slots, roster, free_agents, lineup, week, max_pickups=5, min_gain=0.05):
    """
    Pickups that raise the projected lineup total, best first.

    Each step finds the single free agent whose addition raises the optimal
    lineup the most, then adds him and repeats. Because every candidate is
    scored by re-solving the constrained lineup, a pickup only helps if he can
    legally fill a slot: a QB can bump a QB (or fill an OP slot if the league
    has one) but never an RB. Returns (rows, total_with_pickups).
    """
    pool = list(roster)
    total = sum(c.proj for _, c in lineup)
    remaining = [fa for fa in free_agents if fa.has_proj and fa.status not in UNAVAILABLE_STATUSES]
    rows = []

    for _ in range(max_pickups):
        keys = {c.key for _, c in lineup}
        full = len(lineup) == len(slots)
        # A free agent can't beat the weakest player in a full lineup unless he
        # projects higher than that player, so skip the rest cheaply.
        floor = min((c.proj for _, c in lineup), default=0.0) if full else float("-inf")

        best = None
        for fa in remaining:
            if fa.proj - floor <= min_gain:
                continue
            trial = best_lineup(slots, pool + [fa], prefer=keys)
            gain = sum(c.proj for _, c in trial) - total
            if best is None or gain > best[0]:
                best = (gain, fa, trial)
        if best is None or best[0] < min_gain:
            break

        gain, fa, trial = best
        trial_keys = {c.key for _, c in trial}
        bumped = sorted((c for _, c in lineup if c.key not in trial_keys), key=lambda c: c.proj)
        out = bumped[0] if bumped else None
        slot = next(s for s, c in trial if c.key == fa.key)

        owned = fa.percent_owned if isinstance(fa.percent_owned, (int, float)) and fa.percent_owned >= 0 else None
        rows.append(
            {
                "Add (Free Agent)": fa.label,
                f"Wk {week} Proj": _safe_round(fa.proj),
                "Injury Status": fa.status,
                "Starts At": SLOT_LABELS.get(slot, slot),
                "Replaces in Lineup": out.label if out else "— (open slot)",
                f"Wk {week} Proj (out)": _safe_round(out.proj) if out else None,
                "Gain": _safe_round(gain),
                "Confidence": _confidence(gain),
                "% Owned": _safe_round(owned),
            }
        )
        pool.append(fa)
        remaining = [r for r in remaining if r.key != fa.key]
        lineup, total = trial, total + gain

    return rows, total


@dataclass
class LineupResult:
    current_total: float
    optimal_total: float
    swaps_df: pd.DataFrame
    missing: list
    fa_total: object  # None when free agents weren't considered
    fa_df: pd.DataFrame


def optimize_lineup(league: League, team, week: int, lookup: dict, free_agents=None) -> LineupResult:
    """
    Best lineup for one week using ESPN's WEEKLY projections (never the
    season-long total), respecting every slot's eligibility. Compares it with
    the current lineup, lists the roster swaps, and (when `free_agents` is
    given) finds free-agent pickups that would raise the total further.

    Players with no weekly projection count as 0.0 so they're never promoted
    on a guess; their names are returned in `missing` so the UI can flag them.
    """
    slots = starting_slots(league)
    roster = roster_candidates(team, week, lookup)

    current = [c for c in roster if c.current_slot not in (None, "BE", "IR")]
    current_keys = {c.key for c in current}
    current_total = sum(c.proj for c in current)

    optimal = best_lineup(slots, roster, prefer=current_keys)
    optimal_total = sum(c.proj for _, c in optimal)
    optimal_keys = {c.key for _, c in optimal}

    ins = [c for _, c in optimal if c.key not in current_keys]
    outs = [c for c in current if c.key not in optimal_keys]

    swap_rows = []
    for c_in, c_out in pair_swaps(ins, outs):
        gap = (c_in.proj if c_in else 0.0) - (c_out.proj if c_out else 0.0)
        swap_rows.append(
            {
                "Bench → Start": c_in.label if c_in else "",
                f"Wk {week} Proj (in)": _safe_round(c_in.proj) if c_in else None,
                "Start → Bench": c_out.label if c_out else "",
                f"Wk {week} Proj (out)": _safe_round(c_out.proj) if c_out else None,
                "Confidence": _confidence(gap),
            }
        )

    # Only players who matter to the decision need a weekly number.
    relevant = {c.key: c for c in current}
    relevant.update({c.key: c for _, c in optimal})
    missing = sorted(c.name for c in relevant.values() if not c.has_proj)

    fa_rows, fa_total = [], None
    if free_agents is not None:
        fa_rows, fa_total = free_agent_upgrades(slots, roster, free_agents, optimal, week)

    return LineupResult(
        current_total=current_total,
        optimal_total=optimal_total,
        swaps_df=pd.DataFrame(swap_rows),
        missing=missing,
        fa_total=fa_total,
        fa_df=pd.DataFrame(fa_rows),
    )


# Which free-agent pools to pull for each kind of lineup slot (ESPN's
# free-agent search filters by a single position at a time).
SLOT_FA_POSITIONS = {
    "RB/WR/TE": ("RB", "WR", "TE"),
    "FLEX": ("RB", "WR", "TE"),
    "RB/WR": ("RB", "WR"),
    "WR/TE": ("WR", "TE"),
    "OP": ("QB", "RB", "WR", "TE"),
}


def free_agent_positions(slots: list) -> tuple:
    out = []
    for slot in dict.fromkeys(slots):
        for pos in SLOT_FA_POSITIONS.get(slot, (slot,)):
            if pos not in out:
                out.append(pos)
    return tuple(out)


@st.cache_data(ttl=LEAGUE_CACHE_TTL_SECONDS, show_spinner="Loading free agents...")
def get_free_agent_pool(_league, league_id: int, year: int, week: int, positions: tuple, per_position: int):
    """
    Free agents (with that week's projection) for each position the lineup can
    use. Returns (list of plain dicts, positions that failed to load); plain
    dicts keep the cached value small and picklable. league_id/year are only
    cache keys.
    """
    pool, failed, seen = [], [], set()
    for pos in positions:
        try:
            players = _league.free_agents(week=week, size=per_position, position=pos)
        except Exception:
            failed.append(pos)
            continue
        for p in players:
            key = _player_key(p)
            if key in seen:
                continue
            seen.add(key)
            proj = get_week_proj_points(p, week)
            if proj is None:
                continue
            pool.append(
                {
                    "key": key,
                    "name": p.name,
                    "position": p.position,
                    "eligible": list(getattr(p, "eligibleSlots", None) or []),
                    "proj": proj,
                    "status": normalize_status(getattr(p, "injuryStatus", None)),
                    "percent_owned": getattr(p, "percent_owned", None),
                }
            )
    return pool, failed


def free_agent_candidates(pool: list) -> list:
    return [
        Candidate(
            key=d["key"],
            name=d["name"],
            position=d["position"],
            eligible=frozenset(d["eligible"]),
            proj=d["proj"],
            status=d["status"],
            percent_owned=d["percent_owned"],
            is_fa=True,
        )
        for d in pool
    ]


# ---------------------------------------------------------------------
# Free agents
# ---------------------------------------------------------------------
def build_free_agents_df(
    league: League, week: int, position: str, size: int = FREE_AGENT_LIST_SIZE
) -> pd.DataFrame:
    pos_arg = None if position == "All" else position
    try:
        players = league.free_agents(week=week, size=size, position=pos_arg)
    except Exception:
        players = league.free_agents(size=size)
        if pos_arg:
            players = [
                p for p in players
                if pos_arg in (getattr(p, "eligibleSlots", None) or [p.position])
            ]

    week_col = f"Week {week} Proj"
    rows = []
    for player in players:
        rows.append(
            {
                "Player": player.name,
                "Position": player.position,
                "Pro Team": player.proTeam,
                week_col: _safe_round(get_week_proj_points(player, week)),
                "Avg Points": _safe_round(get_avg_points(player)),
                "Season Proj": _safe_round(get_season_proj_points(player)),
                "% Owned": _safe_round(getattr(player, "percent_owned", None)),
                "Injury Status": normalize_status(getattr(player, "injuryStatus", None)),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=week_col, ascending=False, na_position="last").reset_index(
            drop=True
        )
    return df


# ---------------------------------------------------------------------
# Injury report
# ---------------------------------------------------------------------
def build_injury_report_df(league: League, week: int, lookup: dict) -> pd.DataFrame:
    rows = []
    for team in league.teams:
        for player in team.roster:
            status = normalize_status(getattr(player, "injuryStatus", None))
            if status in NON_ACTIVE_STATUSES:
                rows.append(
                    {
                        "Team": team.team_name,
                        "Player": player.name,
                        "Position": player.position,
                        "Status": status,
                        f"Week {week} Proj": _safe_round(
                            get_week_proj_points(player, week, lookup)
                        ),
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = sort_injury_report(df)
    return df


def sort_injury_report(df: pd.DataFrame) -> pd.DataFrame:
    """Team name A-Z, then least severe status first, most severe last."""
    df = df.assign(
        _team=df["Team"].str.casefold(),
        _sev=df["Status"].map(STATUS_SEVERITY).fillna(len(STATUS_SEVERITY) + 1),
        _player=df["Player"].str.casefold(),
    )
    df = df.sort_values(by=["_team", "_sev", "_player"], kind="mergesort")
    return df.drop(columns=["_team", "_sev", "_player"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Trade analyzer
# ---------------------------------------------------------------------
def get_all_rostered_players(league: League):
    """Returns a list of (player, team) across every roster in the league."""
    pairs = []
    for team in league.teams:
        for player in team.roster:
            pairs.append((player, team))
    return pairs


def trade_side_options(all_pairs: list, selected: list, other_team_id=None) -> tuple:
    """Which players one side of a trade can pick from.

    Before anything is picked, every rostered player is available (except the
    other side's team, once that side is locked). After the first pick, the side
    is locked to that player's fantasy team. Returns (player keys, team_id|None).
    """
    team_of = {_player_key(p): t.team_id for p, t in all_pairs}
    locked = next((team_of[k] for k in selected if k in team_of), None)
    if locked is not None:
        keys = [_player_key(p) for p, t in all_pairs if t.team_id == locked]
    else:
        keys = [_player_key(p) for p, t in all_pairs if t.team_id != other_team_id]
    return keys, locked


def clean_trade_selection(all_pairs: list, sel_a: list, sel_b: list) -> tuple:
    """Drops stale picks so each side only holds players from one team, and the
    two sides never hold the same team. Side A wins a conflict."""
    team_of = {_player_key(p): t.team_id for p, t in all_pairs}

    def one_team(sel, banned=None):
        sel = [k for k in sel if k in team_of and team_of[k] != banned]
        if not sel:
            return []
        return [k for k in sel if team_of[k] == team_of[sel[0]]]

    a = one_team(sel_a)
    lock_a = team_of[a[0]] if a else None
    return a, one_team(sel_b, banned=lock_a)


def build_player_compare_df(players_with_teams, week: int, lookup: dict) -> pd.DataFrame:
    rows = []
    for player, team in players_with_teams:
        rows.append(
            {
                "Player": player.name,
                "Team (NFL)": player.proTeam,
                "Fantasy Owner": team.team_name,
                "Position": player.position,
                f"Week {week} Proj": _safe_round(
                    get_week_proj_points(player, week, lookup)
                ),
                "Avg Points": _safe_round(get_avg_points(player)),
                "Season Proj": _safe_round(get_season_proj_points(player)),
                "Injury Status": normalize_status(getattr(player, "injuryStatus", None)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Weekly recap (actual vs. projected)
# ---------------------------------------------------------------------
def build_weekly_recap_df(league: League, week: int, boxes=None) -> pd.DataFrame:
    """
    Compares each starter's actual points to their pre-game projection
    for a completed week, across the whole league, so you can see who
    beat or missed their number by the widest margin.
    """
    rows = []
    for box in (boxes if boxes is not None else league.box_scores(week=week)):
        for team, lineup in (
            (box.home_team, box.home_lineup),
            (box.away_team, box.away_lineup),
        ):
            if team is None:
                continue
            for player in lineup:
                if player.slot_position in ("BE", "IR"):
                    continue
                proj = get_week_proj_points(player, week)
                actual = getattr(player, "points", None)
                if actual is None or proj is None:
                    continue
                rows.append(
                    {
                        "Team": team.team_name,
                        "Player": player.name,
                        "Position": player.position,
                        "Weekly Proj": _safe_round(proj),
                        "Actual Points": _safe_round(actual),
                        "Diff (Actual − Proj)": _safe_round(actual - proj),
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="Diff (Actual − Proj)", ascending=False).reset_index(
            drop=True
        )
    return df


# ---------------------------------------------------------------------
# Visual kit: theme CSS, pills, stat cards, HTML tables
# ---------------------------------------------------------------------
ACCENT = "#38bdf8"
GOLD, SILVER, BRONZE = "#f5c542", "#b8c2cc", "#cd7f32"
PALETTE = [  # distinct hues that read on both dark and light backgrounds
    "#38bdf8", "#f97316", "#a78bfa", "#34d399", "#f43f5e", "#facc15",
    "#2dd4bf", "#fb7185", "#818cf8", "#a3e635", "#f59e0b", "#60a5fa",
    "#e879f9", "#4ade80", "#fb923c", "#22d3ee",
]

APP_CSS = """
<style>
.block-container { padding-top: 4rem; max-width: 1400px; }
[data-testid="stMetric"] {
  background: rgba(128,128,128,0.10); border: 1px solid rgba(128,128,128,0.25);
  border-radius: 14px; padding: 0.85rem 1.1rem;
}
[data-testid="stMetricLabel"] { opacity: 0.75; }
[data-testid="stMetricValue"] { font-weight: 700; font-size: 1.55rem; }
[data-testid="stMetricValue"] > div { white-space: normal; text-overflow: clip; line-height: 1.2; }
.stTabs [role="tablist"] { gap: 0.35rem; flex-wrap: wrap; }
.stTabs [role="tab"] { border-radius: 10px 10px 0 0; padding: 0.5rem 1rem; }
h2, h3 { letter-spacing: -0.01em; }

.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 0.8rem; margin: 0.4rem 0 0.2rem; }
.stat { background: rgba(128,128,128,0.10); border: 1px solid rgba(128,128,128,0.25);
  border-radius: 14px; padding: 0.8rem 1.1rem; }
.stat-label { font-size: 0.8rem; opacity: 0.7; }
.stat-value { font-size: 1.45rem; font-weight: 700; line-height: 1.25; margin-top: 0.1rem; overflow-wrap: anywhere; }
.stat-sub { font-size: 0.82rem; opacity: 0.65; margin-top: 0.15rem; }
.hero { border-radius: 18px; padding: 1.3rem 1.6rem; margin-bottom: 1rem;
  background: linear-gradient(120deg, rgba(56,189,248,0.22), rgba(52,211,153,0.14));
  border: 1px solid rgba(128,128,128,0.25); }
.hero-title { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em; }
.hero-sub { opacity: 0.75; margin-top: 0.15rem; }

.ft-wrap { border: 1px solid rgba(128,128,128,0.25); border-radius: 14px; overflow: auto; }
.ft { width: 100%; border-collapse: collapse; font-size: 0.93rem; }
.ft th { position: sticky; top: 0; text-align: left; font-weight: 700; font-size: 0.78rem;
  text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.8; padding: 0.65rem 0.85rem;
  background: rgba(128,128,128,0.18); backdrop-filter: blur(6px); white-space: nowrap; }
.ft td { padding: 0.55rem 0.85rem; border-top: 1px solid rgba(128,128,128,0.15); vertical-align: middle; }
.ft tr:hover td { background: rgba(128,128,128,0.08); }
.ft .num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.ft .mark td { padding: 0.15rem 0.85rem; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: #38bdf8; border-top: 2px dashed rgba(56,189,248,0.7); background: rgba(56,189,248,0.07); }
.bar-cell { display: flex; align-items: center; gap: 0.5rem; min-width: 105px; }
.bar-cell .v { min-width: 3.2rem; text-align: right; font-variant-numeric: tabular-nums; }
.bar-track { flex: 1; height: 8px; border-radius: 999px; background: rgba(128,128,128,0.22); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; }

.sb-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 0.9rem; }
.sb-card { border: 1px solid rgba(128,128,128,0.28); border-radius: 16px; padding: 0.55rem 0.4rem;
  background: rgba(128,128,128,0.07); }
.sb-row { display: flex; justify-content: space-between; align-items: center; gap: 0.8rem;
  padding: 0.55rem 0.9rem; border-left: 4px solid transparent; border-radius: 10px; }
.sb-row.lead { border-left-color: #34d399; background: rgba(52,211,153,0.10); }
.sb-team { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sb-tag { font-size: 0.64rem; font-weight: 700; letter-spacing: 0.08em; opacity: 0.55; margin-right: 0.5rem; }
.sb-score { font-size: 1.35rem; font-weight: 800; font-variant-numeric: tabular-nums; white-space: nowrap; }
.sb-row:not(.lead) .sb-score { font-weight: 600; opacity: 0.9; }
.sb-split { margin: 0.35rem 0.9rem 0.25rem; }
.sb-split .bar-track { height: 6px; }
.sb-split-cap { font-size: 0.7rem; opacity: 0.6; display: flex; justify-content: space-between; margin-top: 0.2rem; }
</style>
"""


def inject_css():
    st.markdown(APP_CSS, unsafe_allow_html=True)


def _rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def pill(text, color: str) -> str:
    """A solid rounded badge. Text flips between white and near-black so it
    stays readable on both light and dark badge colors."""
    r, g, b = _rgb(color)
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
    fg = "#0b1220" if lum > 0.5 else "#ffffff"
    return (
        '<span style="display:inline-block;padding:0.12rem 0.7rem;border-radius:999px;'
        "font-size:0.76rem;font-weight:700;letter-spacing:0.03em;white-space:nowrap;"
        f'background:{color};color:{fg};">{html.escape(str(text))}</span>'
    )


def status_pill(value) -> str:
    status = normalize_status(value)
    color = STATUS_COLORS.get(status)
    return pill(status, color) if color else html.escape(status)


def streak_pill(value) -> str:
    text = str(value or "")
    up = text.upper()
    if up.startswith("WIN"):
        return pill(text, WIN_COLOR)
    if up.startswith("LOSS"):
        return pill(text, LOSS_COLOR)
    return html.escape(text) if text else "—"


def stat_cards(items: list):
    """A row of headline tiles. items = [(label, value, small_text_or_None), ...]"""
    tiles = "".join(
        '<div class="stat">'
        f'<div class="stat-label">{html.escape(str(label))}</div>'
        f'<div class="stat-value">{html.escape(str(value))}</div>'
        + (f'<div class="stat-sub">{html.escape(str(sub))}</div>' if sub else "")
        + "</div>"
        for label, value, sub in items
    )
    st.html(f'<div class="stat-grid">{tiles}</div>')


def hero(league_name: str, year, week) -> str:
    return (
        '<div class="hero">'
        f'<div class="hero-title">🏈 {html.escape(str(league_name))}</div>'
        f'<div class="hero-sub">{html.escape(str(year))} season · Week {html.escape(str(week))}</div>'
        "</div>"
    )


def _fmt_cell(v, precision=1) -> tuple:
    """(text, is_numeric)"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—", False
    if isinstance(v, (bool, np.bool_)):
        return str(v), False
    if isinstance(v, (int, np.integer)):
        return str(int(v)), True
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{precision}f}", True
    return html.escape(str(v)), False


def bar_cell(value, vmax, color=ACCENT, precision=1, vmin=0.0) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    span = float(vmax) - float(vmin)
    pct = 0 if span <= 0 else max(0.0, min(100.0, (float(value) - float(vmin)) / span * 100))
    if vmin:  # keep the smallest value visible
        pct = 12 + pct * 0.88
    return (
        '<div class="bar-cell">'
        f'<span class="v">{float(value):.{precision}f}</span>'
        '<div class="bar-track">'
        f'<div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div></div></div>'
    )


def html_table(df, pills=None, bars=None, precision=1, max_height=None, marks=None, raw_cols=()) -> str:
    """Render a DataFrame as a styled HTML table.

    pills:    {column: fn(value) -> html} e.g. {"Injury Status": status_pill}
    bars:     {column: (max_value, color[, min_value])} draws a number plus a horizontal bar
    marks:    {row_index: label} inserts a labelled divider line after that row
    raw_cols: columns whose values are already HTML (you escaped them)
    Everything else is escaped.
    """
    pills, bars, marks = pills or {}, bars or {}, marks or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for i, (_, row) in enumerate(df.iterrows()):
        cells = []
        for c in df.columns:
            v = row[c]
            if c in pills:
                cells.append(f"<td>{pills[c](v)}</td>")
            elif c in bars:
                spec = bars[c]
                cells.append(f"<td>{bar_cell(v, spec[0], spec[1], precision, spec[2] if len(spec) > 2 else 0.0)}</td>")
            elif c in raw_cols:
                cells.append(f"<td>{v}</td>")
            else:
                text, numeric = _fmt_cell(v, precision)
                cells.append(f'<td class="num">{text}</td>' if numeric else f"<td>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
        if i in marks:
            body.append(f'<tr class="mark"><td colspan="{len(df.columns)}">{html.escape(marks[i])}</td></tr>')
    style = f' style="max-height:{int(max_height)}px;"' if max_height else ""
    return (
        f'<div class="ft-wrap"{style}><table class="ft"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def show_table(df, **kw):
    st.html(html_table(df, **kw))


def style_fig(fig):
    """Shared chart polish; colors/fonts follow the app theme via Streamlit."""
    fig.update_layout(
        font=dict(size=13),
        hoverlabel=dict(font_size=13),
    )
    return fig



@dataclass
class LeagueRecap:
    week: int
    completed: bool = False
    highest: dict | None = None      # {"team", "score"}
    lowest: dict | None = None
    narrowest: dict | None = None    # {"winner", "loser", "winner_score", "loser_score", "margin"}
    widest: dict | None = None
    luckiest: dict | None = None     # a row of table below
    unluckiest: dict | None = None
    table: pd.DataFrame | None = None
    upsets: list | None = None
    records: list | None = None


def _real_team(team) -> bool:
    return team is not None and hasattr(team, "team_name")


def build_league_recap(league: League, week: int, boxes=None, history_loader=None) -> LeagueRecap:
    """League-wide story of one completed week: top and bottom scores, closest and
    widest wins, luck (actual result vs. all-play record), projection upsets and
    any season / league-history scoring records."""
    boxes = boxes if boxes is not None else league.box_scores(week=week)
    games = []
    for box in boxes:
        if not (_real_team(box.home_team) and _real_team(box.away_team)):
            continue  # bye weeks
        games.append(box)
    rec = LeagueRecap(week=week, upsets=[], records=[])
    if not games or all(not (b.home_score or b.away_score) for b in games):
        return rec
    rec.completed = True

    scores = []  # (team_name, score, team)
    for b in games:
        scores.append((b.home_team.team_name, float(b.home_score), b.home_team))
        scores.append((b.away_team.team_name, float(b.away_score), b.away_team))

    top = max(scores, key=lambda x: x[1])
    bottom = min(scores, key=lambda x: x[1])
    rec.highest = {"team": top[0], "score": top[1]}
    rec.lowest = {"team": bottom[0], "score": bottom[1]}

    # Margins of victory (ties excluded) and result per team
    result = {}
    wins = []
    for b in games:
        h, a = float(b.home_score), float(b.away_score)
        if h > a:
            w, l, ws, ls = b.home_team, b.away_team, h, a
        elif a > h:
            w, l, ws, ls = b.away_team, b.home_team, a, h
        else:
            result[b.home_team.team_name] = result[b.away_team.team_name] = 0.5
            continue
        result[w.team_name], result[l.team_name] = 1.0, 0.0
        wins.append({"winner": w.team_name, "loser": l.team_name,
                     "winner_score": ws, "loser_score": ls, "margin": ws - ls})
    if wins:
        rec.narrowest = min(wins, key=lambda g: g["margin"])
        rec.widest = max(wins, key=lambda g: g["margin"])

    # All-play record: how each team's score would have done against everyone
    rows = []
    n = len(scores)
    opp_of = {}
    for b in games:
        opp_of[b.home_team.team_name] = b.away_team.team_name
        opp_of[b.away_team.team_name] = b.home_team.team_name
    for name, sc, _ in scores:
        w = sum(1 for o, s2, _ in scores if o != name and sc > s2)
        t = sum(1 for o, s2, _ in scores if o != name and sc == s2)
        l = (n - 1) - w - t
        ap_pct = (w + 0.5 * t) / (n - 1) if n > 1 else 0.0
        actual = result.get(name, 0.0)
        rows.append({
            "Team": name,
            "Score": round(sc, 1),
            "Opponent": opp_of[name],
            "Result": {1.0: "W", 0.0: "L", 0.5: "T"}[actual],
            "All-Play Record": f"{w}-{l}" + (f"-{t}" if t else ""),
            "All-Play Win %": round(ap_pct * 100, 1),
            "Luck": round((actual - ap_pct) * 100, 1),
        })
    table = pd.DataFrame(rows).sort_values(by="Luck", ascending=False, kind="mergesort").reset_index(drop=True)
    rec.table = table
    lucky = table.iloc[0]
    unlucky = table.iloc[-1]
    if lucky["Luck"] > 0:
        rec.luckiest = lucky.to_dict()
    if unlucky["Luck"] < 0:
        rec.unluckiest = unlucky.to_dict()

    # Upsets: the team with the lower projected total won
    for b in games:
        hp, ap = b.home_projected, b.away_projected
        if hp is None or ap is None or abs(hp - ap) < 1e-9:
            continue
        h, a = float(b.home_score), float(b.away_score)
        if h == a:
            continue
        fav_is_home = hp > ap
        home_won = h > a
        if fav_is_home != home_won:
            w, l = (b.home_team, b.away_team) if home_won else (b.away_team, b.home_team)
            wp, lp = (hp, ap) if home_won else (ap, hp)
            ws, ls = (h, a) if home_won else (a, h)
            rec.upsets.append({"winner": w.team_name, "loser": l.team_name,
                               "winner_score": ws, "loser_score": ls,
                               "winner_proj": float(wp), "loser_proj": float(lp),
                               "gap": float(lp - wp)})
    rec.upsets.sort(key=lambda u: -u["gap"])

    # Scoring records: this season so far, then all-time (needs past seasons)
    prior = [float(sc) for t in league.teams for sc in t.scores[: week - 1] if sc and sc > 0]
    for kind, (name, sc, _) in (("high", top), ("low", bottom)):
        better = (lambda x, y: x > y) if kind == "high" else (lambda x, y: x < y)
        pick = max if kind == "high" else min
        season_rec = (not prior) or better(sc, pick(prior))
        if not season_rec:
            continue
        hist = history_loader() if history_loader else []
        if hist:
            best = pick(hist, key=lambda r: r["score"])
            if better(sc, best["score"]):
                rec.records.append({"kind": kind, "scope": "league history", "team": name, "score": sc,
                                    "previous": f'{best["score"]:.1f} by {best["team"]} ({best["year"]})'})
                continue
        if prior:
            rec.records.append({"kind": kind, "scope": "season", "team": name, "score": sc,
                                "previous": f"{pick(prior):.1f}"})
    return rec


def md_escape(text) -> str:
    """Team names are user-typed; keep Markdown/LaTeX characters literal."""
    return re.sub(r"([\\`*_\[\]~$<>|#])", r"\\\1", str(text))


@st.cache_data(show_spinner="Loading past seasons for league records...", ttl=24 * 60 * 60)
def get_league_history_scores(league_id: int, year: int, espn_s2: str, swid: str) -> list:
    """Every recorded team score from earlier seasons (newest first, stops at the
    first season ESPN can't return). Used only for all-time record callouts."""
    rows = []
    for y in range(year - 1, year - 1 - HISTORY_MAX_SEASONS, -1):
        try:
            past = League(league_id=league_id, year=y, espn_s2=espn_s2, swid=swid)
        except Exception:
            break
        for t in past.teams:
            for wk, sc in enumerate(t.scores, start=1):
                if sc and sc > 0:
                    rows.append({"year": y, "week": wk, "team": t.team_name, "score": float(sc)})
    return rows


# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------
def main():
    inject_css()

    league_id, year, espn_s2, swid = load_credentials()

    if not (league_id and espn_s2 and swid):
        st.info(
            "Enter your League ID, espn_s2, and SWID in the sidebar "
            "(or set them in a .env file) to connect."
        )
        st.stop()

    try:
        league = get_league(int(league_id), int(year), espn_s2, swid)
    except Exception as e:
        st.error(f"Couldn't connect to ESPN. Double-check your credentials. ({e})")
        st.stop()

    st.html(hero(league.settings.name, year, league.current_week))

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "Standings",
            "Rosters & Lineup",
            "Matchups",
            "Power Rankings",
            "Free Agents",
            "Injury Report",
            "Trade Analyzer",
            "Weekly Recap",
        ]
    )

    week_options = list(range(1, 19))
    current_week = league.current_week
    default_week_index = (
        week_options.index(current_week) if current_week in week_options else 0
    )

    # --- Standings ---
    with tab1:
        standings_df = build_standings_df(league)
        stat_cards(standings_summary(standings_df))
        st.write("")
        show_table(standings_df, pills={"Streak": streak_pill})
        st.write("")
        st.plotly_chart(build_points_chart(standings_df), width="stretch")

    # --- Rosters & Lineup Optimizer ---
    with tab2:
        rc1, rc2 = st.columns(2)
        with rc1:
            team_names = [t.team_name for t in league.teams]
            selected = dropdown("Select a team", team_names)
        with rc2:
            roster_week = dropdown(
                "Week", week_options, index=default_week_index, key="roster_week"
            )
        team = next(t for t in league.teams if t.team_name == selected)
        roster_week = int(roster_week)
        lookup = get_week_projection_lookup(
            league, int(league_id), int(year), roster_week
        )

        if not lookup:
            st.warning(
                f"ESPN hasn't published Week {roster_week} projections yet, so "
                "weekly numbers will show as blank. Season projections are "
                "shown for context only and are never used for lineup advice."
            )

        roster_df = build_roster_df(team, roster_week, lookup)
        show_table(roster_df, pills={"Injury Status": status_pill})

        st.subheader(f"Lineup Optimizer (Week {roster_week})")
        st.caption(
            "Based on ESPN's projections for the selected week only, and it "
            "respects your league's lineup slots, so a player is only ever "
            "suggested for a slot he's eligible to fill."
        )
        try:
            # Free agents only make sense for the current or an upcoming week.
            fa_candidates, fa_failed = None, []
            if roster_week >= current_week:
                slots = starting_slots(league)
                fa_pool, fa_failed = get_free_agent_pool(
                    league,
                    int(league_id),
                    int(year),
                    roster_week,
                    free_agent_positions(slots),
                    FREE_AGENTS_PER_POSITION,
                )
                fa_candidates = free_agent_candidates(fa_pool)

            res = optimize_lineup(league, team, roster_week, lookup, fa_candidates)
            left_on_bench = round(res.optimal_total - res.current_total, 1)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"Current Starters (Wk {roster_week})", f"{res.current_total:.1f}")
            col2.metric("Optimal (Your Roster)", f"{res.optimal_total:.1f}")
            col3.metric("Points Left on Bench", f"{left_on_bench:+.1f}")
            if res.fa_total is None:
                col4.metric("Optimal + Free Agents", "—")
            else:
                col4.metric(
                    "Optimal + Free Agents",
                    f"{res.fa_total:.1f}",
                    delta=f"{res.fa_total - res.optimal_total:+.1f}",
                )

            if res.missing:
                st.warning(
                    "No Week "
                    f"{roster_week} projection from ESPN for: {', '.join(res.missing)}. "
                    "They're counted as 0.0, so double-check those spots by hand."
                )

            if res.swaps_df.empty or left_on_bench <= 0:
                st.success("Your current lineup already matches the optimal lineup.")
            else:
                st.write("Suggested swaps to reach the optimal lineup:")
                show_table(res.swaps_df)

            st.subheader(f"Free Agent Upgrades (Week {roster_week})")
            if fa_candidates is None:
                st.info(
                    "Free agent upgrades are only shown for the current or an "
                    "upcoming week."
                )
            else:
                if fa_failed:
                    st.warning(
                        "Couldn't load free agents for: "
                        f"{', '.join(fa_failed)}. Results may be incomplete."
                    )
                if res.fa_df.empty:
                    st.success(
                        "No available free agent projects to outscore a player "
                        "in your optimal lineup this week."
                    )
                else:
                    show_table(res.fa_df, pills={"Injury Status": status_pill})
                st.caption(
                    "Measured against your optimal lineup (after the swaps above), "
                    "using each position's top "
                    f"{FREE_AGENTS_PER_POSITION} most-owned free agents. Players "
                    "who are Out, on Injured Reserve or suspended are skipped. "
                    "Positions are respected: if the player replaced plays a "
                    "different position, the move works through a FLEX-type slot "
                    "(e.g. a RB takes a RB slot and another RB moves to FLEX). "
                    "You'll need to drop someone to make room; this doesn't judge "
                    "long-term value."
                )
        except Exception as e:
            st.warning(f"Couldn't compute an optimal lineup for this roster. ({e})")

    # --- Matchups ---
    with tab3:
        week = dropdown(
            "Week", week_options, index=default_week_index, key="matchup_week"
        )
        matchup_rows = build_matchup_rows(league, int(week))
        stat_cards(matchup_summary(matchup_rows))
        st.write("")
        st.html(matchups_html(matchup_rows))
        st.caption(
            "Each score shows ESPN's projected score in gray parentheses, "
            "then the actual score. The leading side is highlighted."
        )

    # --- Power Rankings ---
    with tab4:
        pr_week = dropdown(
            "Week for power rankings",
            week_options,
            index=default_week_index,
            key="pr_week",
        )
        pr_df = build_power_rankings_df(league, int(pr_week))
        if pr_df.empty:
            st.warning("Power rankings aren't available for this week yet.")
        else:
            stat_cards(power_summary(pr_df))
            st.write("")
            st.dataframe(
                pr_df,
                width="stretch",
                hide_index=True,
                height=fit_height(pr_df),
            )
            if not pr_df.attrs.get("formula_matches", True):
                st.warning(
                    "The columns above don't exactly reproduce the Power Score, "
                    "so your espn_api version may use a slightly different "
                    "formula. The score shown is the library's own."
                )
            st.caption(POWER_FOOTNOTE)

    # --- Free Agents / Waiver Wire ---
    with tab5:
        fa_col1, fa_col2 = st.columns(2)
        with fa_col1:
            fa_week = dropdown(
                "Week", week_options, index=default_week_index, key="fa_week"
            )
        with fa_col2:
            position = dropdown(
                "Position",
                ["All", "QB", "RB", "WR", "TE", "DP", "K"],
                key="fa_position",
            )

        fa_df = build_free_agents_df(league, int(fa_week), position)
        if fa_df.empty:
            st.info("No free agent data available for this selection.")
        else:
            stat_cards(free_agent_summary(fa_df, int(fa_week)))
            st.caption(
                f"Showing up to {FREE_AGENT_LIST_SIZE} free agents (the most-owned "
                f"first), sorted by their Week {int(fa_week)} projection."
            )
            show_table(fa_df, pills={"Injury Status": status_pill}, max_height=700)

    # --- Injury Report ---
    with tab6:
        inj_week = week_options[default_week_index]
        st.caption(
            "League-wide view of every rostered player with a non-active "
            f"status, with their Week {inj_week} projection."
        )
        inj_lookup = get_week_projection_lookup(
            league, int(league_id), int(year), inj_week
        )
        injury_df = build_injury_report_df(league, inj_week, inj_lookup)
        if injury_df.empty:
            st.success("No notable injuries reported across the league right now.")
        else:
            stat_cards(injury_summary(injury_df))
            st.write("")
            inj_team = dropdown(
                "Team",
                ["All"] + sorted(t.team_name for t in league.teams),
                key="injury_team",
            )
            if inj_team != "All":
                injury_df = injury_df[injury_df["Team"] == inj_team].reset_index(drop=True)
            st.caption(
                "Sorted by team name, then status from least severe (top) to "
                "most severe (bottom)."
            )
            if injury_df.empty:
                st.success(f"No injuries on {inj_team} right now.")
            else:
                show_table(injury_df, pills={"Status": status_pill})

    # --- Trade Analyzer ---
    with tab7:
        trade_week = week_options[default_week_index]
        st.caption(
            "Pick the players each side would give up. Once a player is chosen, "
            "that side is locked to his fantasy team's roster, and the other "
            f"side can only pick from other teams. Uses ESPN's Week {trade_week} "
            "projection, season average and season-long projection."
        )
        trade_lookup = get_week_projection_lookup(
            league, int(league_id), int(year), trade_week
        )
        all_pairs = get_all_rostered_players(league)
        by_key = {_player_key(p): (p, t) for p, t in all_pairs}

        # Drop stale picks (e.g. after a roster refresh) BEFORE the widgets draw.
        a_sel, b_sel = clean_trade_selection(
            all_pairs,
            st.session_state.get("trade_a", []),
            st.session_state.get("trade_b", []),
        )
        st.session_state["trade_a"], st.session_state["trade_b"] = a_sel, b_sel
        a_opts, lock_a = trade_side_options(all_pairs, a_sel, None)
        b_opts, lock_b = trade_side_options(all_pairs, b_sel, lock_a)
        if lock_b is not None:  # A can't pick from the team B is locked to
            a_opts, _ = trade_side_options(all_pairs, a_sel, lock_b)
        team_names = {t.team_id: t.team_name for t in league.teams}

        def player_label(locked):
            def fmt(k):
                p, t = by_key[k]
                base = f"{p.name} ({p.position})"
                return base if locked is not None else f"{base} — {t.team_name}"
            return fmt

        def order(keys):
            return sorted(keys, key=lambda k: by_key[k][0].name.casefold())

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Trade partner 1 gives up**")
            pick_a = st.multiselect(
                "Players from side 1", order(a_opts), key="trade_a",
                format_func=player_label(lock_a), label_visibility="collapsed",
                placeholder="Choose players...",
            )
            if lock_a is not None:
                st.caption(f"Locked to {team_names[lock_a]}'s roster")
        with col_b:
            st.markdown("**Trade partner 2 gives up**")
            pick_b = st.multiselect(
                "Players from side 2", order(b_opts), key="trade_b",
                format_func=player_label(lock_b), label_visibility="collapsed",
                placeholder="Choose players...",
            )
            if lock_b is not None:
                st.caption(f"Locked to {team_names[lock_b]}'s roster")

        if not pick_a and not pick_b:
            st.info("Pick at least one player on each side to compare the trade.")
        else:
            wk_col = f"Week {trade_week} Proj"
            frames = []
            for label_side, picks in (("Side 1", pick_a), ("Side 2", pick_b)):
                if picks:
                    df_side = build_player_compare_df(
                        [by_key[k] for k in picks if k in by_key], trade_week, trade_lookup
                    )
                    df_side.insert(0, "Side", label_side)
                    frames.append(df_side)
            compare_df = pd.concat(frames, ignore_index=True)

            tcol_a, tcol_b = st.columns(2)
            for col, label_side, picks in ((tcol_a, "Side 1", pick_a), (tcol_b, "Side 2", pick_b)):
                with col:
                    sub = compare_df[compare_df["Side"] == label_side].drop(columns="Side")
                    if sub.empty:
                        st.caption("No players selected yet.")
                        continue
                    show_table(sub, pills={"Injury Status": status_pill})

            if pick_a and pick_b:
                tot = compare_df.groupby("Side")[[wk_col, "Avg Points", "Season Proj"]].sum(min_count=1)
                st.markdown("**Totals (what each side gives up)**")
                show_table(tot.reset_index().rename(columns={"Side": ""}))
                a_name = by_key[pick_a[0]][1].team_name
                b_name = by_key[pick_b[0]][1].team_name
                gain_a = float(np.nan_to_num(tot.loc["Side 2", wk_col])) - float(
                    np.nan_to_num(tot.loc["Side 1", wk_col])
                )
                m1, m2 = st.columns(2)
                m1.metric(f"{a_name} — Week {trade_week} projected change", f"{gain_a:+.1f}")
                m2.metric(f"{b_name} — Week {trade_week} projected change", f"{-gain_a:+.1f}")
                st.caption(
                    "Change = projected points of the players received minus the players given up. "
                    "Raw projected points only; it doesn't account for roster fit or bye weeks."
                )

            fig = px.bar(
                compare_df.assign(Owner=compare_df["Fantasy Owner"]),
                x="Player", y=wk_col, color="Owner",
                title=f"Week {trade_week} Projected Points",
            )
            st.plotly_chart(style_fig(fig), width="stretch")

    # --- Weekly Recap ---
    with tab8:
        st.caption(
            "A league-wide story of a completed week: top and bottom scores, "
            "closest and widest wins, luck, upsets, records, and which starters "
            "beat or missed their projection by the widest margin."
        )
        recap_week = dropdown(
            "Week",
            week_options,
            index=max(default_week_index - 1, 0),  # default to last completed week
            key="recap_week",
        )
        recap_boxes = league.box_scores(week=int(recap_week))
        recap = build_league_recap(
            league, int(recap_week), boxes=recap_boxes,
            history_loader=lambda: get_league_history_scores(
                int(league_id), int(year), espn_s2, swid
            ),
        )

        if not recap.completed:
            st.info("No completed results available for this week yet.")
        else:
            st.subheader(f"Week {recap_week} at a glance")
            e = md_escape
            lines = [
                f"🔥 **Highest score:** {e(recap.highest['team'])} — {recap.highest['score']:.1f}",
                f"🧊 **Lowest score:** {e(recap.lowest['team'])} — {recap.lowest['score']:.1f}",
            ]
            for icon, label, g in (("😅", "Narrowest win", recap.narrowest), ("💥", "Widest win", recap.widest)):
                if g:
                    lines.append(
                        f"{icon} **{label}:** {e(g['winner'])} {g['winner_score']:.1f}, "
                        f"{e(g['loser'])} {g['loser_score']:.1f} (by {g['margin']:.1f})"
                    )
            for icon, label, r, verb in (("🍀", "Luckiest", recap.luckiest, "won"),
                                          ("😤", "Unluckiest", recap.unluckiest, "lost")):
                if r:
                    lines.append(
                        f"{icon} **{label}:** {e(r['Team'])} {verb} with {r['Score']:.1f}, but would "
                        f"have gone {r['All-Play Record']} vs. the whole league "
                        f"({r['All-Play Win %']:.0f}% all-play win rate)"
                    )
            st.markdown("\n\n".join(lines))

            if recap.records:
                st.subheader("Record performances")
                for r in recap.records:
                    kind = "Highest" if r["kind"] == "high" else "Lowest"
                    st.markdown(
                        f"🏆 **{kind} score in {r['scope']}:** {e(r['team'])} — {r['score']:.1f} "
                        f"(previous {'record' if r['scope'] == 'league history' else 'mark this season'}: "
                        f"{e(r['previous'])})"
                    )

            st.subheader("Upsets")
            if recap.upsets:
                for u in recap.upsets:
                    st.markdown(
                        f"⚡ **{e(u['winner'])}** ({u['winner_score']:.1f}) beat **{e(u['loser'])}** "
                        f"({u['loser_score']:.1f}) despite being projected lower "
                        f"({u['winner_proj']:.1f} vs. {u['loser_proj']:.1f}, a {u['gap']:.1f}-point gap)"
                    )
            else:
                st.caption("No upsets: every team projected to win did win.")

            with st.expander("All-play results and luck for every team"):
                st.caption(
                    "All-play record = how the team's score ranks against every other team that week. "
                    "Luck = actual result (win 100, tie 50, loss 0) minus all-play win %."
                )
                st.dataframe(
                    styled(recap.table), width="stretch", hide_index=True,
                    height=fit_height(recap.table),
                )

            recap_df = build_weekly_recap_df(league, int(recap_week), boxes=recap_boxes)
            st.subheader("Player performance vs. projection")
            if recap_df.empty:
                st.info("No player-level projections are available for this week.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Biggest overperformers**")
                    st.dataframe(recap_df.head(10), width="stretch", hide_index=True)
                with col2:
                    st.write("**Biggest underperformers**")
                    st.dataframe(
                        recap_df.tail(10).sort_values(by="Diff (Actual − Proj)", ascending=True),
                        width="stretch", hide_index=True,
                    )


if __name__ == "__main__":
    main()
