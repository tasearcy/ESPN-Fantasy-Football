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
import math
import os
import re
from collections import Counter
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
        get_week_boxes.clear()
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


def get_week_actual_points(player, week: int):
    """Points actually scored in `week`, or None if ESPN has no score yet
    (the game hasn't been played, so show a dash rather than a fake 0.0)."""
    stats = getattr(player, "stats", None) or {}
    week_stats = stats.get(week) or stats.get(str(week)) or {}
    val = week_stats.get("points")
    return val if isinstance(val, (int, float)) else None


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


# Points For bars: soft red (fewest wins) -> cream at the halfway mark -> soft green
# (most wins). Gentler on the eyes; the cream midpoint separates winning from losing.
WIN_BAR_COLORSCALE = [
    [0.00, "#d64545"],
    [0.25, "#f19a82"],
    [0.50, "#ece6d6"],
    [0.75, "#9fd49a"],
    [1.00, "#2a9d5c"],
]

# Points For vs. Points Against scatter keeps the high-contrast scale: dots need to
# stand out more than bars do.
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


def win_scale_bounds(df: pd.DataFrame) -> tuple:
    """Color scale for wins: 0 up to the games played so far, i.e. the fewest and
    most wins that were possible, not just the fewest and most anyone has."""
    ties = df["Ties"] if "Ties" in df else 0
    games = int((df["Wins"] + df["Losses"] + ties).max()) if len(df) else 0
    return 0, max(games, 1)


def build_points_chart(standings_df: pd.DataFrame):
    """Horizontal bars: most points at the top, colored by wins
    (green = most wins, red = fewest)."""
    df = standings_df.sort_values("Points For", ascending=False, kind="stable")
    lo, hi = win_scale_bounds(standings_df)

    fig = px.bar(
        df,
        x="Points For",
        y="Team",
        orientation="h",
        color="Wins",
        color_continuous_scale=WIN_BAR_COLORSCALE,
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
        coloraxis_colorbar=dict(title="Wins", tickmode="linear", tick0=0, dtick=1),
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
                f"Week {week} Actual": _safe_round(get_week_actual_points(player, week)),
                "PPG": _safe_round(get_avg_points(player)),
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


@st.cache_resource(show_spinner=False, ttl=5 * 60)
def get_week_boxes(_league, league_id: int, year: int, week: int) -> list:
    """The week's box scores, shared by the Matchups, Rosters and Recap tabs so
    ESPN is asked once per few minutes instead of once per tab."""
    return list(_league.box_scores(week=week))


TEAM_SCORE_SD = 20.0  # typical gap between a team's actual and projected score


def _lineup_outlook(lineup) -> tuple:
    """(expected final score, full projected total, projected points still to come)
    for a team's starters. Finished games count their real points; games not yet
    started count the projection; games in progress count whichever is larger."""
    expected = total = remaining = 0.0
    for p in lineup or []:
        if getattr(p, "slot_position", None) in ("BE", "IR"):
            continue
        proj = getattr(p, "projected_points", None)
        pts = getattr(p, "points", None)
        proj = float(proj) if isinstance(proj, (int, float)) else 0.0
        pts = float(pts) if isinstance(pts, (int, float)) else 0.0
        played = getattr(p, "game_played", None)
        finished = (played == 100) if played is not None else pts > 0
        total += proj
        if finished:
            expected += pts
        elif pts > 0:
            expected += max(pts, proj)
            remaining += max(proj - pts, 0.0)
        else:
            expected += proj
            remaining += proj
    return expected, total, remaining


def win_probability(home_lineup, away_lineup, sd: float = TEAM_SCORE_SD):
    """Chance each side wins, from ESPN's projections and the points already
    scored: a normal model on the expected score gap, whose uncertainty shrinks as
    games finish. This is our own estimate, not ESPN's displayed win probability.
    Returns {"home": p, "final": bool} or None without projections."""
    eh, ph, rh = _lineup_outlook(home_lineup)
    ea, pa, ra = _lineup_outlook(away_lineup)
    if ph + pa <= 0:
        return None
    frac = (rh + ra) / (ph + pa)
    diff = eh - ea
    if frac < 0.005:  # every starter's game is over
        return {"home": 1.0 if diff > 0 else 0.0 if diff < 0 else 0.5, "final": True}
    sigma = sd * math.sqrt(2 * min(frac, 1.0))
    return {"home": 0.5 * (1 + math.erf(diff / (sigma * math.sqrt(2)))), "final": False}


def build_matchup_rows(league: League, week: int, boxes=None) -> list:
    """One dict per game: each side has team, score and ESPN's projected score,
    plus our win probability estimate for the game."""
    def side(team, score, projected):
        if team is None:
            return {"team": "BYE", "score": None, "proj": None}
        return {"team": team.team_name, "score": score, "proj": projected}

    rows = []
    for box in (boxes if boxes is not None else league.box_scores(week=week)):
        win = None
        if box.home_team is not None and box.away_team is not None:
            win = win_probability(getattr(box, "home_lineup", None), getattr(box, "away_lineup", None))
        rows.append({
            "away": side(box.away_team, box.away_score, box.away_projected),
            "home": side(box.home_team, box.home_score, box.home_projected),
            "win": win,
        })
    return rows


def team_week_summary(boxes: list, team) -> dict | None:
    """Score, projection and opponent for `team` in one week's box scores."""
    for box in boxes:
        for me, opp, my_s, opp_s, my_p in (
            (box.home_team, box.away_team, box.home_score, box.away_score, box.home_projected),
            (box.away_team, box.home_team, box.away_score, box.home_score, box.away_projected),
        ):
            if me is not None and getattr(me, "team_id", None) == team.team_id:
                return {
                    "score": float(my_s or 0), "proj": my_p,
                    "opp": opp.team_name if opp is not None else None,
                    "opp_score": float(opp_s or 0) if opp is not None else None,
                    "played": bool(my_s or opp_s),
                }
    return None


def week_summary_cards(summary: dict | None, week: int) -> list:
    if not summary:
        return [(f"Week {week} Points", "—", "bye week or no game found")]
    pts = f'{summary["score"]:.1f}' if summary["played"] else "—"
    proj = f'{float(summary["proj"]):.1f} projected' if isinstance(summary["proj"], (int, float)) else None
    cards = [(f"Week {week} Points Scored", pts, proj)]
    if summary["opp"] is not None:
        cards.append(("Opponent", summary["opp"],
                      f'scored {summary["opp_score"]:.1f}' if summary["played"] else "not played yet"))
        if summary["played"]:
            margin = summary["score"] - summary["opp_score"]
            cards.append(("Margin", f"{margin:+.1f}", "leading" if margin > 0 else "trailing" if margin < 0 else "tied"))
    return cards


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
        win = r.get("win")
        if bar and win:
            ph = win["home"] * 100
            pa = 100 - ph
            away_leads = pa >= ph
            mid = "Final" if win["final"] else "Win probability"
            split = (
                '<div class="sb-split"><div class="bar-track" style="display:flex;">'
                f'<div style="width:{pa:.1f}%;background:{ACCENT if away_leads else GRAY};'
                f'{"" if away_leads else "opacity:0.55;"}"></div>'
                f'<div style="width:{ph:.1f}%;background:{GRAY if away_leads else ACCENT};'
                f'{"opacity:0.55;" if away_leads else ""}"></div></div>'
                f'<div class="sb-split-cap"><b>{pa:.0f}%</b><span>{mid}</span><b>{ph:.0f}%</b></div></div>'
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
                "PPG": _safe_round(get_avg_points(player)),
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
                "PPG": _safe_round(get_avg_points(player)),
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
.ft-nowrap td:first-child { white-space: nowrap; }
.ft-nowrap td, .ft-nowrap th { padding-left: 0.45rem; padding-right: 0.45rem; }
.ft-nowrap .bar-cell { min-width: 74px; gap: 0.35rem; }
.ft-nowrap .bar-cell .v { min-width: 2.6rem; }
.sc { border-collapse: separate; border-spacing: 2px; font-size: 0.82rem; width: 100%; }
.sc th { font-weight: 700; font-size: 0.72rem; color: inherit; text-align: center; padding: 0.3rem 0.2rem; }
.sc .sc-h { vertical-align: bottom; min-width: 52px; max-width: 76px; }
.sc .sc-h span { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sc .sc-r, .sc .sc-corner { text-align: left; white-space: nowrap; padding-right: 0.7rem; position: sticky; left: 0; background: #0b1220; z-index: 1; }
.sc .sc-corner { font-size: 0.68rem; opacity: 0.7; font-weight: 600; }
.sc .sc-cell { text-align: center; font-weight: 700; font-variant-numeric: tabular-nums; padding: 0.42rem 0.1rem; border-radius: 6px; }
.sc .sc-diag { outline: 2px solid #e6ebf5; outline-offset: -2px; }
.trade-card { border: 1px solid rgba(128,128,128,0.25); border-radius: 14px; padding: 0.9rem 1rem 1rem; margin-bottom: 1rem; background: rgba(128,128,128,0.04); }
.trade-head { display: flex; justify-content: space-between; align-items: center; gap: 0.8rem; margin-bottom: 0.7rem; flex-wrap: wrap; }
.trade-title { font-weight: 800; font-size: 1.05rem; }
.trade-sub { font-size: 0.78rem; opacity: 0.65; }
.trade-net { margin-top: 0.6rem; font-size: 0.85rem; opacity: 0.85; }
.ft-trade th.grp-h { text-align: center; border-bottom: 1px solid rgba(128,128,128,0.25); }
.ft-trade .trade-mv { white-space: nowrap; opacity: 0.85; font-size: 0.85rem; }
.ft .grp td { background: rgba(56,189,248,0.09); font-weight: 700; }
.grp-owner { font-weight: 500; opacity: 0.6; margin-left: 0.4rem; font-size: 0.8rem; }
.tipnum { border-bottom: 1px dotted currentColor; cursor: help; }
.sc .sc-cell small { display: block; font-weight: 500; font-size: 0.66rem; opacity: 0.8; }
.sc .sc-na { background: rgba(128,128,128,0.12); opacity: 0.6; text-align: center; border-radius: 6px; }
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
.sb-split-cap { font-size: 0.75rem; display: flex; justify-content: space-between; margin-top: 0.25rem; }
.sb-split-cap span { opacity: 0.6; }

.podium { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 0.9rem; align-items: end; margin-bottom: 1rem; }
.pod { border-radius: 16px; padding: 1rem 1.2rem; border: 1px solid rgba(128,128,128,0.28);
  background: rgba(128,128,128,0.08); border-top: 5px solid var(--medal); }
.pod.first { padding-top: 1.4rem; padding-bottom: 1.4rem; background: rgba(245,197,66,0.10); }
.pod-medal { font-size: 1.6rem; }
.pod-team { font-size: 1.25rem; font-weight: 800; overflow-wrap: anywhere; }
.pod-sub { opacity: 0.7; font-size: 0.85rem; }
.pod-line { margin-top: 0.5rem; display: flex; justify-content: space-between; font-variant-numeric: tabular-nums; }

.lc-h { font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.7; margin: 0.9rem 0 0.4rem; }
.lc-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 0.7rem; }
.lc { border: 1px solid rgba(128,128,128,0.28); border-radius: 14px; padding: 0.7rem 0.85rem; background: rgba(128,128,128,0.08);
  display: flex; flex-direction: column; gap: 0.15rem; }
.lc.muted { opacity: 0.85; background: rgba(128,128,128,0.04); }
.lc-top { display: flex; justify-content: space-between; align-items: center; min-height: 1.5rem; }
.lc-slot { font-size: 0.7rem; font-weight: 800; letter-spacing: 0.08em; color: #0b1220; background: #38bdf8;
  border-radius: 6px; padding: 0.08rem 0.45rem; }
.lc.muted .lc-slot { background: rgba(128,128,128,0.5); color: #fff; }
.lc-name { font-weight: 700; overflow-wrap: anywhere; line-height: 1.2; margin-top: 0.2rem; }
.lc-sub { font-size: 0.78rem; opacity: 0.65; }
.lc-bot { display: flex; align-items: baseline; justify-content: space-between; margin-top: 0.35rem; }
.lc-proj { font-size: 1.5rem; font-weight: 800; font-variant-numeric: tabular-nums; }
.lc-avg { font-size: 0.75rem; opacity: 0.6; }
.dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }

.mv-list { display: grid; gap: 0.6rem; }
.mv { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 1.4rem; border: 1px solid rgba(128,128,128,0.28);
  border-radius: 14px; padding: 0.7rem 1rem; background: rgba(128,128,128,0.07); }
.mv-side { display: flex; align-items: center; gap: 0.5rem; min-width: 210px; flex: 1; }
.mv-up .arr { color: #34d399; } .mv-down .arr { color: #f87171; }
.mv-p { margin-left: auto; font-variant-numeric: tabular-nums; opacity: 0.8; }
.mv-gain { font-size: 1.3rem; font-weight: 800; color: #34d399; font-variant-numeric: tabular-nums; }
.mv-meta { font-size: 0.8rem; opacity: 0.7; }

.hl-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 0.8rem; }
.hl { border: 1px solid rgba(128,128,128,0.28); border-radius: 16px; padding: 0.9rem 1.1rem; background: rgba(128,128,128,0.08); }
.hl-icon { font-size: 1.5rem; }
.hl-label { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.7; margin-top: 0.15rem; }
.hl-value { font-size: 1.2rem; font-weight: 800; overflow-wrap: anywhere; line-height: 1.25; }
.hl-sub { font-size: 0.82rem; opacity: 0.7; margin-top: 0.2rem; }
.callout { border-left: 5px solid #f5c542; background: rgba(245,197,66,0.10); border-radius: 10px; padding: 0.65rem 1rem; margin: 0.45rem 0; }
.callout.upset { border-left-color: #a78bfa; background: rgba(167,139,250,0.10); }

.verdict { border-radius: 16px; padding: 1rem 1.3rem; margin: 0.6rem 0 0.9rem; text-align: center;
  background: linear-gradient(120deg, rgba(56,189,248,0.20), rgba(52,211,153,0.14)); border: 1px solid rgba(128,128,128,0.25); }
.verdict-big { font-size: 1.35rem; font-weight: 800; }
.verdict-sub { opacity: 0.7; font-size: 0.85rem; margin-top: 0.15rem; }
.tside { border: 1px solid rgba(128,128,128,0.28); border-radius: 16px; padding: 0.9rem 1.1rem; background: rgba(128,128,128,0.07); }
.tside.win { border-color: #34d399; box-shadow: 0 0 0 1px #34d399 inset; }
.tside-h { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.65; }
.tside-team { font-size: 1.15rem; font-weight: 800; margin-bottom: 0.3rem; }
.tside-row { display: flex; align-items: center; gap: 0.6rem; padding: 0.4rem 0; border-top: 1px solid rgba(128,128,128,0.18); }
.tside-row .p { margin-left: auto; font-weight: 700; font-variant-numeric: tabular-nums; }
.tside-total { margin-top: 0.5rem; display: flex; justify-content: space-between; align-items: baseline; border-top: 2px solid rgba(128,128,128,0.35); padding-top: 0.5rem; }
.tside-total .big { font-size: 1.8rem; font-weight: 800; font-variant-numeric: tabular-nums; }
[data-testid="stExpander"] { border-radius: 14px; border-color: rgba(128,128,128,0.28); }
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
        f'<div class="hero-sub">{html.escape(str(year))} Season · Week {html.escape(str(week))}</div>'
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



# ---------------------------------------------------------------------
# Full visual layer: podium, lineup cards, move cards, highlights, charts
# ---------------------------------------------------------------------
def _n(v, digits=1, signed=False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        return f"{float(v):+.{digits}f}" if signed else f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return html.escape(str(v))


def _e(v) -> str:
    return html.escape(str(v))


def record_bar(wins, losses, ties) -> str:
    wins, losses, ties = int(wins), int(losses), int(ties)
    games = wins + losses + ties
    label = f"{wins}-{losses}" + (f"-{ties}" if ties else "")
    if games == 0:
        return label
    w, t = wins / games * 100, ties / games * 100
    return (
        '<div class="bar-cell" style="min-width:120px;">'
        f'<span class="v" style="min-width:3.2rem;text-align:left;">{label}</span>'
        '<div class="bar-track" style="display:flex;">'
        f'<div style="width:{w:.1f}%;background:{WIN_COLOR};"></div>'
        f'<div style="width:{t:.1f}%;background:{GRAY};"></div>'
        f'<div style="width:{100 - w - t:.1f}%;background:{LOSS_COLOR};"></div></div></div>'
    )


def podium_html(df: pd.DataFrame) -> str:
    medals = [("🥇", GOLD, "first"), ("🥈", SILVER, ""), ("🥉", BRONZE, "")]
    cards = []
    for i, (_, r) in enumerate(df.head(3).iterrows()):
        icon, color, cls = medals[i]
        rec = f'{int(r["Wins"])}-{int(r["Losses"])}' + (f'-{int(r["Ties"])}' if r["Ties"] else "")
        owner = f'<div class="pod-sub">{_e(r["Owner"])}</div>' if r["Owner"] else ""
        cards.append(
            f'<div class="{("pod " + cls).strip()}" style="--medal:{color};">'
            f'<div class="pod-medal">{icon}</div><div class="pod-team">{_e(r["Team"])}</div>{owner}'
            f'<div class="pod-line"><span>Record</span><b>{rec}</b></div>'
            f'<div class="pod-line"><span>Points For</span><b>{_n(r["Points For"])}</b></div></div>'
        )
    return f'<div class="podium">{"".join(cards)}</div>'


def standings_view_df(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Rank": df["Rank"],
            "Team": df["Team"],
            "Owner": df["Owner"],
            "Record": [record_bar(w, l, t) for w, l, t in zip(df["Wins"], df["Losses"], df["Ties"])],
            "Points For": df["Points For"],
            "Points Against": df["Points Against"],
            "Streak": df["Streak"],
        }
    )


def standings_table_html(df: pd.DataFrame, playoff_teams=None) -> str:
    view = standings_view_df(df)
    marks = {}
    if isinstance(playoff_teams, int) and 0 < playoff_teams < len(view):
        marks = {playoff_teams - 1: f"Playoff line · top {playoff_teams} advance"}
    return html_table(
        view,
        pills={"Streak": streak_pill},
        bars={
            "Points For": (float(df["Points For"].max() or 1), ACCENT, float(df["Points For"].min()) * 0.85),
            "Points Against": (float(df["Points Against"].max() or 1), "#f97316", float(df["Points Against"].min()) * 0.85),
        },
        raw_cols=("Record",),
        marks=marks,
    )


def build_score_trend_chart(league):
    """Weekly score per team, one line each. None before any game is played."""
    played = max(
        (i + 1 for t in league.teams for i, sc in enumerate(t.scores) if sc and sc > 0), default=0
    )
    if not played:
        return None
    rows = [
        {"Team": t.team_name, "Week": wk, "Score": round(float(sc), 1)}
        for t in league.teams
        for wk, sc in enumerate(t.scores[:played], start=1)
        if sc and sc > 0
    ]
    fig = px.line(
        pd.DataFrame(rows), x="Week", y="Score", color="Team", markers=True,
        color_discrete_sequence=PALETTE, title="Weekly Scores",
    )
    fig.update_xaxes(dtick=1)
    fig.update_layout(height=560, margin=dict(l=0, r=10, t=70, b=20), legend_title_text="")
    return style_fig(fig)


def build_pf_pa_chart(df: pd.DataFrame):
    """Points For vs. Points Against. Top-left = scoring a lot, allowing little."""
    fig = px.scatter(
        df, x="Points Against", y="Points For", text="Team", color="Wins",
        color_continuous_scale=WIN_COLORSCALE,
        range_color=win_scale_bounds(df),
        hover_data={"Wins": True, "Losses": True}, title="Points For vs. Points Against",
    )
    fig.update_traces(textposition="top center", marker=dict(size=14, line=dict(width=1, color="rgba(255,255,255,0.6)")))
    fig.add_hline(y=float(df["Points For"].mean()), line_dash="dash", opacity=0.4)
    fig.add_vline(x=float(df["Points Against"].mean()), line_dash="dash", opacity=0.4)
    pad_x = (df["Points Against"].max() - df["Points Against"].min()) * 0.12 + 1
    fig.update_xaxes(range=[df["Points Against"].min() - pad_x, df["Points Against"].max() + pad_x])
    fig.update_layout(height=620, margin=dict(l=0, r=20, t=70, b=30), coloraxis_showscale=False)
    fig.update_traces(marker=dict(size=18, line=dict(width=1, color="rgba(255,255,255,0.6)")))
    return style_fig(fig)


def lineup_cards_html(df: pd.DataFrame, week: int) -> str:
    proj_col = f"Week {week} Proj"
    actual_col = f"Week {week} Actual"
    starters = df[~df["Slot"].isin(["Bench", "IR"])]
    bench = df[df["Slot"] == "Bench"]
    ir = df[df["Slot"] == "IR"]

    def card(r, muted=False):
        status = normalize_status(r["Injury Status"])
        color = STATUS_COLORS.get(status, GRAY)
        badge = (
            pill(status, color) if status != "ACTIVE"
            else f'<span class="dot" title="ACTIVE" style="background:{color};"></span>'
        )
        sub = f'{_e(r["Position"])} · {_e(r["Pro Team"])}'
        return (
            f'<div class="lc{" muted" if muted else ""}"><div class="lc-top">'
            f'<span class="lc-slot">{_e(r["Slot"]).upper()}</span>{badge}</div>'
            f'<div class="lc-name">{_e(r["Player"])}</div><div class="lc-sub">{sub}</div>'
            f'<div class="lc-bot"><span class="lc-proj">{_n(r[actual_col])}</span>'
            f'<span class="lc-avg">proj {_n(r[proj_col])}</span></div></div>'
        )

    total = pd.to_numeric(starters[proj_col], errors="coerce").sum()
    out = [f'<div class="lc-h">Starting lineup · {total:.1f} projected</div><div class="lc-grid">'
           + "".join(card(r) for _, r in starters.iterrows()) + "</div>"]
    if len(bench):
        out.append('<div class="lc-h">Bench</div><div class="lc-grid">'
                   + "".join(card(r, True) for _, r in bench.iterrows()) + "</div>")
    if len(ir):
        out.append('<div class="lc-h">Injured reserve</div><div class="lc-grid">'
                   + "".join(card(r, True) for _, r in ir.iterrows()) + "</div>")
    return "".join(out)


def swap_cards_html(df: pd.DataFrame, week: int) -> str:
    rows = []
    for _, r in df.iterrows():
        pin, pout = r.get(f"Wk {week} Proj (in)"), r.get(f"Wk {week} Proj (out)")
        gain = (pin - pout) if pd.notna(pin) and pd.notna(pout) else None
        rows.append(
            '<div class="mv">'
            f'<div class="mv-side mv-up"><span class="arr">▲</span><b>{_e(r["Bench → Start"] or "—")}</b>'
            f'<span class="mv-p">{_n(pin)}</span></div>'
            f'<div class="mv-side mv-down"><span class="arr">▼</span><span>{_e(r["Start → Bench"] or "—")}</span>'
            f'<span class="mv-p">{_n(pout)}</span></div>'
            f'<div class="mv-gain">{_n(gain, signed=True)}</div>'
            f'<div class="mv-meta">{_e(r["Confidence"])}</div></div>'
        )
    return f'<div class="mv-list">{"".join(rows)}</div>'


def fa_cards_html(df: pd.DataFrame, week: int) -> str:
    rows = []
    for _, r in df.iterrows():
        owned = r.get("% Owned")
        owned_txt = f" · {_n(owned, 0)}% owned" if pd.notna(owned) else ""
        rows.append(
            '<div class="mv">'
            f'<div class="mv-side mv-up"><span class="arr">➕</span><b>{_e(r["Add (Free Agent)"])}</b>'
            f'{status_pill(r["Injury Status"]) if normalize_status(r["Injury Status"]) != "ACTIVE" else ""}'
            f'<span class="mv-p">{_n(r[f"Wk {week} Proj"])}</span></div>'
            f'<div class="mv-side mv-down"><span class="arr">▼</span>'
            f'<span>{_e(r["Replaces in Lineup"])} <span class="mv-meta">(starts at {_e(r["Starts At"])})</span></span>'
            f'<span class="mv-p">{_n(r[f"Wk {week} Proj (out)"])}</span></div>'
            f'<div class="mv-gain">{_n(r["Gain"], signed=True)}</div>'
            f'<div class="mv-meta">{_e(r["Confidence"])}{owned_txt}</div></div>'
        )
    return f'<div class="mv-list">{"".join(rows)}</div>'


def power_view_df(df: pd.DataFrame, prev: pd.DataFrame | None) -> pd.DataFrame:
    """Power table with medal ranks, a movement column and a score bar."""
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    prev_rank = dict(zip(prev["Team"], prev["Rank"])) if prev is not None and not prev.empty else {}

    def move(team, rank):
        if team not in prev_rank:
            return '<span style="opacity:0.5;">—</span>'
        d = int(prev_rank[team]) - int(rank)
        if d > 0:
            return f'<span style="color:{WIN_COLOR};font-weight:700;">▲ {d}</span>'
        if d < 0:
            return f'<span style="color:{LOSS_COLOR};font-weight:700;">▼ {-d}</span>'
        return '<span style="opacity:0.5;">–</span>'

    out = df.copy()
    out.insert(0, "Move", [move(t, r) for t, r in zip(out["Team"], out["Rank"])])
    out["Rank"] = [f'{medals.get(int(r), "")} {int(r)}'.strip() for r in out["Rank"]]
    return out


def power_table_html(df: pd.DataFrame, prev: pd.DataFrame | None) -> str:
    view = power_view_df(df, prev)
    cols = ["Rank", "Move"] + [c for c in view.columns if c not in ("Rank", "Move")]
    return html_table(
        view[cols], raw_cols=("Rank", "Move"),
        bars={"Power Score": (float(df["Power Score"].max() or 1), ACCENT, float(df["Power Score"].min()) * 0.8)}, precision=2,
    )


def diff_html(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    color = WIN_COLOR if v > 0 else LOSS_COLOR if v < 0 else GRAY
    return f'<b style="color:{color};">{float(v):+.1f}</b>'


def trade_side_html(team_name, df_side, wk_col, favored: bool) -> str:
    rows = "".join(
        '<div class="tside-row">'
        f'<span><b>{_e(r["Player"])}</b> <span class="mv-meta">{_e(r["Position"])} · {_e(r["Team (NFL)"])}</span></span>'
        + (status_pill(r["Injury Status"]) if normalize_status(r["Injury Status"]) != "ACTIVE" else "")
        + f'<span class="p">{_n(r[wk_col])}</span></div>'
        for _, r in df_side.iterrows()
    )
    total = pd.to_numeric(df_side[wk_col], errors="coerce").sum()
    avg = pd.to_numeric(df_side["PPG"], errors="coerce").sum()
    season = pd.to_numeric(df_side["Season Proj"], errors="coerce").sum()
    return (
        f'<div class="tside{" win" if favored else ""}"><div class="tside-h">Gives up</div>'
        f'<div class="tside-team">{_e(team_name)}</div>{rows}'
        f'<div class="tside-total"><span class="mv-meta">PPG {avg:.1f} · Season proj {season:.0f}</span>'
        f'<span class="big">{total:.1f}</span></div></div>'
    )


def trade_verdict_html(a_name, b_name, a_gives, b_gives, week) -> str:
    """Team A receives B's players. Positive diff favors A."""
    diff = b_gives - a_gives
    if abs(diff) < 0.05:
        big, sub = "Dead even on projections", f"Week {week} projected points are equal on both sides"
    else:
        winner, loser = (a_name, b_name) if diff > 0 else (b_name, a_name)
        big = f"{_e(winner)} comes out ahead by {abs(diff):.1f}"
        sub = f"Week {week} projected points gained vs. {_e(loser)} (raw projections; ignores roster fit and bye weeks)"
        return f'<div class="verdict"><div class="verdict-big">⚖️ {big}</div><div class="verdict-sub">{sub}</div></div>'
    return f'<div class="verdict"><div class="verdict-big">⚖️ {big}</div><div class="verdict-sub">{sub}</div></div>'


def highlight_cards_html(recap) -> str:
    cards = []

    def add(icon, label, value, sub):
        cards.append(
            f'<div class="hl"><div class="hl-icon">{icon}</div><div class="hl-label">{_e(label)}</div>'
            f'<div class="hl-value">{_e(value)}</div><div class="hl-sub">{_e(sub)}</div></div>'
        )

    add("🔥", "Highest score", recap.highest["team"], f'{recap.highest["score"]:.1f} points')
    add("🧊", "Lowest score", recap.lowest["team"], f'{recap.lowest["score"]:.1f} points')
    if recap.narrowest:
        g = recap.narrowest
        add("😅", "Narrowest win", g["winner"], f'beat {g["loser"]} by {g["margin"]:.1f}')
    if recap.widest:
        g = recap.widest
        add("💥", "Widest win", g["winner"], f'beat {g["loser"]} by {g["margin"]:.1f}')
    if recap.luckiest:
        r = recap.luckiest
        add("🍀", "Luckiest", r["Team"], f'won with a {r["All-Play Record"]} all-play record')
    if recap.unluckiest:
        r = recap.unluckiest
        add("😤", "Unluckiest", r["Team"], f'lost with a {r["All-Play Record"]} all-play record')
    if recap.upsets:
        u = recap.upsets[0]
        add("⚡", "Upset of the week", u["winner"], f'beat {u["loser"]} as a {u["gap"]:.1f}-pt underdog')
    return f'<div class="hl-grid">{"".join(cards)}</div>'


def callouts_html(items: list, cls: str = "") -> str:
    return "".join(f'<div class="callout {cls}">{t}</div>' for t in items)


def diff_table_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Diff (Actual − Proj)"] = out["Diff (Actual − Proj)"].map(diff_html)
    return out


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


def playoff_start_week(year) -> int:
    """First playoff week: week 14 through the 2020 season, week 15 from 2021 on."""
    return 14 if int(year) <= 2020 else 15


def record_eligible(team, year, week, playoff_teams) -> bool:
    """Should this team's score count toward scoring records in `week`?

    Playoff weeks (three rounds) only count for teams still alive in the bracket:
    teams that missed the playoffs are out for all three weeks, quarterfinal losers
    are out for the last two, and semifinal losers still count in the final week
    because the third-place game matters. Regular-season weeks always count."""
    start = playoff_start_week(year)
    if not (start <= week <= start + 2):
        return True
    seed = getattr(team, "standing", 0) or 0
    if isinstance(playoff_teams, int) and playoff_teams > 0 and seed > playoff_teams:
        return False  # missed the playoffs
    if week > start and isinstance(playoff_teams, int) and playoff_teams > 4:
        outcomes = getattr(team, "outcomes", None) or []
        if len(outcomes) >= start and outcomes[start - 1] == "L":
            return False  # lost in the quarterfinals
    return True


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

    # Scoring records: this season so far, then all-time (needs past seasons).
    # Only teams still in contention count (see record_eligible).
    n_playoff = getattr(league.settings, "playoff_team_count", None)
    year = getattr(league, "year", None)

    def counts(team, wk):
        return year is None or record_eligible(team, year, wk, n_playoff)

    prior = [
        float(sc)
        for t in league.teams
        for wk, sc in enumerate(t.scores[: week - 1], start=1)
        if sc and sc > 0 and counts(t, wk)
    ]
    eligible = [x for x in scores if counts(x[2], week)]
    if not eligible:
        return rec
    rec_top = max(eligible, key=lambda x: x[1])
    rec_bottom = min(eligible, key=lambda x: x[1])
    for kind, (name, sc, _) in (("high", rec_top), ("low", rec_bottom)):
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
    first season ESPN can't return). Used only for all-time record callouts; scores from teams already
    eliminated during the playoffs are left out (see record_eligible)."""
    rows = []
    for y in range(year - 1, year - 1 - HISTORY_MAX_SEASONS, -1):
        try:
            past = League(league_id=league_id, year=y, espn_s2=espn_s2, swid=swid)
        except Exception:
            break
        n_playoff = getattr(past.settings, "playoff_team_count", None)
        for t in past.teams:
            for wk, sc in enumerate(t.scores, start=1):
                if sc and sc > 0 and record_eligible(t, y, wk, n_playoff):
                    rows.append({"year": y, "week": wk, "team": t.team_name, "score": float(sc)})
    return rows


# ---------------------------------------------------------------------
# Playoff machine, playoff odds and schedule comparison
# ---------------------------------------------------------------------
PM_SIMS = 10_000          # Monte Carlo runs behind the playoff odds
PM_MIN_SD, PM_MAX_SD = 12.0, 30.0


def regular_season_weeks(year) -> int:
    """Regular-season length: the week before the playoffs start."""
    return playoff_start_week(year) - 1


def playoff_byes(n_playoff) -> int:
    """First-round byes in a standard single-elimination bracket
    (6 teams -> 2 byes, 7 -> 1, 4 or 8 -> none)."""
    if not isinstance(n_playoff, int) or n_playoff < 2:
        return 0
    return (1 << math.ceil(math.log2(n_playoff))) - n_playoff


def _opp_id(opp):
    return opp.team_id if hasattr(opp, "team_id") else int(opp)


def _decided(team, w) -> bool:
    """Has this team's game in week index `w` been played and finalised?"""
    outs = list(getattr(team, "outcomes", None) or [])
    if w < len(outs):
        return outs[w] != "U"
    scores = list(getattr(team, "scores", None) or [])
    return w < len(scores) and bool(scores[w])


def build_season_model(league, reg_weeks: int) -> dict:
    """Everything the playoff tools need from ESPN's data, in plain lists:
    teams (in league order), the not-yet-final regular-season games as
    (week, team_index_a, team_index_b), and each team's played scores."""
    teams = list(league.teams)
    idx = {t.team_id: k for k, t in enumerate(teams)}
    remaining = set()
    played_scores = []
    for k, t in enumerate(teams):
        mine = []
        for w, opp in enumerate(list(t.schedule)[:reg_weeks]):
            oid = _opp_id(opp)
            if oid == t.team_id or oid not in idx:
                continue  # bye week
            if _decided(t, w):
                sc = t.scores[w] if w < len(t.scores) else 0
                if sc and sc > 0:
                    mine.append(float(sc))
            else:
                a, b = sorted((k, idx[oid]))
                remaining.add((w + 1, a, b))
        played_scores.append(mine)
    return {"teams": teams, "idx": idx, "remaining": sorted(remaining), "played": played_scores}


def team_strength(played_scores: list) -> tuple:
    """(mean score per team, score spread). The mean blends the season average
    with the last three games so a hot or cold stretch counts for more; the spread
    is the league-wide game-to-game variation, kept inside a sane range."""
    league_vals = [s for sc in played_scores for s in sc]
    fallback = float(np.mean(league_vals)) if league_vals else 100.0
    means, resid = [], []
    for sc in played_scores:
        if not sc:
            means.append(fallback)
            continue
        season = float(np.mean(sc))
        recent = float(np.mean(sc[-3:]))
        means.append(0.5 * season + 0.5 * recent)
        resid.extend(s - season for s in sc)
    sd = float(np.std(resid, ddof=1)) if len(resid) > 3 else TEAM_SCORE_SD
    return np.array(means), float(min(PM_MAX_SD, max(PM_MIN_SD, sd)))


def simulate_playoff_odds(base_wins, base_pf, mean, sd, games, picks, n_playoff,
                          n_sims: int = PM_SIMS, seed: int = 2026) -> dict:
    """Play out every remaining regular-season game `n_sims` times.

    base_wins : wins so far (ties = 0.5), one per team
    base_pf   : points for so far
    games     : [(team_a, team_b), ...] remaining matchups, as team indexes
    picks     : {game_number: winning team index}; unpicked games are simulated
    A picked game keeps the simulated score pair but hands the higher score to
    the chosen winner, so points-for tiebreakers stay realistic. The random
    numbers are fixed, so changing one pick moves the odds smoothly instead of
    adding sampling noise. Seeds sort by wins, then points for."""
    base_wins = np.asarray(base_wins, float)
    base_pf = np.asarray(base_pf, float)
    n = len(base_wins)
    G = len(games)
    rng = np.random.default_rng(seed)
    wins = np.tile(base_wins, (n_sims, 1))
    pf = np.tile(base_pf, (n_sims, 1))
    if G:
        ga = np.array([g[0] for g in games])
        gb = np.array([g[1] for g in games])
        sa = rng.normal(mean[ga], sd, (n_sims, G))
        sb = rng.normal(mean[gb], sd, (n_sims, G))
        for g, winner in (picks or {}).items():
            flip = (sa[:, g] < sb[:, g]) if winner == ga[g] else (sb[:, g] < sa[:, g])
            sa[flip, g], sb[flip, g] = sb[flip, g].copy(), sa[flip, g].copy()
        a_wins = (sa > sb).astype(float)
        ea = np.zeros((G, n)); eb = np.zeros((G, n))
        ea[np.arange(G), ga] = 1.0
        eb[np.arange(G), gb] = 1.0
        wins += a_wins @ ea + (1.0 - a_wins) @ eb
        pf += sa @ ea + sb @ eb
    key = wins * 1e5 + pf
    order = np.argsort(-key, axis=1)
    seeds = np.empty_like(order)
    np.put_along_axis(seeds, order, np.broadcast_to(np.arange(n), order.shape), axis=1)
    n_po = n_playoff if isinstance(n_playoff, int) and 0 < n_playoff <= n else max(1, n // 2)
    byes = playoff_byes(n_po)
    seed_probs = np.stack([(seeds == s).mean(0) for s in range(n)], axis=1)  # team x seed
    return {
        "playoff": (seeds < n_po).mean(0),
        "bye": (seeds < byes).mean(0) if byes else np.zeros(n),
        "top_seed": (seeds == 0).mean(0),
        "avg_seed": (seeds + 1).mean(0),
        "proj_wins": wins.mean(0),
        "seed_probs": seed_probs,
        "n_playoff": n_po,
        "byes": byes,
    }


def schedule_comparison(league, reg_weeks: int) -> dict:
    """How each team would have done against every team's schedule, using only
    the weeks all teams have finished. If a schedule's opponent that week is the
    team itself, the team faces the schedule's owner instead."""
    teams = list(league.teams)
    idx = {t.team_id: k for k, t in enumerate(teams)}
    n = len(teams)
    weeks = [w for w in range(reg_weeks)
             if all(w < len(t.schedule) and _decided(t, w) and w < len(t.scores) for t in teams)]
    wins = np.zeros((n, n))   # wins[i, j]: team i's record against team j's schedule
    for i in range(n):
        for j in range(n):
            tot = 0.0
            for w in weeks:
                opp = idx.get(_opp_id(teams[j].schedule[w]), j)
                opp_score = teams[j].scores[w] if opp == i else teams[opp].scores[w]
                mine = teams[i].scores[w]
                tot += 1.0 if mine > opp_score else 0.5 if mine == opp_score else 0.0
            wins[i, j] = tot
    return {"teams": teams, "weeks": weeks, "wins": wins, "games": len(weeks)}


def _lerp_color(scale, t: float) -> str:
    t = min(1.0, max(0.0, t))
    for (p0, c0), (p1, c1) in zip(scale, scale[1:]):
        if t <= p1:
            f = 0 if p1 == p0 else (t - p0) / (p1 - p0)
            a, b = _rgb(c0), _rgb(c1)
            return "#%02x%02x%02x" % tuple(round(a[k] + (b[k] - a[k]) * f) for k in range(3))
    return scale[-1][1]


def _ink_for(color: str) -> str:
    r, g, b = _rgb(color)
    return "#0b1220" if (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 > 0.55 else "#ffffff"


def _fmt_wl(w: float, games: int) -> str:
    losses = games - w
    fw = f"{int(w)}" if abs(w - round(w)) < 1e-9 else f"{w:.1f}"
    fl = f"{int(round(losses))}" if abs(losses - round(losses)) < 1e-9 else f"{losses:.1f}"
    return f"{fw}-{fl}"


def schedule_comparison_html(cmp_: dict) -> str:
    """Grid: rows are teams' scores, columns are schedules. The diagonal is the
    team's real record. Colored on the same red-to-green scale as the standings."""
    teams, wins, games = cmp_["teams"], cmp_["wins"], cmp_["games"]
    if not games:
        return ""
    head = "".join(
        f'<th class="sc-h" title="{html.escape(t.team_name)}"><span>{html.escape(t.team_name)}</span></th>'
        for t in teams
    )
    rows = []
    for i, t in enumerate(teams):
        cells = []
        for j in range(len(teams)):
            w = wins[i, j]
            color = _lerp_color(WIN_BAR_COLORSCALE, w / games)
            cls = "sc-cell sc-diag" if i == j else "sc-cell"
            cells.append(f'<td class="{cls}" style="background:{color};color:{_ink_for(color)};">{_fmt_wl(w, games)}</td>')
        rows.append(f'<tr><th class="sc-r">{html.escape(t.team_name)}</th>{"".join(cells)}</tr>')
    return (
        '<div class="ft-wrap"><table class="sc"><thead><tr><th class="sc-corner">Team ↓ · Schedule →</th>'
        f'{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def schedule_luck_df(cmp_: dict) -> pd.DataFrame:
    """Actual wins vs. the average over every schedule (the team's record if the
    schedule were picked at random), plus how strong the real schedule was."""
    teams, wins, games = cmp_["teams"], cmp_["wins"], cmp_["games"]
    n = len(teams)
    rows = []
    for i, t in enumerate(teams):
        actual = float(wins[i, i])
        expected = float(wins[i].mean())
        # strength of schedule = average score of the opponents faced
        weeks = cmp_["weeks"]
        opp_pts = []
        for w in weeks:
            opp = _opp_id(t.schedule[w])
            other = next((x for x in teams if x.team_id == opp), None)
            if other is not None and other is not t:
                opp_pts.append(float(other.scores[w]))
        rows.append({
            "Team": t.team_name,
            "Actual": _fmt_wl(actual, games),
            "Avg wins": expected,
            "Wins vs. avg": actual - expected,
            "Opp. PPG faced": float(np.mean(opp_pts)) if opp_pts else float("nan"),
        })
    df = pd.DataFrame(rows).sort_values("Wins vs. avg", ascending=True, kind="stable").reset_index(drop=True)
    return df


def luck_pill(v) -> str:
    """Positive = schedule helped (green), negative = schedule hurt (red)."""
    v = float(v)
    if abs(v) < 0.25:
        return pill("even", "#64748b")
    return pill(f"{v:+.1f}", "#2a9d5c" if v > 0 else "#d64545")


def _pct(p: float) -> str:
    if p >= 0.9995:
        return "100%"
    if p <= 0.0005:
        return "0%"
    return f"{p * 100:.1f}%"


def pct_cell(p: float, color: str) -> str:
    return (
        '<div class="bar-cell">'
        f'<span class="v">{_pct(p)}</span>'
        '<div class="bar-track">'
        f'<div class="bar-fill" style="width:{max(0.0, min(1.0, p)) * 100:.1f}%;background:{color};"></div></div></div>'
    )


def playoff_odds_df(model: dict, sim: dict, base: dict | None, add_w=None, add_l=None) -> pd.DataFrame:
    """One row per team. `add_w` / `add_l` are the wins and losses handed out by
    the user's picks, so the Record column moves as games are picked."""
    teams = model["teams"]
    add_w = np.zeros(len(teams), int) if add_w is None else add_w
    add_l = np.zeros(len(teams), int) if add_l is None else add_l
    rows = []
    for i, t in enumerate(teams):
        delta = None
        if base is not None:
            delta = (sim["playoff"][i] - base["playoff"][i]) * 100
        rows.append({
            "Team": t.team_name,
            "Record": f"{t.wins + int(add_w[i])}-{t.losses + int(add_l[i])}" + (f"-{t.ties}" if getattr(t, "ties", 0) else ""),
            "Proj. wins": float(sim["proj_wins"][i]),
            "Playoffs": float(sim["playoff"][i]),
            "Change": delta,
            "Bye": float(sim["bye"][i]),
            "Top seed": float(sim["top_seed"][i]),
            "Avg seed": float(sim["avg_seed"][i]),
        })
    return pd.DataFrame(rows).sort_values(["Playoffs", "Avg seed"], ascending=[False, True], kind="stable").reset_index(drop=True)


def odds_table_html(df: pd.DataFrame, n_playoff: int, byes: int, show_change: bool) -> str:
    cols = ["Team", "Record", "Proj. wins", "Playoffs"] + (["Change"] if show_change else []) + (["Bye"] if byes else []) + ["Avg seed"]
    view = df[cols].copy()
    view["Playoffs"] = [pct_cell(p, ACCENT) for p in df["Playoffs"]]
    raw = ["Playoffs"]
    if show_change:
        def chg(v):
            if v is None or (isinstance(v, float) and np.isnan(v)) or abs(v) < 0.05:
                return '<span style="opacity:.5">—</span>'
            return pill(f"{v:+.1f} pts", "#2a9d5c" if v > 0 else "#d64545")
        view["Change"] = [chg(v) for v in df["Change"]]
        raw.append("Change")
    if byes:
        view["Bye"] = [f'<span class="num">{_pct(p)}</span>' for p in df["Bye"]]
        raw.append("Bye")
    # a divider where the field is cut off, based on the expected order
    marks = {}
    if 0 < n_playoff < len(view):
        marks[n_playoff - 1] = f"Top {n_playoff} make the playoffs · by projected odds"
    return html_table(view, raw_cols=tuple(raw), marks=marks, precision=1).replace(
        '<table class="ft">', '<table class="ft ft-nowrap">', 1
    )


def seed_heatmap(model: dict, sim: dict, order: list):
    """Teams x seeds: chance of finishing in each seed, most likely seed darkest."""
    teams = model["teams"]
    names = [teams[i].team_name for i in order]
    z = sim["seed_probs"][order] * 100
    fig = px.imshow(
        z, x=[f"#{s + 1}" for s in range(z.shape[1])], y=names, aspect="auto",
        color_continuous_scale=[[0, "#131c2e"], [1, "#38bdf8"]], zmin=0, zmax=max(40.0, float(z.max())),
        text_auto=".0f", title="Chance of finishing in each seed (%)",
    )
    fig.update_traces(hovertemplate="%{y} · seed %{x}: %{z:.1f}%<extra></extra>")
    fig.update_xaxes(side="top", title=None)
    fig.update_yaxes(title=None)
    fig.update_layout(height=max(280, 36 * len(names) + 110), margin=dict(l=0, r=10, t=80, b=10),
                      coloraxis_showscale=False)
    return style_fig(fig)


def render_playoff_machine(league, year):
    reg_weeks = regular_season_weeks(year)
    model = build_season_model(league, reg_weeks)
    teams, remaining = model["teams"], model["remaining"]
    n_playoff = getattr(league.settings, "playoff_team_count", None)
    mean, sd = team_strength(model["played"])
    base_wins = np.array([t.wins + 0.5 * getattr(t, "ties", 0) for t in teams], float)
    base_pf = np.array([float(t.points_for) for t in teams], float)
    games = [(a, b) for _, a, b in remaining]

    if not remaining:
        st.info("The regular season is complete, so there's nothing left to pick. "
                "See Standings for the final playoff field.")
        return

    # --- state: picks live in widget keys so every click reruns the odds
    keys = [f"pm_{w}_{teams[a].team_id}_{teams[b].team_id}" for w, a, b in remaining]

    def _clear():
        for k in keys:
            st.session_state[k] = None

    def _favorites():
        for k, (_, a, b) in zip(keys, remaining):
            st.session_state[k] = teams[a].team_name if mean[a] >= mean[b] else teams[b].team_name

    st.caption(
        "Pick winners on the left; the odds on the right update instantly. Leave a game "
        "unpicked and it's simulated 10,000 times using each team's scoring."
    )
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        b1, b2, _ = st.columns([1, 1.2, 1])
        b1.button("Reset picks", on_click=_clear, width="stretch")
        b2.button("Favorites win", on_click=_favorites, width="stretch",
                  help="Picks the team with the higher recent scoring average in every game.")
        weeks_left = sorted({w for w, _, _ in remaining})
        picks = {}
        for wk in weeks_left:
            with st.expander(f"Week {wk}", expanded=(wk == weeks_left[0])):
                for g, ((w, a, b), key) in enumerate(zip(remaining, keys)):
                    if w != wk:
                        continue
                    na, nb = teams[a].team_name, teams[b].team_name
                    choice = st.segmented_control(
                        f"{na} vs {nb}", [na, nb], key=key, label_visibility="collapsed",
                        width="stretch",
                    )
                    if choice == na:
                        picks[g] = a
                    elif choice == nb:
                        picks[g] = b
        n_picked = len(picks)

    base = simulate_playoff_odds(base_wins, base_pf, mean, sd, games, {}, n_playoff)
    sim = simulate_playoff_odds(base_wins, base_pf, mean, sd, games, picks, n_playoff) if picks else base
    add_w = np.zeros(len(teams), int)
    add_l = np.zeros(len(teams), int)
    for g, winner in picks.items():
        loser = games[g][1] if games[g][0] == winner else games[g][0]
        add_w[winner] += 1
        add_l[loser] += 1
    df = playoff_odds_df(model, sim, base if picks else None, add_w, add_l)
    with right:
        label = f"{n_picked} of {len(games)} remaining games picked" if picks else f"{len(games)} games left, none picked"
        st.markdown(f"**Playoff odds** · {label}")
        st.html(odds_table_html(df, sim["n_playoff"], sim["byes"], bool(picks)))
        rule = getattr(league.settings, "playoff_seed_tie_rule", "")
        note = "Seeds sort by wins, then points for."
        if rule and rule != "TOTAL_POINTS_SCORED":
            note = f"Seeds sort by wins, then points for (your league's tiebreaker is {rule.replace('_', ' ').lower()}, which isn't modeled)."
        foot = [
            "Records include your picks." if picks else "",
            "Change = shift in playoff odds compared with picking nothing." if picks else "",
            note,
            f"Each unpicked game is simulated 10,000 times: a team's score is drawn from a bell curve centered on a blend "
            f"of its season average and last three games, with a spread of about {sd:.0f} points (the league's typical "
            f"game-to-game swing). Picked games guarantee your winner but keep realistic scores for the points-for "
            f"tiebreaker. Odds use past scoring only, not injuries or ESPN projections.",
        ]
        st.caption(" ".join(x for x in foot if x))
    with st.expander("Seed-by-seed chances", expanded=False):
        names = [t.team_name for t in teams]
        order = [names.index(n) for n in df["Team"]]
        st.plotly_chart(seed_heatmap(model, sim, order), width="stretch")


def render_schedule_comparison(league, year):
    reg_weeks = regular_season_weeks(year)
    cmp_ = schedule_comparison(league, reg_weeks)
    if not cmp_["games"]:
        st.info("Schedule comparison needs at least one completed week.")
        return
    st.caption(
        f"Each row is a team's weekly scores replayed against the column team's schedule over "
        f"{cmp_['games']} completed week{'s' if cmp_['games'] != 1 else ''}. The outlined diagonal is the real record."
    )
    st.html(schedule_comparison_html(cmp_))
    st.write("")
    st.subheader("Schedule luck")
    df = schedule_luck_df(cmp_)
    st.caption("Avg wins = the wins the team averages across every schedule; Wins vs. avg = actual wins minus that. "
               "Positive (green) means the schedule helped; negative (red) means it hurt. "
               "Sorted unluckiest first.")
    view = df.copy()
    view["Wins vs. avg"] = [luck_pill(v) for v in df["Wins vs. avg"]]
    st.html(html_table(view, raw_cols=("Wins vs. avg",), precision=1))


# ---------------------------------------------------------------------
# Trade history
# ---------------------------------------------------------------------
TRADE_FETCH_SIZE = 200     # most trades a season could contain
TRADE_POOL_SIZE = 450      # most-owned players, used as the field for positional ranks
CARD_BATCH = 40            # player cards per ESPN request
TRADE_CLOSE_MARGIN = 5.0   # under this many points since the trade, call it even


def _as_list(x) -> list:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


try:  # the library's own player class, so cards can be built one player at a time
    from espn_api.football.player import Player as EspnPlayer
except Exception:  # pragma: no cover
    EspnPlayer = None


def _note(diag, msg: str):
    if diag is not None and len(diag["errors"]) < 6 and msg not in diag["errors"]:
        diag["errors"].append(msg[:200])


def fetch_player_cards(league, ids, diag=None) -> dict:
    """{playerId: Player} with full-season weekly stats and transaction history.

    Asks ESPN for the raw cards and builds each Player on its own, so one player the
    library can't parse (retired, odd position id) only drops that player instead of
    the whole batch. Falls back to league.player_info when the raw request isn't there."""
    ids = [int(i) for i in dict.fromkeys(ids)]
    out = {}
    req = getattr(league, "espn_request", None)
    raw_ok = EspnPlayer is not None and req is not None and hasattr(req, "get_player_card")
    pro = None
    if raw_ok:
        try:
            pro = league._get_all_pro_schedule()
        except Exception as e:
            _note(diag, f"pro schedule: {type(e).__name__}: {e}")
    final = getattr(league, "finalScoringPeriod", None) or 18
    for k in range(0, len(ids), CARD_BATCH):
        batch = ids[k:k + CARD_BATCH]
        if raw_ok:
            try:
                data = req.get_player_card(batch, final)
            except Exception as e:
                _note(diag, f"player cards: {type(e).__name__}: {e}")
                continue
            wraps = data.get("players") if isinstance(data, dict) else None
            if not wraps:
                _note(diag, "player cards: ESPN returned no players")
            for w in wraps or []:
                try:
                    p = EspnPlayer(w, league.year, pro, player_map=getattr(league, "player_map", None),
                                   get_team_data=getattr(league, "get_team_data", None))
                except Exception as e:
                    _note(diag, f"player parse: {type(e).__name__}: {e}")
                    continue
                out[int(p.playerId)] = p
        else:
            try:
                got = _as_list(league.player_info(playerId=batch))
            except Exception as e:
                _note(diag, f"player cards: {type(e).__name__}: {e}")
                continue
            for p in got:
                pid = getattr(p, "playerId", None)
                if pid is not None:
                    out[int(pid)] = p
    return out


def weekly_points(player) -> dict:
    """{week: (fantasy points, played?)} from a Player's stats; real games only."""
    out = {}
    for wk, s in (getattr(player, "stats", None) or {}).items():
        try:
            w = int(wk)
        except (TypeError, ValueError):
            continue
        if w <= 0 or not isinstance(s, dict) or "points" not in s:
            continue
        pts = float(s.get("points") or 0)
        bd = s.get("breakdown")
        out[w] = (pts, bool(bd) if bd is not None else pts != 0)
    return out


def window_stats(weekly: dict, lo: int, hi: int) -> dict:
    """Total points, games played and points per game for weeks lo..hi."""
    pts, games = 0.0, 0
    for w, (p, played) in weekly.items():
        if lo <= w <= hi:
            pts += p
            games += 1 if played else 0
    return {"pts": round(pts, 1), "games": games, "ppg": round(pts / games, 1) if games else None}


def positional_rank(pid, pos, field: dict, lo: int, hi: int):
    """1 = most points at the position over weeks lo..hi, among the field of players."""
    if not pos or pid not in field or lo > hi:
        return None
    mine = window_stats(field[pid]["weekly"], lo, hi)
    if not mine["games"]:
        return None
    better = sum(
        1 for q, info in field.items()
        if q != pid and info["pos"] == pos and window_stats(info["weekly"], lo, hi)["pts"] > mine["pts"]
    )
    return better + 1


def split_week(player, trade_dt, week_starts: dict, last_week: int) -> int:
    """The first week whose game for this player kicks off after the trade; games from
    that week on count as 'after'. Falls back to league-wide kickoff dates."""
    def date_of(v):
        return v.get("date") if isinstance(v, dict) else v
    sched = {}
    for k, v in (getattr(player, "schedule", None) or {}).items():
        try:
            sched[int(k)] = date_of(v)
        except (TypeError, ValueError):
            continue
    cand = sched or week_starts
    for wk in sorted(cand):
        if cand[wk] is not None and cand[wk] > trade_dt:
            return wk
    return (max(cand) + 1) if cand else last_week + 1


def _pid_of(x):
    """A player id from a Player object, or from a bare id when ESPN couldn't build one."""
    pid = getattr(x, "playerId", None)
    if pid is None and isinstance(x, int) and not isinstance(x, bool):
        pid = x
    try:
        return int(pid) if pid is not None else None
    except (TypeError, ValueError):
        return None


def parse_trade_activities(activities) -> list:
    """Group ESPN's trade messages into trades: each has a date and the players that
    moved, with who sent and who received them. A move whose player ESPN couldn't
    resolve keeps its bare id (past-season players often aren't on any roster now);
    anything with no id at all is skipped."""
    trades = []
    for act in activities or []:
        sent, recv = {}, {}
        for action in getattr(act, "actions", []) or []:
            team, kind, player = action[0], action[1], action[2]
            pid = _pid_of(player)
            if pid is None:
                continue
            name = getattr(team, "team_name", None) or "Unknown team"
            obj = player if hasattr(player, "playerId") else None
            if kind == "TRADE_SENT":
                sent[pid] = (name, obj)
            elif kind == "TRADE_RECEIVED":
                recv[pid] = (name, obj)
        moves = []
        for pid in dict.fromkeys(list(sent) + list(recv)):
            frm, to = sent.get(pid), recv.get(pid)
            obj = (frm or to)[1] or (to or frm)[1]
            moves.append({"pid": pid, "player": obj,
                          "from": frm[0] if frm else "?", "to": to[0] if to else "?"})
        if not moves:
            continue
        try:
            when = datetime.fromtimestamp(float(act.date) / 1000.0)
        except (TypeError, ValueError, AttributeError, OSError):
            continue
        trades.append({"date": when, "moves": moves})
    trades.sort(key=lambda t: t["date"], reverse=True)
    return trades


def fetch_trade_feed(league, diag: dict) -> list:
    """Trades from ESPN's activity feed, paged until it runs dry."""
    out, seen = [], set()
    for page in range(6):
        try:
            batch = list(league.recent_activity(size=TRADE_FETCH_SIZE, msg_type="TRADED", offset=page * TRADE_FETCH_SIZE))
        except Exception as e:
            diag["errors"].append(f"activity feed: {type(e).__name__}: {e}"[:160])
            break
        diag["feed_topics"] += len(batch)
        new = 0
        for act in batch:
            key = (getattr(act, "date", None), tuple(sorted(str(_pid_of(a[2])) for a in getattr(act, "actions", []) or [])))
            if key in seen:
                continue
            seen.add(key)
            out.append(act)
            new += 1
        if len(batch) < TRADE_FETCH_SIZE or not new:
            break
    return out


def trades_from_transactions(league, diag: dict) -> list:
    """Second source: the executed-trade rows in ESPN's transaction log, week by week.
    Used when the activity feed has nothing for the season."""
    raw, seen = [], set()
    top = int(getattr(league, "finalScoringPeriod", 0) or 18)

    def team_name(tid):
        t = league.get_team_data(tid) if tid is not None else None
        return getattr(t, "team_name", None) or "?"

    for sp in range(1, top + 1):
        txns = []
        for fill in (False, True):
            try:
                txns = list(league.transactions(scoring_period=sp, types={"TRADE_ACCEPT"}, fill_trade_items=fill))
            except Exception as e:
                diag["errors"].append(f"transactions wk {sp}: {type(e).__name__}"[:120])
                txns = []
                break
            if all(any(getattr(i, "type", None) == "TRADE" for i in (t.items or [])) for t in txns):
                break  # every trade row already lists its players
        for t in txns:
            if getattr(t, "type", None) != "TRADE_ACCEPT" or getattr(t, "status", "EXECUTED") != "EXECUTED":
                continue
            key = str(getattr(t, "related_transaction_id", None) or getattr(t, "id", None) or id(t))
            legs = [i for i in (t.items or []) if getattr(i, "type", None) == "TRADE"]
            diag["txn_rows"] += 1
            if key in seen or not legs:
                continue
            seen.add(key)
            moves = [{"pid": int(i.playerId), "player": None, "from": team_name(i.from_team_id),
                      "to": team_name(i.to_team_id)} for i in legs if getattr(i, "playerId", None) is not None]
            try:
                when = datetime.fromtimestamp(float(t.date) / 1000.0)
            except (TypeError, ValueError, AttributeError, OSError):
                continue
            if moves:
                raw.append({"date": when, "moves": moves})
    raw.sort(key=lambda r: r["date"], reverse=True)
    return raw


def trades_from_player_cards(cards: dict, league, diag: dict) -> list:
    """Third source, for seasons where ESPN's feed and transaction log are empty (past
    seasons usually): every player card carries that player's own transaction history,
    including executed trades with all of their legs."""
    def team_name(tid):
        t = league.get_team_data(tid) if tid is not None else None
        return getattr(t, "team_name", None) or "?"

    merged = {}
    for p in cards.values():
        txns = getattr(p, "transactions", None) or []
        if txns:
            diag["cards_with_txns"] += 1
        for t in txns:
            diag["txn_types"][f'{getattr(t, "type", "?")}/{getattr(t, "status", "?")}'] += 1
            if getattr(t, "type", None) != "TRADE_ACCEPT" or getattr(t, "status", None) != "EXECUTED":
                continue
            legs = [i for i in (t.items or []) if getattr(i, "type", None) == "TRADE" and getattr(i, "playerId", None) is not None]
            if not legs:
                continue
            key = str(getattr(t, "related_transaction_id", None) or getattr(t, "id", None))
            entry = merged.setdefault(key, {"date": getattr(t, "date", None), "legs": {}})
            for i in legs:
                entry["legs"][int(i.playerId)] = (i.from_team_id, i.to_team_id)
            entry["date"] = entry["date"] or getattr(t, "date", None)
    raw = []
    for e in merged.values():
        try:
            when = datetime.fromtimestamp(float(e["date"]) / 1000.0)
        except (TypeError, ValueError, OSError):
            continue
        raw.append({"date": when, "moves": [
            {"pid": pid, "player": None, "from": team_name(f), "to": team_name(to)} for pid, (f, to) in e["legs"].items()]})
    diag["card_trades"] = len(raw)
    raw.sort(key=lambda r: r["date"], reverse=True)
    return raw


def build_trades(league, year) -> dict:
    """Every trade of the season, each player's positional rank, points and PPG before
    and after the trade (regular season only), and who came out ahead."""
    reg = regular_season_weeks(year)
    cw = getattr(league, "current_week", reg) or reg
    last_week = reg if cw > reg else max(0, cw - 1)
    diag = {"feed_topics": 0, "txn_rows": 0, "source": "none", "dropped": 0, "errors": [],
            "cards": 0, "cards_with_txns": 0, "card_trades": 0, "txn_types": Counter()}
    result = {"trades": [], "last_week": last_week, "year": year, "diag": diag}

    def pool_ids():
        try:
            return list(league.espn_request.get_player_pool_ids(max(1, last_week), 500))[:TRADE_POOL_SIZE]
        except Exception as e:
            _note(diag, f"player pool: {type(e).__name__}: {e}")
            return [p.playerId for t in getattr(league, "teams", []) for p in getattr(t, "roster", [])]

    cards, pool = {}, None
    raw = parse_trade_activities(fetch_trade_feed(league, diag))
    if raw:
        diag["source"] = "activity feed"
    else:
        raw = trades_from_transactions(league, diag)
        if raw:
            diag["source"] = "transaction log"
    if not raw:
        pool = pool_ids()
        rostered = [p.playerId for t in getattr(league, "teams", []) for p in getattr(t, "roster", [])]
        cards = fetch_player_cards(league, rostered + pool, diag)
        raw = trades_from_player_cards(cards, league, diag)
        if raw:
            diag["source"] = "player cards"
    diag["cards"] = len(cards)
    if not raw:
        return result

    traded_ids = [m["pid"] for t in raw for m in t["moves"]]
    if pool is None:
        pool = pool_ids()
    missing = [i for i in traded_ids + pool if i not in cards]
    cards.update(fetch_player_cards(league, missing, diag))
    for t in raw:  # players ESPN can't identify at all (draft picks, stale ids) can't be scored
        keep = [m for m in t["moves"] if m["player"] is not None or m["pid"] in cards]
        diag["dropped"] += len(t["moves"]) - len(keep)
        t["moves"] = keep
    raw = [t for t in raw if t["moves"]]
    if not raw:
        return result

    field, week_starts = {}, {}
    for pid, p in cards.items():
        field[pid] = {"pos": getattr(p, "position", "") or "", "weekly": weekly_points(p)}
        for k, v in (getattr(p, "schedule", None) or {}).items():
            try:
                d = v.get("date") if isinstance(v, dict) else v
                wk = int(k)
            except (TypeError, ValueError, AttributeError):
                continue
            if d is not None and (wk not in week_starts or d < week_starts[wk]):
                week_starts[wk] = d

    for t in raw:
        rows, teams = [], []
        for m in t["moves"]:
            p = cards.get(m["pid"], m["player"])
            pos = getattr(p, "position", "") or ""
            split = split_week(p, t["date"], week_starts, last_week)
            weekly = field.get(m["pid"], {}).get("weekly") or weekly_points(p)
            field.setdefault(m["pid"], {"pos": pos, "weekly": weekly})
            side = {}
            for label, lo, hi in (("before", 1, split - 1), ("after", split, last_week)):
                st_ = window_stats(weekly, lo, hi)
                st_["rank"] = positional_rank(m["pid"], pos, field, lo, hi)
                side[label] = st_
            rows.append({"name": getattr(p, "name", str(m["pid"])), "pos": pos, "from": m["from"],
                         "to": m["to"], "split": split, **side})
            for n in (m["from"], m["to"]):
                if n != "?" and n not in teams:
                    teams.append(n)
        net = {n: 0.0 for n in teams}
        for r in rows:
            pts = r["after"]["pts"]
            if r["to"] in net:
                net[r["to"]] += pts
            if r["from"] in net:
                net[r["from"]] -= pts
        winner, margin = None, 0.0
        if net:
            ordered = sorted(net.items(), key=lambda kv: -kv[1])
            winner, margin = ordered[0][0], ordered[0][1] - (ordered[1][1] if len(ordered) > 1 else 0.0)
            margin = margin / 2 if len(net) == 2 else margin
            if margin < TRADE_CLOSE_MARGIN:
                winner = None
        result["trades"].append({"date": t["date"], "teams": teams, "players": rows,
                                 "net": net, "winner": winner, "margin": round(margin, 1),
                                 "week": min(r["split"] for r in rows)})
    return result


def _rank_text(pos, rank) -> str:
    return f"{pos}{rank}" if rank and pos else "—"


def _num_text(v, fmt="{:.1f}") -> str:
    return "—" if v is None else fmt.format(v)


def trade_card_html(trade: dict, last_week: int) -> str:
    when = trade["date"].strftime("%b %d, %Y")
    title = " ⇄ ".join(html.escape(n) for n in trade["teams"]) or "Trade"
    head = (
        '<tr><th rowspan="2">Player</th><th rowspan="2">Pos</th><th rowspan="2">Moved</th>'
        '<th colspan="3" class="grp-h">Before the trade</th><th colspan="3" class="grp-h">After the trade</th></tr>'
        '<tr><th>Rank</th><th>Pts</th><th>PPG</th><th>Rank</th><th>Pts</th><th>PPG</th></tr>'
    )
    body = []
    for r in trade["players"]:
        b, a = r["before"], r["after"]
        body.append(
            "<tr>"
            f'<td><b>{html.escape(r["name"])}</b></td><td>{html.escape(r["pos"] or "—")}</td>'
            f'<td class="trade-mv">{html.escape(r["from"])} → {html.escape(r["to"])}</td>'
            f'<td class="num">{_rank_text(r["pos"], b["rank"])}</td><td class="num">{_num_text(b["pts"]) if b["games"] else "—"}</td>'
            f'<td class="num">{_num_text(b["ppg"])}</td>'
            f'<td class="num">{_rank_text(r["pos"], a["rank"])}</td><td class="num">{_num_text(a["pts"]) if a["games"] else "—"}</td>'
            f'<td class="num">{_num_text(a["ppg"])}</td></tr>'
        )
    if trade["winner"]:
        verdict = pill(f'{trade["winner"]} ahead by {trade["margin"]:.1f} pts', "#2a9d5c")
    elif any(r["after"]["games"] for r in trade["players"]):
        verdict = pill("Too close to call", "#64748b")
    else:
        verdict = pill("No games played since the trade", "#64748b")
    net = " · ".join(
        f'{html.escape(n)} <b>{v:+.1f}</b>' for n, v in sorted(trade["net"].items(), key=lambda kv: -kv[1])
    )
    return (
        '<div class="trade-card">'
        f'<div class="trade-head"><div><div class="trade-title">{title}</div>'
        f'<div class="trade-sub">{when} · games from week {trade["week"]} on count as "after"</div></div>{verdict}</div>'
        f'<div class="ft-wrap"><table class="ft ft-trade"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>'
        f'<div class="trade-net">Points since the trade, received minus sent: {net}</div>'
        '</div>'
    )


# ---------------------------------------------------------------------
# League history
# ---------------------------------------------------------------------
def _owner_key(team) -> str:
    owners = getattr(team, "owners", None) or []
    own = owners[0] if owners else {}
    return str(own.get("id") or f"team-{team.team_id}")


def _owner_label(team) -> str:
    owners = getattr(team, "owners", None) or []
    own = owners[0] if owners else {}
    return str(own.get("firstName") or own.get("displayName") or "")


def season_snapshot(league, year) -> dict:
    """One season boiled down to plain data: for each team (keyed by its owner, since
    team names change) every finished regular-season game with both scores."""
    reg = regular_season_weeks(year)
    teams = list(league.teams)
    by_id = {t.team_id: t for t in teams}
    out = []
    for t in teams:
        games = []
        for w, opp in enumerate(list(t.schedule)[:reg]):
            if w >= len(t.scores) or not _decided(t, w):
                continue
            o = opp if hasattr(opp, "team_id") else by_id.get(opp)
            if o is None or o is t or w >= len(o.scores):
                continue
            pf, pa = float(t.scores[w] or 0), float(o.scores[w] or 0)
            if pf <= 0 and pa <= 0:
                continue
            games.append({"week": w + 1, "pf": pf, "pa": pa, "opp": _owner_key(o), "opp_team": o.team_name})
        out.append({"key": _owner_key(t), "owner": _owner_label(t), "team": t.team_name,
                    "finish": int(getattr(t, "final_standing", 0) or 0), "games": games})
    return {"year": int(year), "teams": out}


@st.cache_data(show_spinner="Loading past seasons for league history...", ttl=24 * 60 * 60)
def get_past_season_snapshots(league_id: int, year: int, espn_s2: str, swid: str) -> list:
    """Snapshots of every season before `year`, newest first, stopping at the first
    season ESPN can't return."""
    snaps = []
    for y in range(year - 1, year - 1 - HISTORY_MAX_SEASONS, -1):
        try:
            past = League(league_id=league_id, year=y, espn_s2=espn_s2, swid=swid)
        except Exception:
            break
        snaps.append(season_snapshot(past, y))
    return snaps


def _result(pf, pa) -> str:
    return "W" if pf > pa else "L" if pf < pa else "T"


def season_rows(snapshots: list) -> list:
    """One row per owner per season: record, average points for / against, finish,
    and the individual scores behind the averages."""
    rows = []
    for snap in snapshots:
        for t in snap["teams"]:
            g = t["games"]
            if not g:
                continue
            res = [_result(x["pf"], x["pa"]) for x in g]
            rows.append({
                "key": t["key"], "owner": t["owner"], "team": t["team"], "year": snap["year"],
                "finish": t["finish"], "w": res.count("W"), "l": res.count("L"), "t": res.count("T"),
                "pf_avg": float(np.mean([x["pf"] for x in g])), "pa_avg": float(np.mean([x["pa"] for x in g])),
                "games": g,
            })
    return rows


def head_to_head(snapshots: list) -> dict:
    """{(owner, opponent): {'w','l','t','pf': [...], 'pa': [...]}} across every season."""
    h2h = {}
    for snap in snapshots:
        for t in snap["teams"]:
            for x in t["games"]:
                if x["opp"] == t["key"]:
                    continue
                d = h2h.setdefault((t["key"], x["opp"]), {"w": 0, "l": 0, "t": 0, "pf": [], "pa": []})
                d[{"W": "w", "L": "l", "T": "t"}[_result(x["pf"], x["pa"])]] += 1
                d["pf"].append(x["pf"])
                d["pa"].append(x["pa"])
    return h2h


def latest_names(snapshots: list) -> dict:
    """{owner key: (latest team name, owner first name)}, from the newest season each appears in."""
    names = {}
    for snap in sorted(snapshots, key=lambda s: s["year"]):
        for t in snap["teams"]:
            names[t["key"]] = (t["team"], t["owner"])
    return names


def _tip(text: str) -> str:
    return html.escape(text, quote=True).replace("\n", "&#10;")


def scores_tip(games: list, field: str) -> str:
    lines = []
    for x in games:
        lines.append(f'Wk {x["week"]}: {x[field]:.2f}  vs {x["opp_team"]} ({_result(x["pf"], x["pa"])})')
    return "\n".join(lines)


def record_text(w, l, t=0) -> str:
    return f"{w}-{l}" + (f"-{t}" if t else "")


def _pct_text(w, l, t) -> str:
    n = w + l + t
    return "—" if not n else f"{(w + 0.5 * t) / n * 100:.0f}%"


def season_history_html(rows: list, names: dict, keys: list) -> str:
    """Grouped table: an all-time line per team, then one row per season (newest first).
    Hover an average to see the scores behind it."""
    head = ("<th>Season</th><th>Team name</th><th>Finish</th><th>Record</th><th>Win %</th>"
            "<th>Avg PF</th><th>Avg PA</th>")
    body = []
    for key in keys:
        mine = sorted((r for r in rows if r["key"] == key), key=lambda r: -r["year"])
        if not mine:
            continue
        w, l, t = (sum(r[k] for r in mine) for k in ("w", "l", "t"))
        allg = [g for r in mine for g in r["games"]]
        pf = float(np.mean([g["pf"] for g in allg]))
        pa = float(np.mean([g["pa"] for g in allg]))
        team, owner = names.get(key, (mine[0]["team"], mine[0]["owner"]))
        label = html.escape(team) + (f' <span class="grp-owner">{html.escape(owner)}</span>' if owner else "")
        body.append(
            f'<tr class="grp"><td colspan="3">{label}</td><td class="num">{record_text(w, l, t)}</td>'
            f'<td class="num">{_pct_text(w, l, t)}</td>'
            f'<td class="num"><span class="tipnum" title="{_tip(scores_tip(allg, "pf"))}">{pf:.1f}</span></td>'
            f'<td class="num"><span class="tipnum" title="{_tip(scores_tip(allg, "pa"))}">{pa:.1f}</span></td></tr>'
        )
        for r in mine:
            fin = f'{r["finish"]}' if r["finish"] else "—"
            body.append(
                f'<tr><td>{r["year"]}</td><td>{html.escape(r["team"])}</td><td class="num">{fin}</td>'
                f'<td class="num">{record_text(r["w"], r["l"], r["t"])}</td><td class="num">{_pct_text(r["w"], r["l"], r["t"])}</td>'
                f'<td class="num"><span class="tipnum" title="{_tip(scores_tip(r["games"], "pf"))}">{r["pf_avg"]:.1f}</span></td>'
                f'<td class="num"><span class="tipnum" title="{_tip(scores_tip(r["games"], "pa"))}">{r["pa_avg"]:.1f}</span></td></tr>'
            )
    return f'<div class="ft-wrap"><table class="ft ft-hist"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def h2h_matrix_html(h2h: dict, names: dict, order: list) -> str:
    """Win % of each row team against each column team, all seasons combined. Colored on
    the same red-to-green scale; hover a cell for the record and average scores."""
    head = "".join(
        f'<th class="sc-h" title="{html.escape(names[k][0])}"><span>{html.escape(names[k][0])}</span></th>' for k in order
    )
    rows = []
    for a in order:
        cells = []
        for b in order:
            d = h2h.get((a, b))
            if a == b or not d:
                cells.append('<td class="sc-cell sc-na">—</td>')
                continue
            n = d["w"] + d["l"] + d["t"]
            p = (d["w"] + 0.5 * d["t"]) / n
            color = _lerp_color(WIN_BAR_COLORSCALE, p)
            tip = (f'{names[a][0]} vs {names[b][0]}\n{record_text(d["w"], d["l"], d["t"])} over {n} games\n'
                   f'Avg PF {np.mean(d["pf"]):.1f} · Avg PA {np.mean(d["pa"]):.1f}')
            cells.append(
                f'<td class="sc-cell" title="{_tip(tip)}" style="background:{color};color:{_ink_for(color)};">'
                f'{p * 100:.0f}%<small>{record_text(d["w"], d["l"], d["t"])}</small></td>'
            )
        rows.append(f'<tr><th class="sc-r">{html.escape(names[a][0])}</th>{"".join(cells)}</tr>')
    return (
        '<div class="ft-wrap"><table class="sc"><thead><tr><th class="sc-corner">Team ↓ · vs →</th>'
        f'{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def h2h_detail_df(h2h: dict, names: dict, key: str, order: list) -> pd.DataFrame:
    rows = []
    for b in order:
        d = h2h.get((key, b))
        if b == key or not d:
            continue
        n = d["w"] + d["l"] + d["t"]
        pf, pa = float(np.mean(d["pf"])), float(np.mean(d["pa"]))
        rows.append({"Opponent": names[b][0], "Record": record_text(d["w"], d["l"], d["t"]),
                     "Win %": (d["w"] + 0.5 * d["t"]) / n * 100, "Avg PF": pf, "Avg PA": pa,
                     "Avg margin": pf - pa, "Games": n})
    return pd.DataFrame(rows).sort_values("Win %", ascending=False, kind="stable").reset_index(drop=True) if rows else pd.DataFrame()


def render_trade_history(league_id, year, espn_s2, swid):
    first = 2019  # ESPN only serves trade activity from 2019 on
    years = list(range(first, int(year) + 1))
    c1, c2 = st.columns([1, 2])
    sel_year = c1.selectbox("Season", years[::-1], index=0, key="trade_year")
    try:
        data = get_trade_history(int(league_id), int(sel_year), espn_s2, swid)
    except Exception as e:
        st.error(f"Couldn't load trades for {sel_year} from ESPN. ({e})")
        return
    trades = data["trades"]
    if not trades:
        st.info(f"No trades were found for the {sel_year} season.")
        d = data.get("diag") or {}
        st.caption(
            f"Checked ESPN's activity feed ({d.get('feed_topics', 0)} trade entries), the transaction log "
            f"({d.get('txn_rows', 0)} executed-trade rows) and {d.get('cards', 0)} player cards "
            f"({d.get('cards_with_txns', 0)} with transactions, {d.get('card_trades', 0)} trades found). "
            + ("Transaction types seen: " + ", ".join(f"{k} x{v}" for k, v in sorted(d.get("txn_types", {}).items())[:8]) + ". "
               if d.get("txn_types") else "")
            + (f"{d['dropped']} traded item(s) couldn't be matched to a player. " if d.get("dropped") else "")
            + ("Errors: " + "; ".join(d["errors"][:3]) if d.get("errors") else "")
        )
        return
    teams = sorted({n for t in trades for n in t["teams"]})
    picked = c2.multiselect("Filter by team", teams, key="trade_teams", placeholder="All teams",
                            filter_mode=None)
    shown = [t for t in trades if not picked or any(n in picked for n in t["teams"])]
    st.caption(f"{len(shown)} of {len(trades)} trade{'s' if len(trades) != 1 else ''} in {sel_year}, newest first.")
    if not shown:
        st.info("No trades match that team filter.")
        return
    for t in shown:
        st.html(trade_card_html(t, data["last_week"]))
    st.caption(
        "Positional rank is each player's rank by total fantasy points among the most-owned players at his position, "
        "for the games before the trade and for the games after it. The 'after' window runs through week "
        f"{data['last_week']} (regular season only). Points since the trade, received minus sent, decide who is ahead; "
        f"anything under {TRADE_CLOSE_MARGIN:.0f} points is called too close. Draft picks in a trade aren't counted."
    )


@st.cache_data(show_spinner="Loading trades and player stats...", ttl=6 * 60 * 60)
def get_trade_history(league_id: int, year: int, espn_s2: str, swid: str) -> dict:
    return build_trades(get_league(league_id, year, espn_s2, swid), year)


def render_league_history(league, league_id, year, espn_s2, swid):
    try:
        past = get_past_season_snapshots(int(league_id), int(year), espn_s2, swid)
    except Exception as e:
        st.error(f"Couldn't load past seasons. ({e})")
        return
    snapshots = [season_snapshot(league, year)] + list(past)
    rows = season_rows(snapshots)
    if not rows:
        st.info("League history needs at least one completed game.")
        return
    names = latest_names(snapshots)
    h2h = head_to_head(snapshots)
    all_keys = {r["key"] for r in rows}

    def win_pct(k):
        mine = [r for r in rows if r["key"] == k]
        w, l, t = (sum(r[x] for r in mine) for x in ("w", "l", "t"))
        return (w + 0.5 * t) / max(1, w + l + t)

    order = sorted(all_keys, key=lambda k: -win_pct(k))
    years_n = len({r["year"] for r in rows})
    st.subheader("Season by season")
    labels = {k: names[k][0] for k in order}
    picked = st.multiselect("Teams", order, format_func=lambda k: labels[k], key="hist_teams",
                            placeholder="All teams", filter_mode=None)
    st.html(season_history_html(rows, names, picked or order))
    st.caption(
        f"{years_n} season{'s' if years_n != 1 else ''} of regular-season games, grouped by owner (team names change from year "
        "to year). Each group's first line is the all-time total. Hover an Avg PF or Avg PA number to see every score behind it. "
        "Finish is the final league rank."
    )
    st.write("")
    st.subheader("All-time head-to-head")
    st.html(h2h_matrix_html(h2h, names, order))
    st.caption("Win % of the row team against the column team, with the record underneath. Hover a cell for average points for and against.")
    pick = st.selectbox("Team detail", order, format_func=lambda k: labels[k], key="hist_detail")
    det = h2h_detail_df(h2h, names, pick, order)
    if det.empty:
        st.info("No head-to-head games yet for this team.")
    else:
        st.html(html_table(det, bars={"Win %": (100, ACCENT)}, precision=1))
    st.caption("Regular-season games only. Ties count as half a win.")


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

    tab1, tab2, tab3, tab_pm, tab_sc, tab4, tab5, tab6, tab7, tab_th, tab_lh, tab8 = st.tabs(
        [
            "Standings",
            "Rosters & Lineup",
            "Matchups",
            "Playoff Machine",
            "Schedule Comparison",
            "Power Rankings",
            "Free Agents",
            "Injury Report",
            "Trade Analyzer",
            "Trade History",
            "League History",
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
        st.html(podium_html(standings_df))
        st.html(standings_table_html(
            standings_df, getattr(league.settings, "playoff_team_count", None)
        ))
        st.write("")
        st.plotly_chart(build_points_chart(standings_df), width="stretch")
        st.write("")
        trend_fig = build_score_trend_chart(league)
        if trend_fig is not None:
            st.plotly_chart(trend_fig, width="stretch")
        else:
            st.info("Weekly scores will chart here once games are played.")
        st.write("")
        st.plotly_chart(build_pf_pa_chart(standings_df), width="stretch")
        st.caption(
            "Points For vs. Points Against: top-left teams score a lot and allow little. "
            "Dashed lines mark the league averages."
        )

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
        try:
            week_boxes = get_week_boxes(league, int(league_id), int(year), roster_week)
            stat_cards(week_summary_cards(team_week_summary(week_boxes, team), roster_week))
        except Exception:
            pass  # the cards are a nicety; never block the roster
        st.html(lineup_cards_html(roster_df, roster_week))
        with st.expander("Full roster table (season projection and averages)"):
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
                st.html(swap_cards_html(res.swaps_df, roster_week))

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
                    st.html(fa_cards_html(res.fa_df, roster_week))
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
        matchup_rows = build_matchup_rows(
            league, int(week),
            boxes=get_week_boxes(league, int(league_id), int(year), int(week)),
        )
        stat_cards(matchup_summary(matchup_rows))
        st.write("")
        st.html(matchups_html(matchup_rows, bar=True))
        st.caption(
            "Each score shows ESPN's projected score in gray parentheses, "
            "then the actual score. The leading side is highlighted, and the bar "
            "under each game is our win probability estimate, built from ESPN's "
            "projections and the points already scored (not ESPN's own figure)."
        )

    # --- Playoff Machine (picks + live odds) ---
    with tab_pm:
        render_playoff_machine(league, year)

    # --- Schedule Comparison ---
    with tab_sc:
        render_schedule_comparison(league, year)

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
            prev_df = None
            if int(pr_week) > 1:
                try:
                    prev_df = build_power_rankings_df(league, int(pr_week) - 1)
                except Exception:
                    prev_df = None
            st.html(power_table_html(pr_df, prev_df))
            st.caption("Move = change in power rank since the previous week.")
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
            proj_col = f"Week {int(fa_week)} Proj"
            show_table(
                fa_df,
                pills={"Injury Status": status_pill},
                bars={
                    proj_col: (float(fa_df[proj_col].max() or 1), ACCENT),
                    "% Owned": (100.0, "#a78bfa"),
                },
                max_height=700,
            )

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
                for team_name_, grp in injury_df.groupby("Team", sort=False):
                    with st.expander(f"{team_name_} · {len(grp)} flagged", expanded=True):
                        show_table(
                            grp.drop(columns="Team").reset_index(drop=True),
                            pills={"Status": status_pill},
                        )

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

            a_name = by_key[pick_a[0]][1].team_name if pick_a else "Side 1"
            b_name = by_key[pick_b[0]][1].team_name if pick_b else "Side 2"
            sums = {
                side: float(pd.to_numeric(compare_df[compare_df["Side"] == side][wk_col], errors="coerce").sum())
                for side in ("Side 1", "Side 2")
            }
            both = bool(pick_a and pick_b)
            if both:
                st.html(trade_verdict_html(a_name, b_name, sums["Side 1"], sums["Side 2"], trade_week))
            tcol_a, tcol_b = st.columns(2)
            for col, side, name, other in ((tcol_a, "Side 1", a_name, "Side 2"), (tcol_b, "Side 2", b_name, "Side 1")):
                with col:
                    sub = compare_df[compare_df["Side"] == side]
                    if sub.empty:
                        st.caption("No players selected yet.")
                        continue
                    # the team that RECEIVES more projected points is favored
                    favored = both and sums[other] > sums[side]
                    st.html(trade_side_html(name, sub, wk_col, favored))
            st.write("")

            fig = px.bar(
                compare_df.assign(Owner=compare_df["Fantasy Owner"]),
                x="Player", y=wk_col, color="Owner",
                title=f"Week {trade_week} Projected Points",
                color_discrete_sequence=PALETTE,
            )
            chart_a, chart_b = st.columns(2)
            with chart_a:
                st.plotly_chart(style_fig(fig), width="stretch")
            with chart_b:
                if both:
                    tot_rows = []
                    for side, name in (("Side 1", a_name), ("Side 2", b_name)):
                        sub = compare_df[compare_df["Side"] == side]
                        tot_rows.append({"Team": name, "Metric": f"Week {trade_week} proj",
                                         "Points": float(pd.to_numeric(sub[wk_col], errors="coerce").sum())})
                        tot_rows.append({"Team": name, "Metric": "PPG",
                                         "Points": float(pd.to_numeric(sub["PPG"], errors="coerce").sum())})
                    fig2 = px.bar(
                        pd.DataFrame(tot_rows), x="Metric", y="Points", color="Team", barmode="group",
                        title="Totals each side gives up", color_discrete_sequence=PALETTE,
                    )
                    st.plotly_chart(style_fig(fig2), width="stretch")

    # --- Trade History ---
    with tab_th:
        render_trade_history(league_id, year, espn_s2, swid)

    # --- League History ---
    with tab_lh:
        render_league_history(league, league_id, year, espn_s2, swid)

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
        recap_boxes = get_week_boxes(league, int(league_id), int(year), int(recap_week))
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
            st.html(highlight_cards_html(recap))

            if recap.records:
                st.subheader("Record performances")
                items = []
                for r in recap.records:
                    kind = "Highest" if r["kind"] == "high" else "Lowest"
                    prev_label = "record" if r["scope"] == "league history" else "mark this season"
                    items.append(
                        f"🏆 <b>{kind} score in {_e(r['scope'])}:</b> {_e(r['team'])} — {r['score']:.1f} "
                        f"(previous {prev_label}: {_e(r['previous'])})"
                    )
                st.html(callouts_html(items))

            st.subheader("Upsets")
            if recap.upsets:
                st.html(callouts_html([
                    f"⚡ <b>{_e(u['winner'])}</b> ({u['winner_score']:.1f}) beat <b>{_e(u['loser'])}</b> "
                    f"({u['loser_score']:.1f}) despite being projected lower "
                    f"({u['winner_proj']:.1f} vs. {u['loser_proj']:.1f}, a {u['gap']:.1f}-point gap)"
                    for u in recap.upsets
                ], "upset"))
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
                st.write("**Biggest overperformers**")
                show_table(diff_table_df(recap_df.head(10)), raw_cols=("Diff (Actual − Proj)",))
                st.write("**Biggest underperformers**")
                show_table(
                    diff_table_df(recap_df.tail(10).sort_values(by="Diff (Actual − Proj)", ascending=True)),
                    raw_cols=("Diff (Actual − Proj)",),
                )


if __name__ == "__main__":
    main()
