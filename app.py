"""
ESPN Fantasy Football Dashboard
--------------------------------
Pulls live league data (standings, rosters, matchups, power rankings,
free agents, injuries) from a private ESPN Fantasy Football league
using the espn_api library, and displays it in a Streamlit dashboard.

Setup: see README.md
Run:   streamlit run app.py
"""

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from espn_api.football import League

load_dotenv()

st.set_page_config(page_title="ESPN Fantasy Dashboard", layout="wide")

NON_ACTIVE_STATUSES = {
    "QUESTIONABLE",
    "DOUBTFUL",
    "OUT",
    "INJURY_RESERVE",
    "SUSPENSION",
    "PUP",
}


# ---------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------
# ttl forces Streamlit to drop the cached League and reconnect periodically,
# which is what actually pulls fresh rosters/waivers from ESPN. Without a
# ttl, cache_resource holds the same connection (and its stale data) forever.
LEAGUE_CACHE_TTL_SECONDS = 30 * 60  # 30 minutes


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
                "Injury Status": getattr(player, "injuryStatus", "ACTIVE"),
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


def build_matchup_df(league: League, week: int) -> pd.DataFrame:
    rows = []
    for box in league.box_scores(week=week):
        rows.append(
            {
                "Away Team": box.away_team.team_name if box.away_team else "BYE",
                "Away Score": round(box.away_score, 1) if box.away_team else None,
                "Home Team": box.home_team.team_name if box.home_team else "BYE",
                "Home Score": round(box.home_score, 1) if box.home_team else None,
            }
        )
    return pd.DataFrame(rows)


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
def optimize_lineup(league: League, team, week: int, lookup: dict):
    """
    Greedy lineup optimizer: fills the most restrictive slots first
    (single-position slots before flex-type slots), always taking the
    highest WEEKLY-projected eligible player left. Season-long numbers
    are never used, since they ignore this week's matchup, bye and
    injury situation.

    Players with no weekly projection are ranked as 0.0 so they never
    get promoted on a guess; their names are returned so the UI can
    flag them. Returns:
      current_total, optimal_total, swaps_df, missing_names
    """
    def wp(p):
        val = get_week_proj_points(p, week, lookup)
        return val if val is not None else 0.0

    slot_counts = dict(getattr(league.settings, "position_slot_counts", {}))
    slot_counts.pop("BE", None)
    slot_counts.pop("IR", None)

    def slot_sort_key(slot_name):
        is_flex = "/" in slot_name or slot_name.upper() in ("FLEX", "OP", "UTIL", "DP")
        return (1 if is_flex else 0, slot_name)

    ordered_slots = sorted(slot_counts.keys(), key=slot_sort_key)

    available = list(team.roster)
    assigned = []  # (slot, player)

    for slot_name in ordered_slots:
        count = slot_counts[slot_name]
        for _ in range(count):
            eligible = [
                p for p in available if slot_name in getattr(p, "eligibleSlots", [])
            ]
            if not eligible:
                continue
            best = max(eligible, key=wp)
            assigned.append((slot_name, best))
            available.remove(best)

    optimal_total = sum(wp(p) for _, p in assigned)
    optimal_names = {p.name for _, p in assigned}

    current_starters = [p for p in team.roster if p.lineupSlot not in ("BE", "IR")]
    current_total = sum(wp(p) for p in current_starters)
    current_names = {p.name for p in current_starters}

    # Only players who matter to the decision (starters or optimal picks)
    # need a weekly number; bench players left on the bench don't.
    relevant = {p.name: p for p in current_starters}
    relevant.update({p.name: p for _, p in assigned})
    missing_names = sorted(
        name
        for name, p in relevant.items()
        if get_week_proj_points(p, week, lookup) is None
    )

    bench_but_should_start = [p for _, p in assigned if p.name not in current_names]
    starting_but_should_bench = [
        p for p in current_starters if p.name not in optimal_names
    ]

    swap_rows = []
    for i in range(max(len(bench_but_should_start), len(starting_but_should_bench))):
        bench_player = (
            bench_but_should_start[i] if i < len(bench_but_should_start) else None
        )
        start_player = (
            starting_but_should_bench[i] if i < len(starting_but_should_bench) else None
        )

        proj_in = wp(bench_player) if bench_player else 0.0
        proj_out = wp(start_player) if start_player else 0.0
        gap = proj_in - proj_out

        if gap >= 4:
            confidence = "🟢 Strong"
        elif gap >= 1.5:
            confidence = "🟡 Moderate"
        else:
            confidence = "🟠 Close call"

        swap_rows.append(
            {
                "Bench → Start": bench_player.name if bench_player else "",
                f"Wk {week} Proj (in)": _safe_round(proj_in) if bench_player else None,
                "Start → Bench": start_player.name if start_player else "",
                f"Wk {week} Proj (out)": _safe_round(proj_out) if start_player else None,
                "Confidence": confidence,
            }
        )

    swaps_df = pd.DataFrame(swap_rows)
    return current_total, optimal_total, swaps_df, missing_names


# ---------------------------------------------------------------------
# Free agents
# ---------------------------------------------------------------------
def build_free_agents_df(league: League, week: int, position: str, size: int = 50) -> pd.DataFrame:
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
                "Injury Status": getattr(player, "injuryStatus", "ACTIVE"),
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
            status = getattr(player, "injuryStatus", None)
            if status and status.upper() in NON_ACTIVE_STATUSES:
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
        df = df.sort_values(by=["Status", "Team"]).reset_index(drop=True)
    return df


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
                "Injury Status": getattr(player, "injuryStatus", "ACTIVE"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Weekly recap (actual vs. projected)
# ---------------------------------------------------------------------
def build_weekly_recap_df(league: League, week: int) -> pd.DataFrame:
    """
    Compares each starter's actual points to their pre-game projection
    for a completed week, across the whole league, so you can see who
    beat or missed their number by the widest margin.
    """
    rows = []
    for box in league.box_scores(week=week):
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
# App
# ---------------------------------------------------------------------
def main():
    st.title("🏈 ESPN Fantasy Football Dashboard")

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

    st.success(f"Connected to **{league.settings.name}** — {year} season")

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
        st.dataframe(
            standings_df,
            use_container_width=True,
            hide_index=True,
            height=fit_height(standings_df),
        )

        fig = px.bar(
            standings_df,
            x="Team",
            y="Points For",
            color="Wins",
            title="Points For by Team",
        )
        st.plotly_chart(fig, use_container_width=True)

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
        st.dataframe(roster_df, use_container_width=True, hide_index=True)

        st.subheader(f"Lineup Optimizer (Week {roster_week})")
        st.caption(
            "Based on ESPN's projections for the selected week only. It "
            "re-slots your current roster, so it's most reliable for the "
            "current week."
        )
        try:
            current_total, optimal_total, swaps_df, missing = optimize_lineup(
                league, team, roster_week, lookup
            )
            left_on_bench = round(optimal_total - current_total, 1)

            col1, col2, col3 = st.columns(3)
            col1.metric(f"Current Starters (Wk {roster_week} Proj)", f"{current_total:.1f}")
            col2.metric(f"Optimal Lineup (Wk {roster_week} Proj)", f"{optimal_total:.1f}")
            col3.metric("Points Left on Bench", f"{left_on_bench:+.1f}")

            if missing:
                st.warning(
                    "No Week "
                    f"{roster_week} projection from ESPN for: {', '.join(missing)}. "
                    "They're counted as 0.0, so double-check those spots by hand."
                )

            if swaps_df.empty or left_on_bench <= 0:
                st.success("Your current lineup already matches the optimal lineup.")
            else:
                st.write("Suggested swaps to reach the optimal lineup:")
                st.dataframe(swaps_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Couldn't compute an optimal lineup for this roster. ({e})")

    # --- Matchups ---
    with tab3:
        week = dropdown(
            "Week", week_options, index=default_week_index, key="matchup_week"
        )
        matchup_df = build_matchup_df(league, int(week))
        st.dataframe(matchup_df, use_container_width=True, hide_index=True)

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
            st.dataframe(
                pr_df,
                use_container_width=True,
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
            st.dataframe(fa_df, use_container_width=True, hide_index=True)

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
            st.dataframe(injury_df, use_container_width=True, hide_index=True)

    # --- Trade Analyzer ---
    with tab7:
        trade_week = week_options[default_week_index]
        st.caption(
            "Compare up to 4 rostered players side by side using ESPN's "
            f"Week {trade_week} projection, season average and season-long "
            "projection."
        )
        trade_lookup = get_week_projection_lookup(
            league, int(league_id), int(year), trade_week
        )
        all_pairs = get_all_rostered_players(league)
        player_names = sorted({p.name for p, _ in all_pairs})

        selected_names = st.multiselect(
            "Select players to compare (2-4)",
            player_names,
            max_selections=4,
        )

        if len(selected_names) < 2:
            st.info("Pick at least 2 players to compare.")
        else:
            selected_pairs = [
                (p, t) for p, t in all_pairs if p.name in selected_names
            ]
            compare_df = build_player_compare_df(
                selected_pairs, trade_week, trade_lookup
            )
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

            fig = px.bar(
                compare_df,
                x="Player",
                y=f"Week {trade_week} Proj",
                color="Position",
                title=f"Week {trade_week} Projected Points",
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Weekly Recap ---
    with tab8:
        st.caption(
            "For a completed week, shows which starters beat or missed "
            "their projection by the widest margin, league-wide."
        )
        recap_week = dropdown(
            "Week",
            week_options,
            index=max(default_week_index - 1, 0),  # default to last completed week
            key="recap_week",
        )
        recap_df = build_weekly_recap_df(league, int(recap_week))

        if recap_df.empty:
            st.info("No completed results available for this week yet.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Biggest overperformers**")
                st.dataframe(
                    recap_df.head(10), use_container_width=True, hide_index=True
                )
            with col2:
                st.write("**Biggest underperformers**")
                st.dataframe(
                    recap_df.tail(10).sort_values(
                        by="Diff (Actual − Proj)", ascending=True
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


if __name__ == "__main__":
    main()
