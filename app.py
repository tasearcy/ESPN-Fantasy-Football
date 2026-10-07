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
    year = st.sidebar.selectbox("Season", year_options, index=default_index)

    espn_s2 = st.sidebar.text_input(
        "espn_s2", value=os.getenv("ESPN_S2", ""), type="password"
    )
    swid = st.sidebar.text_input(
        "SWID", value=os.getenv("ESPN_SWID", ""), type="password"
    )

    st.sidebar.divider()
    if st.sidebar.button("🔄 Refresh Data Now"):
        get_league.clear()
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


def get_proj_points(player) -> float:
    """Best-effort projected points, tolerating espn_api version differences."""
    proj = (
        getattr(player, "projected_total_points", None)
        or getattr(player, "projected_points", None)
        or getattr(player, "projected_avg_points", None)
    )
    return proj if isinstance(proj, (int, float)) else 0.0


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


def build_roster_df(team) -> pd.DataFrame:
    rows = []
    for player in team.roster:
        rows.append(
            {
                "Player": player.name,
                "Position": player.position,
                "Pro Team": player.proTeam,
                "Slot": player.lineupSlot,
                "Proj Points": _safe_round(get_proj_points(player)),
                "Avg Points": _safe_round(get_avg_points(player)),
                "Injury Status": getattr(player, "injuryStatus", "ACTIVE"),
            }
        )
    return pd.DataFrame(rows)


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


def build_power_rankings_df(league: League, week: int) -> pd.DataFrame:
    try:
        rankings = league.power_rankings(week=week)
    except Exception:
        return pd.DataFrame()
    rows = [
        {"Rank": i + 1, "Team": team.team_name, "Score": score}
        for i, (score, team) in enumerate(rankings)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Lineup optimizer
# ---------------------------------------------------------------------
def optimize_lineup(league: League, team):
    """
    Greedy lineup optimizer: fills the most restrictive slots first
    (fewest eligible positions), always taking the highest-projected
    remaining eligible player. Returns:
      current_starters_total, optimal_total, swaps_df
    """
    slot_counts = dict(getattr(league.settings, "position_slot_counts", {}))
    slot_counts.pop("BE", None)
    slot_counts.pop("IR", None)

    # Order slots by scarcity: fewer total roster-wide eligible players
    # for that slot name should be filled first. As a simple proxy,
    # single-position slots (QB, RB, WR, TE, D/ST, K) go before
    # flex-type slots (RB/WR/TE, OP, etc.), which go before anything
    # with an even broader pool.
    def slot_sort_key(slot_name):
        is_flex = "/" in slot_name or slot_name.upper() in ("FLEX", "OP", "UTIL")
        return (1 if is_flex else 0, slot_name)

    ordered_slots = sorted(slot_counts.keys(), key=slot_sort_key)

    available = list(team.roster)
    assigned = []  # (slot, player)

    for slot_name in ordered_slots:
        count = slot_counts[slot_name]
        for _ in range(count):
            eligible = [
                p
                for p in available
                if slot_name in getattr(p, "eligibleSlots", [])
            ]
            if not eligible:
                continue
            best = max(eligible, key=get_proj_points)
            assigned.append((slot_name, best))
            available.remove(best)

    optimal_total = sum(get_proj_points(p) for _, p in assigned)
    optimal_names = {p.name for _, p in assigned}

    current_starters = [
        p for p in team.roster if p.lineupSlot not in ("BE", "IR")
    ]
    current_total = sum(get_proj_points(p) for p in current_starters)
    current_names = {p.name for p in current_starters}

    bench_but_should_start = [
        p for _, p in assigned if p.name not in current_names
    ]
    starting_but_should_bench = [
        p for p in current_starters if p.name not in optimal_names
    ]

    swap_rows = []
    for i in range(max(len(bench_but_should_start), len(starting_but_should_bench))):
        bench_player = bench_but_should_start[i] if i < len(bench_but_should_start) else None
        start_player = starting_but_should_bench[i] if i < len(starting_but_should_bench) else None

        proj_in = get_proj_points(bench_player) if bench_player else 0.0
        proj_out = get_proj_points(start_player) if start_player else 0.0
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
                "Proj Pts (in)": _safe_round(proj_in) if bench_player else None,
                "Start → Bench": start_player.name if start_player else "",
                "Proj Pts (out)": _safe_round(proj_out) if start_player else None,
                "Confidence": confidence,
            }
        )

    swaps_df = pd.DataFrame(swap_rows)
    return current_total, optimal_total, swaps_df


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
            players = [p for p in players if p.position == pos_arg]

    rows = []
    for player in players:
        rows.append(
            {
                "Player": player.name,
                "Position": player.position,
                "Pro Team": player.proTeam,
                "Proj Points": _safe_round(get_proj_points(player)),
                "Avg Points": _safe_round(get_avg_points(player)),
                "% Owned": _safe_round(getattr(player, "percent_owned", None)),
                "Injury Status": getattr(player, "injuryStatus", "ACTIVE"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="Proj Points", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Injury report
# ---------------------------------------------------------------------
def build_injury_report_df(league: League) -> pd.DataFrame:
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
                        "Proj Points": _safe_round(get_proj_points(player)),
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


def build_player_compare_df(players_with_teams) -> pd.DataFrame:
    rows = []
    for player, team in players_with_teams:
        rows.append(
            {
                "Player": player.name,
                "Team (NFL)": player.proTeam,
                "Fantasy Owner": team.team_name,
                "Position": player.position,
                "Proj Points": _safe_round(get_proj_points(player)),
                "Avg Points": _safe_round(get_avg_points(player)),
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
                proj = get_proj_points(player)
                actual = getattr(player, "points", None)
                if actual is None:
                    continue
                rows.append(
                    {
                        "Team": team.team_name,
                        "Player": player.name,
                        "Position": player.position,
                        "Proj Points": _safe_round(proj),
                        "Actual Points": _safe_round(actual),
                        "Diff (Actual − Proj)": _safe_round(actual - proj)
                        if proj is not None
                        else None,
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
        st.dataframe(standings_df, use_container_width=True, hide_index=True)

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
        team_names = [t.team_name for t in league.teams]
        selected = st.selectbox("Select a team", team_names)
        team = next(t for t in league.teams if t.team_name == selected)

        roster_df = build_roster_df(team)
        st.dataframe(roster_df, use_container_width=True, hide_index=True)

        st.subheader("Lineup Optimizer")
        try:
            current_total, optimal_total, swaps_df = optimize_lineup(league, team)
            left_on_bench = round(optimal_total - current_total, 1)

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Starters (Proj)", f"{current_total:.1f}")
            col2.metric("Optimal Lineup (Proj)", f"{optimal_total:.1f}")
            col3.metric("Points Left on Bench", f"{left_on_bench:+.1f}")

            if swaps_df.empty or left_on_bench <= 0:
                st.success("Your current lineup already matches the optimal lineup.")
            else:
                st.write("Suggested swaps to reach the optimal lineup:")
                st.dataframe(swaps_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Couldn't compute an optimal lineup for this roster. ({e})")

    # --- Matchups ---
    with tab3:
        week = st.selectbox(
            "Week", week_options, index=default_week_index, key="matchup_week"
        )
        matchup_df = build_matchup_df(league, int(week))
        st.dataframe(matchup_df, use_container_width=True, hide_index=True)

    # --- Power Rankings ---
    with tab4:
        pr_week = st.selectbox(
            "Week for power rankings",
            week_options,
            index=default_week_index,
            key="pr_week",
        )
        pr_df = build_power_rankings_df(league, int(pr_week))
        if pr_df.empty:
            st.warning("Power rankings aren't available for this week yet.")
        else:
            st.dataframe(pr_df, use_container_width=True, hide_index=True)

    # --- Free Agents / Waiver Wire ---
    with tab5:
        fa_col1, fa_col2 = st.columns(2)
        with fa_col1:
            fa_week = st.selectbox(
                "Week", week_options, index=default_week_index, key="fa_week"
            )
        with fa_col2:
            position = st.selectbox(
                "Position",
                ["All", "QB", "RB", "WR", "TE", "D/ST", "K"],
                key="fa_position",
            )

        fa_df = build_free_agents_df(league, int(fa_week), position)
        if fa_df.empty:
            st.info("No free agent data available for this selection.")
        else:
            st.dataframe(fa_df, use_container_width=True, hide_index=True)

    # --- Injury Report ---
    with tab6:
        st.caption("League-wide view of every rostered player with a non-active status.")
        injury_df = build_injury_report_df(league)
        if injury_df.empty:
            st.success("No notable injuries reported across the league right now.")
        else:
            st.dataframe(injury_df, use_container_width=True, hide_index=True)

    # --- Trade Analyzer ---
    with tab7:
        st.caption(
            "Compare up to 4 rostered players side by side using ESPN's "
            "projections and season averages."
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
            compare_df = build_player_compare_df(selected_pairs)
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

            fig = px.bar(
                compare_df,
                x="Player",
                y="Proj Points",
                color="Position",
                title="Projected Points Comparison",
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Weekly Recap ---
    with tab8:
        st.caption(
            "For a completed week, shows which starters beat or missed "
            "their projection by the widest margin, league-wide."
        )
        recap_week = st.selectbox(
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
