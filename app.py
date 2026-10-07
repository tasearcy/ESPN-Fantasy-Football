"""
ESPN Fantasy Football Dashboard
--------------------------------
Pulls live league data (standings, rosters, matchups, power rankings)
from a private ESPN Fantasy Football league using the espn_api library,
and displays it in a Streamlit dashboard.

Setup: see README.md
Run:   streamlit run app.py
"""

import os
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from espn_api.football import League

load_dotenv()

st.set_page_config(page_title="ESPN Fantasy Dashboard", layout="wide")


# ---------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to ESPN...")
def get_league(league_id: int, year: int, espn_s2: str, swid: str) -> League:
    return League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)


def load_credentials():
    """Pull credentials from .env, falling back to sidebar inputs."""
    st.sidebar.header("League Connection")

    league_id = st.sidebar.text_input(
        "League ID", value=os.getenv("LEAGUE_ID", "")
    )
    year = st.sidebar.number_input(
        "Season", value=int(os.getenv("LEAGUE_YEAR", 2026)), step=1
    )
    espn_s2 = st.sidebar.text_input(
        "espn_s2", value=os.getenv("ESPN_S2", ""), type="password"
    )
    swid = st.sidebar.text_input(
        "SWID", value=os.getenv("ESPN_SWID", ""), type="password"
    )

    return league_id, year, espn_s2, swid


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


def _safe_round(value, digits=1):
    """Round a stat value, tolerating None or missing attributes."""
    return round(value, digits) if isinstance(value, (int, float)) else None


def build_roster_df(team) -> pd.DataFrame:
    rows = []
    for player in team.roster:
        # Attribute names vary across espn_api versions, so try a few.
        proj = (
            getattr(player, "projected_total_points", None)
            or getattr(player, "projected_points", None)
            or getattr(player, "projected_avg_points", None)
        )
        avg = getattr(player, "avg_points", None) or getattr(
            player, "total_points", None
        )

        rows.append(
            {
                "Player": player.name,
                "Position": player.position,
                "Pro Team": player.proTeam,
                "Slot": player.lineupSlot,
                "Proj Points": _safe_round(proj),
                "Avg Points": _safe_round(avg),
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

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Standings", "Rosters", "Matchups", "Power Rankings"]
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

    # --- Rosters ---
    with tab2:
        team_names = [t.team_name for t in league.teams]
        selected = st.selectbox("Select a team", team_names)
        team = next(t for t in league.teams if t.team_name == selected)
        roster_df = build_roster_df(team)
        st.dataframe(roster_df, use_container_width=True, hide_index=True)

    # --- Matchups ---
    with tab3:
        current_week = league.current_week
        week = st.number_input(
            "Week", min_value=1, max_value=18, value=current_week, step=1
        )
        matchup_df = build_matchup_df(league, int(week))
        st.dataframe(matchup_df, use_container_width=True, hide_index=True)

    # --- Power Rankings ---
    with tab4:
        pr_week = st.number_input(
            "Week for power rankings",
            min_value=1,
            max_value=18,
            value=league.current_week,
            step=1,
            key="pr_week",
        )
        pr_df = build_power_rankings_df(league, int(pr_week))
        if pr_df.empty:
            st.warning("Power rankings aren't available for this week yet.")
        else:
            st.dataframe(pr_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
