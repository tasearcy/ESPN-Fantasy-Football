# ESPN Fantasy Football Dashboard — Setup Guide

## Step 1: Install dependencies

```bash
pip install espn_api streamlit pandas plotly python-dotenv
```

## Step 2: Get your league ID

1. Log into your ESPN Fantasy league in a browser.
2. Look at the URL — it'll look like:
   `https://fantasy.espn.com/football/league?leagueId=123456`
3. Your `leagueId` is the number at the end. Save it.

## Step 3: Get your `espn_s2` and `swid` cookies

These two cookies authenticate you to your *private* league. Here's how to grab them:

**Chrome / Edge:**
1. Go to `fantasy.espn.com` and make sure you're logged in.
2. Open DevTools (`F12` or right-click → Inspect).
3. Go to the **Application** tab → **Storage** → **Cookies** → `https://fantasy.espn.com`.
4. Find the rows named `espn_s2` and `SWID`.
5. Copy the full **Value** for each (the `espn_s2` one is long — grab the whole thing, it often starts with `AE`).

**Firefox:**
1. Same idea — DevTools (`F12`) → **Storage** tab → **Cookies** → `https://fantasy.espn.com`.

> Note: `SWID` usually includes curly braces, e.g. `{A1B2C3D4-...}` — keep those when you copy it.

## Step 4: Store credentials safely (don't hardcode them)

Create a file named `.env` in the same folder as `app.py`:

```
ESPN_S2=paste_your_espn_s2_value_here
ESPN_SWID={paste-your-swid-value-here}
LEAGUE_ID=123456
LEAGUE_YEAR=2026
```

This keeps your cookies out of the script itself — never commit `.env` to GitHub or share it.

## Step 5: Run the dashboard

```bash
streamlit run app.py
```

It'll open in your browser at `localhost:8501`.

## Notes

- `espn_s2` cookies are long-lived but can expire (usually after a season or if you log out everywhere) — if the dashboard starts throwing auth errors, just re-grab the cookies.
- This only talks to ESPN directly from your machine — nothing about your league or credentials goes through me or any third party.
- League settings → make sure "league visibility" doesn't matter here since we're authenticating directly; this works for private leagues.
