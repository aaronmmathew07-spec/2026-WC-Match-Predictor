"""
features.py
===========
Transforms raw match + team data into a fixed 14-element feature vector
for each match. Designed to work in two phases:

  Phase 1 — pre-tournament or early in group stage:
    Uses only historical features (f1–f9). Live features set to 0.

  Phase 2 — after round 1 of group stage:
    Adds live tournament stats (f10–f14), significantly improving accuracy.

Feature vector layout (always same shape — model never needs recompiling):
  [0]  FIFA rank differential (home - away)
  [1]  Head-to-head World Cup win rate (home team, last 5 meetings)
  [2]  Avg WC stage reached by home team (last 3 tournaments)
  [3]  Avg WC stage reached by away team (last 3 tournaments)
  [4]  Goals scored per game — home (historical WC)
  [5]  Goals conceded per game — home (historical WC)
  [6]  Goals scored per game — away (historical WC)
  [7]  Goals conceded per game — away (historical WC)
  [8]  Confederation strength (home - away)
  [9]  xG differential (home - away) [Phase 2 only]
  [10] Possession advantage (home %) [Phase 2 only]
  [11] Progressive carries p90 — home [Phase 2 only]
  [12] Defensive actions p90 — home [Phase 2 only]
  [13] Momentum index differential (home - away) [Phase 2 only]
"""

import numpy as np
import pandas as pd
from data_loader import FIFA_RANK_BANDS, CONFEDERATION, CONF_STRENGTH


# ══════════════════════════════════════════════════════════════════════════
# Historical feature helpers (Phase 1)
# ══════════════════════════════════════════════════════════════════════════

def fifa_rank_diff(home: str, away: str) -> float:
    """
    FIFA rank differential: positive = home team is ranked higher (better).
    Lower rank number = better team, so we negate.
    Uses static lookup; replace with live FIFA API for production.
    """
    home_rank = FIFA_RANK_BANDS.get(home, 50)
    away_rank = FIFA_RANK_BANDS.get(away, 50)
    return float(away_rank - home_rank)   # positive = home better


def h2h_win_rate(home: str, away: str, historical_df: pd.DataFrame,
                 n_meetings: int = 5) -> float:
    """
    Historical head-to-head World Cup win rate for the home team.
    Returns 0.5 if the teams have never met (neutral prior).
    """
    h2h = historical_df[
        ((historical_df["home_team"] == home) & (historical_df["away_team"] == away)) |
        ((historical_df["home_team"] == away) & (historical_df["away_team"] == home))
    ].tail(n_meetings)

    if len(h2h) == 0:
        return 0.5

    wins = 0
    for _, row in h2h.iterrows():
        if row["home_team"] == home and row["outcome"] == 0:
            wins += 1
        elif row["away_team"] == home and row["outcome"] == 2:
            wins += 1

    return wins / len(h2h)


def avg_stage_reached(team: str, historical_df: pd.DataFrame,
                      n_tournaments: int = 3) -> float:
    """
    Average furthest stage reached in the last n World Cups.
    Stage encoding: group=0, R16=1, QF=2, SF=3, Final=4.
    Returns 0.5 if team has limited history.

    Economic intuition: this is a noisy proxy for squad quality
    and tournament experience — both proven predictors in the
    football analytics literature.
    """
    seasons = sorted(historical_df["season"].unique())[-n_tournaments:]
    stages = []

    for season in seasons:
        season_data = historical_df[historical_df["season"] == season]
        team_matches = season_data[
            (season_data["home_team"] == team) |
            (season_data["away_team"] == team)
        ]
        if len(team_matches) > 0:
            stages.append(team_matches["stage"].max())

    return np.mean(stages) if stages else 0.5


def goals_per_game(team: str, historical_df: pd.DataFrame,
                   scored: bool = True) -> float:
    """
    Average goals scored (or conceded) per WC match.
    Scored=True → attacking strength. Scored=False → defensive strength.
    """
    home_mask = historical_df["home_team"] == team
    away_mask = historical_df["away_team"] == team

    if scored:
        home_goals = historical_df.loc[home_mask, "home_goals"]
        away_goals = historical_df.loc[away_mask, "away_goals"]
    else:
        home_goals = historical_df.loc[home_mask, "away_goals"]   # conceded
        away_goals = historical_df.loc[away_mask, "home_goals"]   # conceded

    all_goals = pd.concat([home_goals, away_goals])
    return all_goals.mean() if len(all_goals) > 0 else 1.2   # WC average prior


def confederation_diff(home: str, away: str) -> float:
    """
    Difference in confederation strength weights.
    UEFA and CONMEBOL teams consistently outperform other confederations
    at World Cups — this captures that systematic effect.
    """
    home_conf = CONFEDERATION.get(home, "UEFA")
    away_conf = CONFEDERATION.get(away, "UEFA")
    return CONF_STRENGTH.get(home_conf, 0.75) - CONF_STRENGTH.get(away_conf, 0.75)


# ══════════════════════════════════════════════════════════════════════════
# Live stat helpers (Phase 2 — activate after round 1)
# ══════════════════════════════════════════════════════════════════════════

def _get_live(live_df: pd.DataFrame, team: str, col: str,
              default: float = 0.0) -> float:
    """Safe lookup of a live stat for a team. Returns default if not found."""
    if live_df is None:
        return default
    row = live_df[live_df["team"] == team]
    if len(row) == 0:
        return default
    return float(row[col].iloc[0])


# ══════════════════════════════════════════════════════════════════════════
# Main feature builder
# ══════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "fifa_rank_diff",          # f0
    "h2h_win_rate",            # f1
    "home_avg_stage",          # f2
    "away_avg_stage",          # f3
    "home_goals_scored_pg",    # f4
    "home_goals_conceded_pg",  # f5
    "away_goals_scored_pg",    # f6
    "away_goals_conceded_pg",  # f7
    "confederation_diff",      # f8
    "xg_diff",                 # f9  (Phase 2)
    "possession_home",         # f10 (Phase 2)
    "prog_carries_home_p90",   # f11 (Phase 2)
    "defensive_actions_home",  # f12 (Phase 2)
    "momentum_diff",           # f13 (Phase 2)
]


def build_feature_vector(home: str, away: str,
                         historical_df: pd.DataFrame,
                         live_stats_df=None) -> np.ndarray:
    """
    Build the 14-element feature vector for one match.

    Parameters
    ----------
    home : str
        Name of home team (must match naming convention in historical data)
    away : str
        Name of away team
    historical_df : DataFrame
        All historical WC matches (from data_loader.fetch_historical_from_api)
    live_stats_df : DataFrame or None
        Aggregated live stats (from data_loader.aggregate_live_stats).
        None = Phase 1 (pre-tournament). Provide = Phase 2 (in-tournament).

    Returns
    -------
    np.ndarray of shape (14,)
    """
    # ── Phase 1: Historical features ──────────────────────────────────────
    f0  = fifa_rank_diff(home, away)
    f1  = h2h_win_rate(home, away, historical_df)
    f2  = avg_stage_reached(home, historical_df)
    f3  = avg_stage_reached(away, historical_df)
    f4  = goals_per_game(home, historical_df, scored=True)
    f5  = goals_per_game(home, historical_df, scored=False)
    f6  = goals_per_game(away, historical_df, scored=True)
    f7  = goals_per_game(away, historical_df, scored=False)
    f8  = confederation_diff(home, away)

    # ── Phase 2: Live tournament features ─────────────────────────────────
    # All default to 0 in Phase 1 so the feature vector shape never changes.
    # The model learns to ignore zeroed features during Phase 1 training.
    # After round 1, these carry real information and the model is refitted.
    home_xg   = _get_live(live_stats_df, home, "xg_per90")
    away_xg   = _get_live(live_stats_df, away, "xg_per90")
    f9  = home_xg - away_xg                                          # xG differential

    f10 = _get_live(live_stats_df, home, "possession_avg")           # home possession
    f11 = _get_live(live_stats_df, home, "prog_carries_p90")         # home carries
    f12 = _get_live(live_stats_df, home, "defensive_actions_p90")    # home defence
    f13 = (_get_live(live_stats_df, home, "momentum_index") -
           _get_live(live_stats_df, away, "momentum_index"))         # momentum diff

    return np.array([f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13],
                    dtype=np.float64)


def build_dataset(historical_df: pd.DataFrame,
                  live_stats_df=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct full (X, y) matrices from all historical matches.

    Outcome encoding:
      0 = home win
      1 = draw
      2 = away win

    Parameters
    ----------
    historical_df : DataFrame
    live_stats_df : DataFrame or None

    Returns
    -------
    X : np.ndarray, shape (n_matches, 14)
    y : np.ndarray, shape (n_matches,)
    """
    X_rows, y_rows = [], []

    for _, row in historical_df.iterrows():
        try:
            fv = build_feature_vector(
                row["home_team"], row["away_team"],
                historical_df, live_stats_df
            )
            X_rows.append(fv)
            y_rows.append(int(row["outcome"]))
        except Exception as e:
            print(f"  ⚠  Skipped match {row.get('home_team')} vs "
                  f"{row.get('away_team')}: {e}")

    X = np.vstack(X_rows)
    y = np.array(y_rows)
    print(f"Dataset built: {X.shape[0]} matches × {X.shape[1]} features")
    print(f"Outcome distribution: "
          f"{(y==0).sum()} home wins, {(y==1).sum()} draws, {(y==2).sum()} away wins")
    return X, y


if __name__ == "__main__":
    from data_loader import load_all_data
    hist, live = load_all_data()
    X, y = build_dataset(hist, live)
    print(f"\nFeature names: {FEATURE_NAMES}")
    print(f"Sample vector: {X[0].round(3)}")
