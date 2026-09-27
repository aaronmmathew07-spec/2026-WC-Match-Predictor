"""
predict.py
==========
Load a trained model and predict any fixture.

Usage
-----
  python predict.py                         → predict hardcoded fixtures
  python predict.py "France" "Brazil"       → single match
  python predict.py --round2                → predict WC 2026 round 2 fixtures

After round 1, update ROUND_2_FIXTURES below with the actual matchups
(known once group stage round 1 is complete).
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all_data
from features import build_feature_vector
from model import load_model, predict_match, predict_tournament

# ── Update these after round 1 of WC 2026 is played ──────────────────────
# Replace placeholder teams with actual round 2 fixtures once known.
ROUND_2_FIXTURES = [
    {"home_team": "France",        "away_team": "Argentina",  "stage": 0},
    {"home_team": "Brazil",        "away_team": "Germany",    "stage": 0},
    {"home_team": "Spain",         "away_team": "England",    "stage": 0},
    {"home_team": "Netherlands",   "away_team": "Portugal",   "stage": 0},
    {"home_team": "Morocco",       "away_team": "Japan",      "stage": 0},
    {"home_team": "United States", "away_team": "Mexico",     "stage": 0},
]


def print_prediction(home: str, away: str, pred: dict) -> None:
    bar_w = int(pred["home_win"]  * 40)
    bar_d = int(pred["draw"]      * 40)
    bar_l = int(pred["away_win"]  * 40)

    print(f"\n  {home} vs {away}")
    print(f"  {'─'*48}")
    print(f"  Win  {pred['home_win']:.1%}  {'█'*bar_w}")
    print(f"  Draw {pred['draw']:.1%}  {'█'*bar_d}")
    print(f"  Loss {pred['away_win']:.1%}  {'█'*bar_l}")
    print(f"  → Prediction: {pred['prediction'].upper()}  "
          f"(confidence {pred['confidence']:.1%})")


def main():
    parser = argparse.ArgumentParser(description="WC match predictor")
    parser.add_argument("teams", nargs="*", help="'HomeTeam' 'AwayTeam'")
    parser.add_argument("--round2", action="store_true",
                        help="Predict all round 2 fixtures")
    parser.add_argument("--no-live", action="store_true",
                        help="Don't use live stats (force Phase 1)")
    args = parser.parse_args()

    use_live = not args.no_live
    hist, live = load_all_data(use_live=use_live)

    try:
        pipe = load_model()
    except FileNotFoundError:
        print("No trained model found. Run:  python train.py")
        sys.exit(1)

    if args.teams and len(args.teams) == 2:
        home, away = args.teams
        fv   = build_feature_vector(home, away, hist, live)
        pred = predict_match(pipe, fv)
        print_prediction(home, away, pred)

    elif args.round2:
        print("\nROUND 2 PREDICTIONS")
        print("=" * 55)
        results_df = predict_tournament(pipe, ROUND_2_FIXTURES, hist, live)
        for _, row in results_df.iterrows():
            print_prediction(row["home_team"], row["away_team"], row.to_dict())
        results_df.to_csv("predictions_round2.csv", index=False)
        print(f"\nSaved to predictions_round2.csv")

    else:
        # Default: a set of interesting WC 2026 group fixtures
        print("\nSAMPLE PREDICTIONS — WC 2026")
        print("=" * 55)
        sample = [
            ("France",     "Brazil"),
            ("Argentina",  "England"),
            ("Germany",    "Spain"),
            ("Morocco",    "Japan"),
            ("United States", "Canada"),
        ]
        for home, away in sample:
            fv   = build_feature_vector(home, away, hist, live)
            pred = predict_match(pipe, fv)
            print_prediction(home, away, pred)


if __name__ == "__main__":
    main()
