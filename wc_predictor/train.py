"""
train.py
========
End-to-end training script.

Run this:
  python train.py              → Phase 1 (historical data only)
  python train.py --live       → Phase 2 (includes live stats)
  python train.py --retrain    → Force re-pull from APIs (clears cache)

After round 1 of WC 2026:
  1. Drop data/statsbomb_wc2022.csv  (or update with 2026 data)
  2. python train.py --live
  → Model refitted with real round 1 xG/possession data
"""

import argparse
import sys
from pathlib import Path

# ── ensure local imports work when run as script ──
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all_data, HISTORICAL_CACHE, STATSBOMB_CACHE
from features import build_dataset, FEATURE_NAMES, build_feature_vector
from model import train, evaluate, get_feature_importance, save_model


def main():
    parser = argparse.ArgumentParser(description="Train WC outcome predictor")
    parser.add_argument("--live",    action="store_true",
                        help="Include live tournament stats (Phase 2)")
    parser.add_argument("--retrain", action="store_true",
                        help="Delete caches and re-fetch from APIs")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip hyperparameter search (use C=0.5)")
    args = parser.parse_args()

    if args.retrain:
        for f in [HISTORICAL_CACHE, STATSBOMB_CACHE]:
            if f.exists():
                f.unlink()
                print(f"Deleted cache: {f}")

    print("=" * 55)
    phase = 2 if args.live else 1
    print(f"WORLD CUP PREDICTOR — TRAINING (Phase {phase})")
    print("=" * 55)

    # ── 1. Load data ────────────────────────────────────────────────────
    hist, live = load_all_data(use_live=args.live)

    if not args.live:
        print("\nPhase 1: no live stats — using historical features only.")
        print("Run with --live after round 1 to activate xG/possession features.")
    else:
        if live is not None:
            print(f"\nPhase 2: live stats loaded for {len(live)} teams.")
            games = live["games_played"].mean()
            print(f"  Average games per team: {games:.1f}")
            if games < 2:
                print("  ⚠  Only 1 game per team — live features are noisy. "
                      "Refit after round 2 for stronger signal.")
        else:
            print("  ⚠  No live stats available yet.")

    # ── 2. Build feature matrix ─────────────────────────────────────────
    print("\nBuilding feature matrix...")
    X, y = build_dataset(hist, live)

    # ── 3. Train ────────────────────────────────────────────────────────
    print("\nTraining model...")
    pipe = train(X, y, tune=not args.no_tune)

    # ── 4. Evaluate ─────────────────────────────────────────────────────
    metrics = evaluate(pipe, X, y)
    coef_df = get_feature_importance(pipe, FEATURE_NAMES)

    # ── 5. Quick prediction demo ────────────────────────────────────────
    print("\n── EXAMPLE PREDICTIONS ──")
    demos = [
        ("France",     "Brazil"),
        ("Argentina",  "England"),
        ("Germany",    "Spain"),
        ("Morocco",    "Japan"),
        ("United States", "Mexico"),
    ]
    from model import predict_match
    for home, away in demos:
        fv   = build_feature_vector(home, away, hist, live)
        pred = predict_match(pipe, fv)
        print(f"  {home:<18} vs {away:<18}  →  "
              f"W:{pred['home_win']:.2f}  D:{pred['draw']:.2f}  "
              f"L:{pred['away_win']:.2f}  ({pred['prediction']}, "
              f"conf={pred['confidence']:.2f})")

    # ── 6. Save ─────────────────────────────────────────────────────────
    save_model(pipe)

    print("\n✓ Training complete.")
    print(f"  CV accuracy: {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
    print(f"  Model saved to: models/wc_model.pkl")
    print(f"\nNext steps:")
    if phase == 1:
        print("  1. After round 1 of WC 2026, run:  python train.py --live")
        print("  2. Use predict.py to generate bracket predictions.")
    else:
        print("  1. Run predict.py to predict all upcoming fixtures.")
        print("  2. Refit after round 2 for stronger live signal.")


if __name__ == "__main__":
    main()
