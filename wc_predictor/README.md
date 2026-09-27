# World Cup Outcome Predictor

Multinomial logistic regression model for predicting FIFA World Cup match outcomes.
Built in two phases: historical priors first, live tournament stats second.

## Setup

```bash
pip install -r requirements.txt
export API_FOOTBALL_KEY="your_key_here"   # from rapidapi.com (free tier)
```

No key needed for StatsBomb data — it's fetched directly from GitHub.
If you have no API key yet, the model trains on synthetic data automatically.

## Workflow

### Phase 1 — before or during round 1 (historical features only)
```bash
python train.py
python predict.py
```

### Phase 2 — after round 1 completes (~June 19, 2026)
```bash
python ingest_round1.py        # fetch live stats, refit model
python predict.py --round2     # predict round 2 fixtures
```

### Predict a single match
```bash
python predict.py "France" "Brazil"
```

### Force re-pull from APIs (clears cache)
```bash
python train.py --retrain --live
```

## Files

| File | Purpose |
|------|---------|
| `data_loader.py` | API fetch + caching + synthetic fallback |
| `features.py` | 14-feature vector engineering |
| `model.py` | Pipeline: scaler → logistic regression |
| `train.py` | End-to-end training script |
| `predict.py` | Inference on fixtures |
| `ingest_round1.py` | Live data ingestion after round 1 |

## Feature vector (14 inputs)

| # | Feature | Phase | Source |
|---|---------|-------|--------|
| 0 | FIFA rank differential | 1 | Static lookup |
| 1 | H2H WC win rate | 1 | API-Football |
| 2 | Home avg WC stage (last 3) | 1 | API-Football |
| 3 | Away avg WC stage (last 3) | 1 | API-Football |
| 4 | Home goals scored per game | 1 | API-Football |
| 5 | Home goals conceded per game | 1 | API-Football |
| 6 | Away goals scored per game | 1 | API-Football |
| 7 | Away goals conceded per game | 1 | API-Football |
| 8 | Confederation strength diff | 1 | Static lookup |
| 9 | xG differential | 2 | StatsBomb / API-Football |
| 10 | Possession % (home) | 2 | StatsBomb / API-Football |
| 11 | Progressive carries p90 (home) | 2 | StatsBomb |
| 12 | Defensive actions p90 (home) | 2 | StatsBomb |
| 13 | Momentum index differential | 2 | Computed |

## APIs used

- **API-Football** (RapidAPI free tier): historical WC results 1986–2022
  https://rapidapi.com/api-sports/api/api-football
  
- **StatsBomb Open Data** (free, no key): xG + event data
  https://github.com/statsbomb/open-data

## Scaling path

Once data is sufficient, swap the classifier in `model.py`:
```python
from xgboost import XGBClassifier
# Replace LogisticRegression(...) with XGBClassifier(...)
# Feature engineering and all other code stays identical.
```

## Expected accuracy

| Phase | Typical CV accuracy | Notes |
|-------|--------------------|----|
| Phase 1 | 58–63% | Historical features only |
| Phase 2 (after R1) | 63–68% | + xG/possession |
| Phase 2 (after R3) | 65–70% | Strongest signal |

Baseline (always predict home win): ~45%.
Random: ~33%.
