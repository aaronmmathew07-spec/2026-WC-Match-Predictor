"""
model.py
========
Multinomial logistic regression pipeline for World Cup outcome prediction.

Architecture:
  StandardScaler → LogisticRegression(multinomial, L2, softmax)

Output: P(home_win), P(draw), P(away_win) — calibrated probabilities.

Designed to be swapped for XGBoost or a neural net by changing only
build_pipeline() — all other code stays identical.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, log_loss)
from sklearn.calibration import CalibratedClassifierCV

OUTCOMES    = ["home_win", "draw", "away_win"]
MODEL_PATH  = Path(__file__).parent / "models" / "wc_model.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# Pipeline factory
# ══════════════════════════════════════════════════════════════════════════

def build_pipeline(C: float = 0.5) -> Pipeline:
    """
    Build the sklearn Pipeline.

    StandardScaler
    --------------
    Zero-mean, unit-variance normalisation fitted on training data only.
    Critical because features span very different scales:
      - FIFA rank diff:    range ≈ [-150, 150]
      - xG differential:  range ≈ [-3, 3]
      - H2H win rate:      range ≈ [0, 1]
    Without scaling, gradient descent is ill-conditioned and
    L2 regularisation penalises large-scale features unfairly.

    LogisticRegression
    ------------------
    multi_class='multinomial': uses softmax (not one-vs-rest).
      softmax(z_k) = exp(z_k) / Σ exp(z_j)  for k ∈ {win, draw, loss}
    solver='lbfgs': quasi-Newton method, good for small-medium datasets.
    C: inverse regularisation strength. Smaller = stronger L2 penalty.
      → controls bias-variance tradeoff. Tune via cross-validation.
    max_iter=1000: WC dataset is small, converges fast.

    Parameters
    ----------
    C : float
        Regularisation inverse. Try [0.01, 0.05, 0.1, 0.5, 1.0, 5.0].
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            C=C,
            max_iter=1000,
            random_state=42,
            class_weight="balanced",   # handles draw underrepresentation
        ))
    ])


# ══════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════

def tune_C(X: np.ndarray, y: np.ndarray,
           candidates: list = None) -> float:
    """
    Grid search over regularisation strength C using stratified 5-fold CV.
    Returns the C with highest mean accuracy.

    The small WC dataset (~640 matches) makes this fast — runs in <2s.
    """
    if candidates is None:
        candidates = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print("Tuning regularisation strength C:")
    for c in candidates:
        pipe = build_pipeline(C=c)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        results.append((c, scores.mean(), scores.std()))
        print(f"  C={c:<6}  acc={scores.mean():.4f} ± {scores.std():.4f}")

    best_C = max(results, key=lambda r: r[1])[0]
    print(f"\n  Best C: {best_C}")
    return best_C


def train(X: np.ndarray, y: np.ndarray,
          C: float = None,
          tune: bool = True) -> Pipeline:
    """
    Fit the full pipeline on training data.

    Parameters
    ----------
    X : np.ndarray, shape (n, 14)
    y : np.ndarray, shape (n,) — 0=home win, 1=draw, 2=away win
    C : float or None
        If None and tune=True, performs grid search.
    tune : bool
        Whether to run hyperparameter search.

    Returns
    -------
    Fitted sklearn Pipeline.
    """
    if C is None and tune:
        C = tune_C(X, y)
    elif C is None:
        C = 0.5

    pipe = build_pipeline(C=C)
    pipe.fit(X, y)
    print(f"\nModel fitted on {len(X)} matches.")
    return pipe


# ══════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════

def evaluate(pipe: Pipeline, X: np.ndarray, y: np.ndarray,
             cv_folds: int = 5) -> dict:
    """
    Full model evaluation report.

    Metrics
    -------
    accuracy:       % of outcomes correctly predicted (top-1)
    log_loss:       cross-entropy — penalises confident wrong predictions.
                    Lower is better. Random baseline ≈ 1.099 (ln 3).
    cv_accuracy:    stratified k-fold cross-validation accuracy
    confusion_matrix: rows=actual, cols=predicted

    The draw class is the hardest to predict (even professional models
    struggle here). Accuracy ~58–68% is typical for WC logistic models.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    y_pred    = pipe.predict(X)
    y_proba   = pipe.predict_proba(X)
    ll        = log_loss(y, y_proba)

    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred,
                                   target_names=OUTCOMES,
                                   output_dict=True)

    print("\n" + "="*55)
    print("MODEL EVALUATION REPORT")
    print("="*55)
    print(f"Training accuracy       : {accuracy_score(y, y_pred):.4f}")
    print(f"CV accuracy ({cv_folds}-fold)    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Log-loss                : {ll:.4f}  (random baseline: 1.099)")
    print(f"\nOutcome breakdown:")
    for outcome in OUTCOMES:
        r = report[outcome]
        print(f"  {outcome:<12}  precision={r['precision']:.3f}  "
              f"recall={r['recall']:.3f}  f1={r['f1-score']:.3f}")
    print(f"\nConfusion matrix (rows=actual, cols=predicted):")
    print(f"             home_win  draw  away_win")
    labels = ["home_win", "draw    ", "away_win"]
    for i, row in enumerate(cm):
        print(f"  {labels[i]}  {row}")
    print("="*55)

    return {
        "train_accuracy": accuracy_score(y, y_pred),
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "log_loss": ll,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def get_feature_importance(pipe: Pipeline,
                           feature_names: list) -> pd.DataFrame:
    """
    Extract logistic regression coefficients as a feature importance table.

    Coefficients are in log-odds space after scaling, so they are directly
    comparable. Positive = increases probability of that outcome class.

    This is one of the key advantages of logistic regression over black-box
    models — every weight has a clear economic interpretation.
    """
    clf = pipe.named_steps["clf"]
    coef_df = pd.DataFrame(
        clf.coef_,
        index=OUTCOMES,
        columns=feature_names
    )
    print("\nFeature coefficients (log-odds, scaled):")
    print(coef_df.round(4).to_string())
    return coef_df


# ══════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════

def predict_match(pipe: Pipeline,
                  feature_vector: np.ndarray) -> dict:
    """
    Predict outcome probabilities for one match.

    Parameters
    ----------
    pipe : fitted Pipeline
    feature_vector : np.ndarray of shape (14,)

    Returns
    -------
    dict with keys: home_win, draw, away_win (probabilities summing to 1)
                    prediction (most likely outcome)
                    confidence (probability of the predicted outcome)
    """
    X = feature_vector.reshape(1, -1)
    probs  = pipe.predict_proba(X)[0]
    pred_i = np.argmax(probs)

    result = dict(zip(OUTCOMES, probs.round(4)))
    result["prediction"] = OUTCOMES[pred_i]
    result["confidence"] = round(float(probs[pred_i]), 4)
    return result


def predict_tournament(pipe: Pipeline, fixtures: list[dict],
                       historical_df, live_stats_df=None) -> pd.DataFrame:
    """
    Predict all matches in a fixture list.

    Parameters
    ----------
    fixtures : list of dicts with keys 'home_team', 'away_team', 'stage'
    historical_df, live_stats_df : from data_loader

    Returns
    -------
    DataFrame with one row per fixture + probability columns.
    """
    from features import build_feature_vector

    rows = []
    for fx in fixtures:
        fv   = build_feature_vector(fx["home_team"], fx["away_team"],
                                    historical_df, live_stats_df)
        pred = predict_match(pipe, fv)
        rows.append({
            "home_team": fx["home_team"],
            "away_team": fx["away_team"],
            "stage":     fx.get("stage", 0),
            **pred
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════════

def save_model(pipe: Pipeline, path: Path = MODEL_PATH) -> None:
    """Serialise fitted pipeline to disk."""
    joblib.dump(pipe, path)
    print(f"Model saved → {path}")


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Load a previously trained pipeline."""
    pipe = joblib.load(path)
    print(f"Model loaded ← {path}")
    return pipe


if __name__ == "__main__":
    from data_loader import load_all_data
    from features import build_dataset, FEATURE_NAMES

    hist, live = load_all_data()
    X, y = build_dataset(hist, live)
    pipe = train(X, y, tune=True)
    metrics = evaluate(pipe, X, y)
    get_feature_importance(pipe, FEATURE_NAMES)
    save_model(pipe)
