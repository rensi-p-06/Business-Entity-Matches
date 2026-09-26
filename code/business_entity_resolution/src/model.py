"""
The matching model itself: a scikit-learn HistGradientBoostingClassifier
trained from scratch on the engineered pairwise features (features.py).

This is a classical, non-pretrained model — it has no license or parameter
count of its own (it's fit on this challenge's training pairs only), so the
"MIT/Apache-2.0, <=8B params" constraint is trivially satisfied and there is
no external data or pretrained-weight dependency anywhere in the pipeline.
"""
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from features import FEATURE_COLUMNS


def train_model(X, y, random_state: int = 42) -> HistGradientBoostingClassifier:
    """Train the pair classifier. `y` is 1 for a true match, 0 otherwise."""
    n_pos, n_total = int(np.sum(y)), len(y)
    pos_weight = (n_total - n_pos) / max(n_pos, 1)  # upweight the rare positive class
    sample_weight = np.where(np.asarray(y) == 1, pos_weight, 1.0)

    clf = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=6,
        learning_rate=0.08,
        l2_regularization=1.0,
        random_state=random_state,
    )
    clf.fit(X[FEATURE_COLUMNS], y, sample_weight=sample_weight)
    return clf


def predict_proba(clf, X) -> np.ndarray:
    return clf.predict_proba(X[FEATURE_COLUMNS])[:, 1]


def save_model(clf, threshold: float, path: str) -> None:
    joblib.dump({"model": clf, "threshold": threshold, "features": FEATURE_COLUMNS}, path)


def load_model(path: str):
    bundle = joblib.load(path)
    return bundle["model"], bundle["threshold"], bundle["features"]
