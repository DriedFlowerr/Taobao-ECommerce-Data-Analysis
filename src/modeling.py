"""Temporal purchase-propensity benchmarks and capacity-constrained targeting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits


FEATURES = [
    "pv_4d", "cart_4d", "fav_4d", "buy_4d", "active_days_4d",
    "pv_1d", "cart_1d", "fav_1d", "buy_1d", "recency_days", "activity_4d",
]
BUDGETS = (0.01, 0.05, 0.10, 0.20)
SEED = 42


def _metrics(y, score):
    return {
        "ap": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "brier": brier_score_loss(y, score),
        "base_rate": float(y.mean()),
        "n": len(y),
    }


def _policy_metrics(y, order, name, selected=False):
    """A fixed capacity shared by every policy; random rows are expectations."""
    rows = []
    n, positives = len(y), float(y.sum())
    for fraction in BUDGETS:
        k = max(1, int(np.ceil(n * fraction)))
        hits = k * positives / n if order is None else int(y[order[:k]].sum())
        rows.append({
            "policy": name,
            "selected_model": selected,
            "budget_fraction": fraction,
            "n_selected": k,
            "test_size": n,
            "total_purchasers": int(positives),
            "hits": hits,
            "precision": hits / k,
            "recall": hits / positives,
            "lift": (hits / k) / (positives / n),
        })
    return rows


def _plot_diagnostics(y, scores, policies, selected_model, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    for name, score in scores.items():
        precision, recall, _ = precision_recall_curve(y, score)
        # Curves need not retain every threshold in the rendered image.
        step = max(1, len(precision) // 6000)
        axes[0].plot(recall[::step], precision[::step], label=name)
        observed, predicted = calibration_curve(y, score, n_bins=10, strategy="quantile")
        axes[1].plot(predicted, observed, marker="o", label=name)
    axes[0].axhline(y.mean(), linestyle="--", color="gray", label="Base rate")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Test precision-recall")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes[1].set(xlabel="Mean predicted probability", ylabel="Observed purchase rate",
                title="Test calibration (quantile bins)", xlim=(0, 1), ylim=(0, 1))
    for name in ["Random (expected)", "Recent cart rule", selected_model]:
        group = policies[policies["policy"] == name]
        axes[2].plot(group["budget_fraction"] * 100, group["lift"], marker="o", label=name)
    axes[2].set(xlabel="Targeting budget (%)", ylabel="Lift over random",
                title="Test targeting efficiency", xticks=[1, 5, 10, 20])
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(out_dir / "model_diagnostics.png", dpi=160)
    plt.close(fig)


def train_and_evaluate(snapshots: pd.DataFrame, out_dir: Path) -> dict:
    """Fit on train, choose by validation AP and report on a later test period.

    AP is average precision, a discrete precision-recall summary (not
    trapezoidal PR-AUC). Targeting outcomes are naturally observed purchases;
    these do not identify the incremental effect of a campaign.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    required = ["user_id", "score_date", "fold", "y", *FEATURES]
    missing = set(required) - set(snapshots.columns)
    if missing:
        raise ValueError(f"Missing snapshot columns: {sorted(missing)}")
    if snapshots.duplicated(["user_id", "score_date"]).any():
        raise ValueError("Each user/date must have at most one snapshot.")
    folds = {}
    dates = pd.to_datetime(snapshots["score_date"], errors="raise")
    if dates.isna().any() or set(snapshots["fold"].unique()) != {"train", "validation", "test"}:
        raise ValueError("Expected valid dates and exactly train/validation/test folds.")
    previous_end = None
    for fold in ("train", "validation", "test"):
        mask = snapshots["fold"].eq(fold)
        start, end = dates[mask].min(), dates[mask].max()
        if previous_end is not None and start <= previous_end:
            raise ValueError("Temporal folds overlap or are out of order.")
        previous_end = end
        subset = snapshots.loc[mask, required].sort_values(["score_date", "user_id"])
        x = subset[FEATURES].to_numpy(dtype=np.float32)
        if not np.isfinite(x).all() or (x < 0).any():
            raise ValueError(f"Invalid feature values in {fold}: expected finite nonnegative values.")
        if set(subset["y"].unique()) != {0, 1}:
            raise ValueError(f"Both binary target classes must occur in {fold}.")
        y = subset["y"].to_numpy(dtype=np.int8)
        # Only test needs a row mapping for the rule ranking.
        folds[fold] = (x, y, subset if fold == "test" else None)

    x_train, y_train, _ = folds["train"]
    x_val, y_val, _ = folds["validation"]
    x_test, y_test, test = folds["test"]
    models = {
        "Logistic regression": make_pipeline(
            FunctionTransformer(np.log1p), StandardScaler(),
            LogisticRegression(max_iter=300, random_state=SEED),
        ),
        "Histogram gradient boosting": HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=70, max_leaf_nodes=31,
            min_samples_leaf=50, l2_regularization=1.0,
            early_stopping=False, random_state=SEED,
        ),
    }
    metrics_rows, validation_scores, test_scores = [], {}, {}
    with threadpool_limits(limits=4):
        for name, model in models.items():
            model.fit(x_train, y_train)
            val_score = model.predict_proba(x_val)[:, 1]
            val_metrics = _metrics(y_val, val_score)
            metrics_rows.append({"model": name, "fold": "validation", **val_metrics})
            validation_scores[name] = val_metrics["ap"]
        selected_model = max(validation_scores, key=validation_scores.get)

        # A small validation subset makes model-agnostic interpretation inexpensive.
        rng = np.random.default_rng(SEED)
        indices = rng.choice(len(y_val), size=min(len(y_val), 10000), replace=False)
        if len(np.unique(y_val[indices])) == 2:
            importance = permutation_importance(
                models[selected_model], x_val[indices], y_val[indices],
                scoring="average_precision", n_repeats=2, random_state=SEED, n_jobs=1,
            )
            imp_df = pd.DataFrame({"feature": FEATURES, "mean_ap_decrease": importance.importances_mean,
                                   "std_ap_decrease": importance.importances_std})
        else:
            imp_df = pd.DataFrame({"feature": FEATURES, "mean_ap_decrease": np.nan,
                                   "std_ap_decrease": np.nan})
        imp_df.sort_values("mean_ap_decrease", ascending=False).to_csv(
            out_dir / "model_importance.csv", index=False)

        # Test scores are never used for model selection or parameter fitting.
        for name, model in models.items():
            score = model.predict_proba(x_test)[:, 1]
            test_scores[name] = score
            metrics_rows.append({"model": name, "fold": "test", **_metrics(y_test, score)})

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["selected_model"] = metrics_df["model"].eq(selected_model)
    metrics_df.to_csv(out_dir / "model_metrics.csv", index=False)

    # Rows have already been ordered by date/user_id; stable sorts resolve ties.
    rule_order = np.lexsort((np.arange(len(test)), -test["activity_4d"].to_numpy(dtype=float),
                            test["recency_days"].to_numpy(dtype=float),
                            -test["cart_4d"].to_numpy(dtype=float),
                            -test["cart_1d"].to_numpy(dtype=float)))
    policy_rows = _policy_metrics(y_test, None, "Random (expected)")
    policy_rows += _policy_metrics(y_test, rule_order, "Recent cart rule")
    for name, score in test_scores.items():
        order = np.argsort(-score, kind="stable")
        policy_rows += _policy_metrics(y_test, order, name, name == selected_model)
    policy_df = pd.DataFrame(policy_rows)
    policy_df.to_csv(out_dir / "targeting_policy.csv", index=False)
    _plot_diagnostics(y_test, test_scores, policy_df, selected_model, out_dir)
    return {
        "selected_model": selected_model,
        "metrics": metrics_df.to_dict("records"),
        "targeting": policy_df.to_dict("records"),
        "importance": imp_df.sort_values("mean_ap_decrease", ascending=False).to_dict("records"),
        "fold_sizes": {name: len(data[1]) for name, data in folds.items()},
    }
