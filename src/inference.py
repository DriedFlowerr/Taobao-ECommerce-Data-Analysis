"""Held-out association between recent cart behavior and next-day purchase."""

from math import sqrt
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, norm


Z95 = float(norm.ppf(0.975))


def _rate(n: int, buyers: int) -> dict:
    """Wilson interval remains informative when the observed rate is 0 or 1."""
    if n == 0:
        return {"n": n, "buyers": buyers, "purchase_rate": None,
                "ci_low": None, "ci_high": None}
    p = buyers / n
    scale = 1 + Z95**2 / n
    center = (p + Z95**2 / (2 * n)) / scale
    half = Z95 * sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2)) / scale
    return {"n": n, "buyers": buyers, "purchase_rate": p,
            "ci_low": min(p, max(0.0, center - half)),
            "ci_high": max(p, min(1.0, center + half))}


def _difference(cart: dict, no_cart: dict) -> dict:
    """Newcombe-Wilson CI for the difference of independent proportions."""
    if not cart["n"] or not no_cart["n"]:
        return {"risk_difference": None, "ci_low": None, "ci_high": None}
    p1, p0 = cart["purchase_rate"], no_cart["purchase_rate"]
    rd = p1 - p0
    lower = sqrt((p1 - cart["ci_low"])**2 + (no_cart["ci_high"] - p0)**2)
    upper = sqrt((cart["ci_high"] - p1)**2 + (p0 - no_cart["ci_low"])**2)
    return {"risk_difference": rd, "ci_low": max(-1.0, rd - lower),
            "ci_high": min(1.0, rd + upper)}


def _pvalue(cart: dict, no_cart: dict) -> dict:
    if not cart["n"] or not no_cart["n"]:
        return {"p_value": None, "test": "not estimable (one exposure group absent)"}
    table = np.array([[cart["buyers"], cart["n"] - cart["buyers"]],
                      [no_cart["buyers"], no_cart["n"] - no_cart["buyers"]]])
    if (table.sum(axis=0) == 0).any():
        return {"p_value": 1.0, "test": "constant outcome (no variation)"}
    _, p_chi2, _, expected = chi2_contingency(table, correction=False)
    if (expected < 5).any():
        return {"p_value": float(fisher_exact(table, alternative="two-sided").pvalue),
                "test": "Fisher exact (two-sided)"}
    return {"p_value": float(p_chi2), "test": "Pearson chi-square (two-sided)"}


def _group_rates(frame: pd.DataFrame) -> tuple[dict, dict]:
    no_cart = frame.loc[~frame["had_cart"], "y"]
    cart = frame.loc[frame["had_cart"], "y"]
    return _rate(len(cart), int(cart.sum())), _rate(len(no_cart), int(no_cart.sum()))


def _prepare_test(snapshots: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    fields = ["user_id", "fold", "y", "cart_4d", "buy_4d", "active_days_4d"]
    missing = set(fields) - set(snapshots.columns)
    if missing:
        raise ValueError(f"Missing snapshot columns: {sorted(missing)}")
    test = snapshots.loc[snapshots["fold"].eq("test"), fields].copy()
    n_original = len(test)

    # Check the independence unit; never silently retain a repeated user's first row.
    if test.loc[test["user_id"].notna(), "user_id"].duplicated().any():
        raise ValueError("Each user must occur at most once in the test fold.")
    numeric = ["y", "cart_4d", "buy_4d", "active_days_4d"]
    for column in numeric:
        test[column] = pd.to_numeric(test[column], errors="coerce")
    valid = (
        test["user_id"].notna()
        & test[numeric].notna().all(axis=1)
        & np.isfinite(test[numeric]).all(axis=1)
        & test["y"].isin([0, 1])
        & (test[["cart_4d", "buy_4d"]] >= 0).all(axis=1)
        & test["active_days_4d"].isin([1, 2, 3, 4])
    )
    test = test.loc[valid].copy()
    test["had_cart"] = test["cart_4d"].gt(0)
    test["prior_purchase"] = test["buy_4d"].gt(0)
    test["activity"] = pd.cut(test["active_days_4d"], [0, 1, 3, 4],
                              labels=["1 day", "2-3 days", "4 days"])
    return test, n_original


def _standardize(strata: list[dict], n_total: int) -> dict:
    """Weight stratum differences by their observed common-support population."""
    supported = [s for s in strata if s["risk_difference"] is not None]
    n_supported = sum(s["cart_n"] + s["no_cart_n"] for s in supported)
    result = {"risk_difference": None, "ci_low": None, "ci_high": None,
              "n_common_support": n_supported,
              "common_support_fraction": n_supported / n_total if n_total else None,
              "method": "Direct standardization across cart/no-cart overlap strata; "
                        "approximate Wilson-combined CI conditional on stratum weights"}
    if not n_supported:
        return result

    rd, low_sq, high_sq = 0.0, 0.0, 0.0
    for s in supported:
        weight = (s["cart_n"] + s["no_cart_n"]) / n_supported
        rd += weight * s["risk_difference"]
        low_sq += (weight * (s["risk_difference"] - s["ci_low"]))**2
        high_sq += (weight * (s["ci_high"] - s["risk_difference"]))**2
    result.update(risk_difference=rd, ci_low=max(-1.0, rd - sqrt(low_sq)),
                  ci_high=min(1.0, rd + sqrt(high_sq)))
    return result


def _plot_groups(groups: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, row in enumerate(groups):
        p = row["purchase_rate"]
        if p is None:
            ax.text(i, 0.01, "No users", ha="center", va="bottom")
            continue
        ax.bar(i, p * 100, width=0.55, color=["#8ba9bd", "#227c9d"][i])
        ax.errorbar(i, p * 100,
                    yerr=[[(p - row["ci_low"]) * 100], [(row["ci_high"] - p) * 100]],
                    fmt="none", color="#253847", capsize=4)
        ax.annotate(f"{p:.2%}\n(n={row['n']:,})", (i, row["ci_high"] * 100),
                    xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks([0, 1], ["No recent cart", "Recent cart"])
    ax.set_ylabel("Next-day purchase rate (%)")
    ax.set_title("Recent cart activity and subsequent purchase")
    ax.margins(y=0.22)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def analyze_association(snapshots: pd.DataFrame, out_dir: Path) -> dict:
    """Estimate association, not causal lift, using one held-out row per user.

    Six strata are defined in advance by active days in the previous four days
    (1, 2-3, 4) crossed with whether the user previously purchased. Standardizing
    controls only these two measured dimensions, so residual confounding remains.
    All rates, intervals and differences are fractions, not percentage points.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test, n_original = _prepare_test(snapshots)
    cart, no_cart = _group_rates(test)
    groups = [{"group": "no_cart", **no_cart}, {"group": "cart", **cart}]
    crude = {**_difference(cart, no_cart), **_pvalue(cart, no_cart),
             "method": "Unadjusted difference (Newcombe-Wilson 95% CI)"}

    strata = []
    for activity in ["1 day", "2-3 days", "4 days"]:
        for prior in [False, True]:
            subgroup = test.loc[test["activity"].eq(activity) & test["prior_purchase"].eq(prior)]
            c, nc = _group_rates(subgroup)
            strata.append({"activity": activity, "prior_purchase": prior,
                           **{f"cart_{k}": v for k, v in c.items()},
                           **{f"no_cart_{k}": v for k, v in nc.items()},
                           **_difference(c, nc)})
    standardized = _standardize(strata, len(test))

    pd.DataFrame(groups).to_csv(out_dir / "association_groups.csv", index=False)
    pd.DataFrame(strata).to_csv(out_dir / "association_strata.csv", index=False)
    pd.DataFrame([{"estimate": "crude", **crude},
                  {"estimate": "standardized", **standardized}]).to_csv(
                      out_dir / "association_effects.csv", index=False)
    _plot_groups(groups, out_dir / "association_cart.png")

    return {"rows_test": n_original, "rows_analyzed": len(test),
            "rows_dropped": n_original - len(test), "groups": groups,
            "crude": crude, "standardized": standardized, "strata": strata,
            "interpretation": "Association only; no randomized treatment or exposure "
                              "assignment log. Standardizing by observed history does not "
                              "justify a causal effect of cart behavior or marketing."}


def _self_test() -> None:
    test = pd.DataFrame({"user_id": range(10), "fold": ["test"] * 9 + ["train"],
                         "y": [0, 1, 1, 1, 0, 0, 0, 1, None, 1],
                         "cart_4d": [0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
                         "buy_4d": [0] * 10, "active_days_4d": [1] * 10})
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        result = analyze_association(test, out)
        assert result["rows_test"] == 9 and result["rows_analyzed"] == 8
        assert result["crude"]["risk_difference"] == 1.0
        assert result["standardized"]["n_common_support"] == 8
        assert 0 <= result["crude"]["p_value"] <= 1
        assert len(list(out.iterdir())) == 4
        empty = analyze_association(test.loc[test["fold"].eq("train")], out)
        assert empty["crude"]["risk_difference"] is None
        one_group = analyze_association(test.loc[test["cart_4d"].eq(0)], out)
        assert one_group["crude"]["p_value"] is None
        assert one_group["standardized"]["n_common_support"] == 0
        constant = analyze_association(test.assign(y=0), out)
        assert constant["crude"]["p_value"] == 1.0
        nullable = test.assign(cart_4d=pd.array(test["cart_4d"], dtype="Int64"))
        nullable.loc[0, "cart_4d"] = pd.NA
        assert analyze_association(nullable, out)["rows_dropped"] == 2


if __name__ == "__main__":
    _self_test()
    print("Association self-test passed.")
