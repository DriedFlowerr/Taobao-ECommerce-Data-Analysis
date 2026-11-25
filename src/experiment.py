"""Prospective randomized experiment planning; no observed treatment data."""

from math import ceil, sqrt
from pathlib import Path

import pandas as pd
from scipy.stats import norm


def plan_trial(targeting: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Estimate equal-allocation sample size for illustrative absolute MDEs."""
    row = targeting.loc[targeting.selected_model.eq(True) & targeting.budget_fraction.eq(0.10)]
    if len(row) != 1:
        raise ValueError('Expected exactly one selected-model row at the 10% budget.')
    p0 = float(row.precision.iloc[0])
    alpha, power = 0.05, 0.80
    z_alpha, z_power = norm.ppf(1 - alpha / 2), norm.ppf(power)
    rows = []
    for delta in (0.005, 0.01, 0.02):
        p1 = p0 + delta
        if not 0 < p0 < p1 < 1:
            continue
        pbar = (p0 + p1) / 2
        n = ceil((z_alpha * sqrt(2 * pbar * (1 - pbar)) +
                  z_power * sqrt(p0 * (1 - p0) + p1 * (1 - p1)))**2 / delta**2)
        rows.append({'historical_targeted_purchase_rate': p0, 'hypothetical_mde_pp': delta * 100,
                     'alpha_two_sided': alpha, 'power': power, 'users_per_arm': n,
                     'total_distinct_users': 2 * n})
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / 'ab_sample_size_plan.csv', index=False)
    return result
