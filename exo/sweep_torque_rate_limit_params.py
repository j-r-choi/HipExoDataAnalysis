"""Tabulate compute_rate_limit_metrics() across candidate rate_limit_torque() values,
for picking a value in the notebook before porting it into main.c."""
import pandas as pd

from .torque_rate_limit_sim import simulate_rate_limit_torque, compute_rate_limit_metrics


def sweep_torque_rate_limit_params(df, side, param_sets):
    """
    param_sets: list of (label, torque_rate_limit) tuples -- torque_rate_limit in
    [Nm] per 10ms sample, matching main.c's torqueRateLimit.
    Returns one row per candidate, best smoothing (max_rate_reduction_pct) first.
    """
    t = df["time"]
    fs = 1.0 / t.diff().median()
    td = df[f"{side}_torqueDesired"]

    rows = []
    for label, torque_rate_limit in param_sets:
        filt = simulate_rate_limit_torque(td, torque_rate_limit=torque_rate_limit)
        metrics = compute_rate_limit_metrics(td, filt, fs)
        rows.append({"label": label, "torque_rate_limit": torque_rate_limit, **metrics})

    return pd.DataFrame(rows).sort_values("max_rate_reduction_pct", ascending=False).reset_index(drop=True)
