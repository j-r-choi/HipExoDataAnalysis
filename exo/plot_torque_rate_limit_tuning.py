"""Overlay candidate rate_limit_torque() values against the existing (recorded,
pre-limit) torqueDesired, for offline tuning before porting a chosen value into main.c."""
import matplotlib.pyplot as plt
import numpy as np

from .torque_rate_limit_sim import simulate_rate_limit_torque


def plot_torque_rate_limit_tuning(df, side, param_sets, time_range=None):
    """
    param_sets: list of (label, torque_rate_limit) tuples to compare -- torque_rate_limit
    in [Nm] per 10ms sample, matching main.c's torqueRateLimit.

    Top panel: torque value, raw vs each candidate -- smoothing is subtle here.
    Bottom panel: d(torque)/dt, where rate-limiting's effect is actually visible
    (each candidate's rate gets clipped at its own torque_rate_limit).
    """
    t = df["time"]
    fs = 1.0 / t.diff().median()
    td = df[f"{side}_torqueDesired"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, td, label="Existing (recorded, pre-limit)", color="0.3", linewidth=1)
    axes[1].plot(t[1:], np.diff(td) * fs, color="0.3", linewidth=1, label="Existing")

    for label, torque_rate_limit in param_sets:
        filt = simulate_rate_limit_torque(td, torque_rate_limit=torque_rate_limit)
        line, = axes[0].plot(t, filt, label=label, linewidth=1.5)
        axes[1].plot(t[1:], np.diff(filt) * fs, label=label, linewidth=1.5, color=line.get_color())

    axes[0].set_ylabel("Desired Torque (Nm)")
    axes[0].legend(loc="best", fontsize=7)

    axes[1].set_ylabel("d(Torque)/dt (Nm/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="best", fontsize=7)

    if time_range:
        for ax in axes:
            ax.set_xlim(time_range)

    fig.suptitle(f"{side} — torque rate-limit tuning")
    fig.tight_layout()
    return fig, axes
