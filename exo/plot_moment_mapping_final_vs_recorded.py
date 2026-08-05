"""Compares the actually-recorded torqueDesired (whatever formula was live in
moment_control.c when this CSV was logged) against the current final formula
(Delayed-rise, istride-buffered) re-simulated from the same logged
momentPredicted -- see moment_mapping_sim.py."""
import matplotlib.pyplot as plt

from .moment_mapping_sim import (
    simulate_delayed_rise_moment_mapping_istride_buffered,
    shift_for_delivery,
    RISE_DELAY_SAMPLES,
)


def plot_moment_mapping_final_vs_recorded(df, sim, side="left", time_range=None,
                                           rise_delay_samples=RISE_DELAY_SAMPLES, **params):
    """
    sim: simulate_gait_cycle(df, side) result, for istride.
    side: defaults to "left" -- moment-mapping exploration is standardized on one leg.
    Current formula shown at its actual delivery timing (shift_for_delivery), matching
    what the firmware would really command, not the same-index comparison view used
    elsewhere in this notebook.
    params: overrides for the current formula (e.g. shape=1.0, scale=0.4, delay_ms=0.0).
    """
    t = df["time"]
    mp = df[f"{side}_momentPredicted"]
    recorded = df[f"{side}_torqueDesired"]
    istride = sim["istride"]

    raw = simulate_delayed_rise_moment_mapping_istride_buffered(
        mp, istride, rise_delay_samples=rise_delay_samples, **params)
    current = shift_for_delivery(raw, rise_delay_samples)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, mp, color="0.3", linewidth=1)
    axes[0].set_ylabel("Joint Moment (Nm/kg)")

    axes[1].axhline(0, color="gray", linewidth=1, zorder=0)
    axes[1].plot(t, recorded, label="Recorded (MCU, as logged)", linewidth=1, color="0.3")
    axes[1].plot(t, current, label="Current (Delayed-rise, istride-buffered)", linewidth=1, color="tab:red")
    axes[1].set_ylabel("Desired Torque (Nm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="best", fontsize=8)

    if time_range:
        for ax in axes:
            ax.set_xlim(time_range)

    fig.suptitle(f"{side} — recorded (MCU) vs current formula torqueDesired")
    fig.tight_layout()
    return fig, axes
