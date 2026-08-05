"""Compares Peak-avg (peak drifts with shape) against the current final
Delayed-rise formula (peak stays exactly invariant to shape) -- see
moment_mapping_sim.py.

Peak-avg normalizes against a multi-stride trailing average; x**shape is
shape-invariant only at x in {0,1}, so any other trailing-average value lets
shape shift the peak. Delayed-rise normalizes against each stride's own causal
running peak instead and delays the "is this the peak?" decision by
rise_delay_samples -- zero-delay peak-invariance-with-climb-shaping is
provably impossible for any causal algorithm (two strides with identical
history up to sample i are indistinguishable there), so the bounded delay is
what makes delayed-rise's invariance exact rather than approximate.

Delayed-rise's output is shown at its actual delivery time (shifted forward by
rise_delay_samples via shift_for_delivery), not the same-index shape-comparison
view -- i.e. what the real firmware would actually be commanding at each instant,
delay included."""
import matplotlib.pyplot as plt

from .moment_mapping_sim import (
    simulate_peak_avg_moment_mapping,
    simulate_delayed_rise_moment_mapping,
    shift_for_delivery,
    RISE_DELAY_SAMPLES,
)


def plot_moment_mapping_peak_invariance(df, sim, side="left", shapes=(0.5, 1.0, 2.0),
                                         time_range=None, **params):
    """
    sim: simulate_gait_cycle(df, side) result, for istride (Peak-avg formula's stride boundary).
    side: defaults to "left" -- moment-mapping exploration is standardized on one leg.
    shapes: shape (lambda) values to overlay on each formula, to make peak drift (or its
        absence) visible.
    params: overrides shared by both formulas (e.g. scale=0.4). rise_delay_samples for
        the Delayed-rise formula defaults to RISE_DELAY_SAMPLES unless overridden.
    """
    t = df["time"]
    mp = df[f"{side}_momentPredicted"]
    istride = sim["istride"]
    rise_delay_samples = params.pop("rise_delay_samples", RISE_DELAY_SAMPLES)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True)
    axes[0].axhline(0, color="gray", linewidth=1, zorder=0)
    axes[1].axhline(0, color="gray", linewidth=1, zorder=0)

    for shape in shapes:
        peak_avg = simulate_peak_avg_moment_mapping(mp, istride, shape=shape, **params)
        axes[0].plot(t, peak_avg, label=f"shape={shape}", linewidth=1)
    axes[0].set_title("Peak-avg -- peak drifts with shape")
    axes[0].set_ylabel("Desired Torque (Nm)")
    axes[0].legend(loc="best", fontsize=8)

    for shape in shapes:
        final_raw = simulate_delayed_rise_moment_mapping(mp, istride, shape=shape,
                                                           rise_delay_samples=rise_delay_samples, **params)
        final = shift_for_delivery(final_raw, rise_delay_samples)
        axes[1].plot(t, final, label=f"shape={shape}", linewidth=1)
    axes[1].set_title("Delayed-rise (current final, actual delivery timing) "
                       "-- peak invariant to shape")
    axes[1].set_ylabel("Desired Torque (Nm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="best", fontsize=8)

    if time_range:
        for ax in axes:
            ax.set_xlim(time_range)

    fig.suptitle(f"{side} — peak-shape invariance: Peak-avg vs Delayed-rise (final, real-time view)")
    fig.tight_layout()
    return fig, axes
