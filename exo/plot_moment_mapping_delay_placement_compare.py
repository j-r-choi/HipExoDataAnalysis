"""Compares Delayed-rise's two ways of placing delay_ms's ring buffer, across shape.

delay_ms (the Jetson moment-estimate latency) needs a ring buffer somewhere, so a
stale-but-just-arrived moment estimate gets paired with the right stride context.
Moment-buffered (simulate_delayed_rise_moment_mapping) ring-buffers `m` before the
peak-tracking loop but leaves `istride` real-time, so the two disagree about "now"
by delay_samples around every stride boundary -- this is what's actually live in
moment_control.c today, which is why this package's other, plainer _compare plots
intentionally still show moment-buffered behavior. Istride-buffered
(simulate_delayed_rise_moment_mapping_istride_buffered) ring-buffers `istride`
instead (a real streaming FIFO, matching how MCU firmware would have to do it),
keeping `m`/`istride` time-matched -- the correct alignment, shown here for contrast.

Note: moment-buffered's stride-boundary misalignment can sometimes visually cancel
the separate, real torque spike caused by peak-tracking resetting to 0 at each new
stride -- that's two independent quirks coincidentally offsetting in places, not
evidence moment-buffered is the better choice; don't read too much into agreement
in any one dataset.

Overlaying shape=0.5/1.0/2.0 on each panel lets you see whether the proven
peak-shape invariance survives the moment-buffered misalignment or only holds
cleanly for the istride-buffered version."""
import matplotlib.pyplot as plt

from .moment_mapping_sim import (
    simulate_delayed_rise_moment_mapping,
    simulate_delayed_rise_moment_mapping_istride_buffered,
    shift_for_delivery,
    RISE_DELAY_SAMPLES,
)


def plot_moment_mapping_delay_placement_compare(df, sim, side="left", shapes=(0.5, 1.0, 2.0),
                                                 time_range=None, delay_ms=200.0, fs=100.0,
                                                 rise_delay_samples=RISE_DELAY_SAMPLES, **params):
    """
    sim: simulate_gait_cycle(df, side) result, for istride (real-time stride boundary).
    side: defaults to "left" -- moment-mapping exploration is standardized on one leg.
    shapes: shape (lambda) values to overlay on each panel, to make peak drift (or its
        absence) visible.
    delay_ms, fs, rise_delay_samples: shared by both panels.
    params: further overrides shared by both panels (e.g. scale=0.4).
    """
    t = df["time"]
    mp = df[f"{side}_momentPredicted"]
    istride = sim["istride"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True)
    axes[0].axhline(0, color="gray", linewidth=1, zorder=0)
    axes[1].axhline(0, color="gray", linewidth=1, zorder=0)

    for shape in shapes:
        raw = simulate_delayed_rise_moment_mapping(
            mp, istride, shape=shape, delay_ms=delay_ms, fs=fs,
            rise_delay_samples=rise_delay_samples, **params)
        moment_buffered = shift_for_delivery(raw, rise_delay_samples)
        axes[0].plot(t, moment_buffered, label=f"shape={shape}", linewidth=1)
    axes[0].set_title("Delayed-rise (moment-buffered) -- delay_ms's ring buffer on m, istride real-time")
    axes[0].set_ylabel("Desired Torque (Nm)")
    axes[0].legend(loc="best", fontsize=8)

    for shape in shapes:
        istride_buffered_raw = simulate_delayed_rise_moment_mapping_istride_buffered(
            mp, istride, shape=shape, delay_ms=delay_ms, fs=fs,
            rise_delay_samples=rise_delay_samples, **params)
        istride_buffered = shift_for_delivery(istride_buffered_raw, rise_delay_samples)
        axes[1].plot(t, istride_buffered, label=f"shape={shape}", linewidth=1)
    axes[1].set_title("Delayed-rise (istride-buffered) -- delay_ms's ring buffer on istride, m/istride time-matched")
    axes[1].set_ylabel("Desired Torque (Nm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="best", fontsize=8)

    if time_range:
        for ax in axes:
            ax.set_xlim(time_range)

    fig.suptitle(f"{side} — Delayed-rise: moment-buffered vs istride-buffered across shape (delay_ms={delay_ms:g})")
    fig.tight_layout()
    return fig, axes
