"""Compares moment_predicted before/after filter_dip_fill(), and the resulting
Delayed-rise (istride-buffered) desired torque before/after.

filter_dip_fill() (see dip_fill_sim.py) exists because momentPredicted
sometimes shows a small secondary local peak+dip mid-phase before its genuine
final decline -- a confirmed real, recurring artifact (e.g. HIP006.csv), not
noise. This module is the visual check for that fix: panel 1 shows raw vs.
filtered moment, panel 2 shows how the artifact (and its removal) propagates
into the downstream torque mapping."""
import matplotlib.pyplot as plt

from .dip_fill_sim import filter_dip_fill
from .moment_mapping_sim import simulate_delayed_rise_moment_mapping_istride_buffered


def plot_dip_fill_compare(df, sim, side="left", time_range=None, shape=1.0,
                           delay_ms=200.0, rise_delay_samples=10, **filter_params):
    """
    sim: simulate_gait_cycle(df, side) result, for istride.
    side: defaults to "left" -- moment-mapping exploration is standardized on one leg.
    shape, delay_ms, rise_delay_samples: passed through to the Delayed-rise
        (istride-buffered) mapping formula for the before/after torque panel.
    filter_params: overrides for filter_dip_fill (D, min_seg_height, recover_frac).
        Shipped defaults (D=20 samples/200ms, min_seg_height=0.025 Nm/kg,
        recover_frac=0.85) are tuned, not arbitrary: min_seg_height was relaxed
        from an original 0.035 to rescue small dips in noisy pre-walk data, and
        recover_frac is deliberately NOT relaxed further -- lower values were
        found to wrongly bridge a genuine decline into a smaller hump
        elsewhere. By design, some dips still won't get filled (cases sitting
        right at the recover_frac cutoff, low-prominence noise-period dips,
        and stale-reference-peak cases in noisy tails) -- this is expected,
        not a bug to chase.
    """
    t = df["time"]
    mp = df[f"{side}_momentPredicted"]
    istride = sim["istride"]

    filtered = filter_dip_fill(mp, istride, **filter_params)

    torque_before = simulate_delayed_rise_moment_mapping_istride_buffered(
        mp, istride, shape=shape, delay_ms=delay_ms, rise_delay_samples=rise_delay_samples)
    torque_after = simulate_delayed_rise_moment_mapping_istride_buffered(
        filtered, istride, shape=shape, delay_ms=delay_ms, rise_delay_samples=rise_delay_samples)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t, mp, color="0.5", linewidth=1.5, label="raw")
    axes[0].plot(t, filtered, color="tab:blue", linewidth=1.2, linestyle="--", label="filtered (filter_dip_fill)")
    axes[0].axhline(0, color="gray", linewidth=1, zorder=0)
    axes[0].set_ylabel("Joint Moment (Nm/kg)")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_title("Joint moment: raw vs filtered")

    axes[1].plot(t, torque_before, color="0.5", linewidth=1.5, label="BEFORE (raw moment)")
    axes[1].plot(t, torque_after, color="tab:blue", linewidth=1.2, linestyle="--", label="AFTER (filtered moment)")
    axes[1].axhline(0, color="gray", linewidth=1, zorder=0)
    axes[1].set_ylabel("Desired Torque (Nm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].set_title("Delayed-rise (istride-buffered) desired torque: BEFORE vs AFTER dip-fill")

    if time_range:
        for ax in axes:
            ax.set_xlim(time_range)

    fig.suptitle(f"{side} — dip-fill pre-processing: moment and resulting torque")
    fig.tight_layout()
    return fig, axes
