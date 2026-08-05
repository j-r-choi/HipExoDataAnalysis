"""Sweeps rise_delay_samples (K) to show how much of a stride's rise gets
shape-reshaped vs. how much extra latency it costs.

Making shape reshape a stride's climb (not just its post-peak decline) while
keeping the peak exactly invariant is provably impossible at K=0 for any
causal algorithm: two strides with identical history up to sample i but
differing after it look identical at sample i, so peak invariance forces
shape invariance on the whole climb. Bounded extra delay K fixes this by
turning "was that the peak?" into a question answerable in hindsight.
K=15 (150ms) was picked from real onset-to-peak timing (~260ms flexion,
~750ms extension in HIP006) and a false-record-rate elbow -- most of the
benefit is already captured by K=15, larger K buys little more. Peak
invariance itself is exact for any K>=0; K only trades reshape-of-climb
against added latency. See moment_mapping_sim.py for the
simulate_delayed_rise_moment_mapping implementation this sweeps."""
import matplotlib.pyplot as plt

from .moment_mapping_sim import (
    simulate_delayed_rise_moment_mapping_istride_buffered,
    shift_for_delivery,
)


def plot_moment_mapping_rise_delay_sweep(df, sim, side="left", time_range=None,
                                          rise_delay_samples_list=(0, 5, 10, 15, 20, 30),
                                          **params):
    """
    sim: simulate_gait_cycle(df, side) result, for istride.
    side: defaults to "left" -- moment-mapping exploration is standardized on one leg.
    rise_delay_samples_list: K values (samples, 10ms each) to compare, each shown
        at its actual delivery timing (shift_for_delivery), not the same-index view.
    params: overrides shared by all traces (e.g. shape=2.0, scale=0.4).
    """
    t = df["time"]
    mp = df[f"{side}_momentPredicted"]
    istride = sim["istride"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, mp, color="0.3", linewidth=1)
    axes[0].set_ylabel("Joint Moment (Nm/kg)")

    axes[1].axhline(0, color="gray", linewidth=1, zorder=0)
    for k in rise_delay_samples_list:
        raw = simulate_delayed_rise_moment_mapping_istride_buffered(
            mp, istride, rise_delay_samples=k, **params)
        delivered = shift_for_delivery(raw, k)
        axes[1].plot(t, delivered, label=f"K={k} (+{k * 10:.0f}ms)", linewidth=1)
    axes[1].set_ylabel("Desired Torque (Nm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="best", fontsize=8, title="rise_delay_samples")

    if time_range:
        for ax in axes:
            ax.set_xlim(time_range)

    fig.suptitle(f"{side} — torque profile across rise_delay_samples (K), actual delivery timing")
    fig.tight_layout()
    return fig, axes
