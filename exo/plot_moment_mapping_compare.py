"""Compare the Linear and Peak-avg moment-to-torque mapping,
simulated from the same logged momentPredicted -- see moment_mapping_sim.py."""
import matplotlib.pyplot as plt

from .moment_mapping_sim import simulate_linear_moment_mapping, simulate_peak_avg_moment_mapping


def plot_moment_mapping_compare(df, sim, side="left", time_range=None, show_soft_cap=True, **peak_avg_params):
    """
    sim: simulate_gait_cycle(df, side) result, for istride (Peak-avg formula's stride boundary).
    side: defaults to "left" -- moment-mapping exploration is standardized on one leg.
    peak_avg_params: overrides shared by both Peak-avg curves (e.g. shape=2.0).
    show_soft_cap: also plot the soft-cap variant (cap_mode='soft') alongside the hard-cap one.
    """
    t = df["time"]
    mp = df[f"{side}_momentPredicted"]

    linear = simulate_linear_moment_mapping(mp)
    peak_avg_hard = simulate_peak_avg_moment_mapping(mp, sim["istride"], cap_mode="hard", **peak_avg_params)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, mp, color="0.3", linewidth=1)
    axes[0].set_ylabel("Joint Moment (Nm/kg)")

    axes[1].axhline(0, color="gray", linewidth=1, zorder=0)
    axes[1].plot(t, linear, label="Linear", linewidth=1)
    axes[1].plot(t, peak_avg_hard, label="Peak-avg (hard cap)", linewidth=1)
    if show_soft_cap:
        peak_avg_soft = simulate_peak_avg_moment_mapping(mp, sim["istride"], cap_mode="soft", **peak_avg_params)
        axes[1].plot(t, peak_avg_soft, label="Peak-avg (soft cap)", linewidth=1)
    axes[1].set_ylabel("Desired Torque (Nm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="best", fontsize=8)

    if time_range:
        for ax in axes:
            ax.set_xlim(time_range)

    fig.suptitle(f"{side} — Linear vs Peak-avg torque mapping (simulated)")
    fig.tight_layout()
    return fig, axes
