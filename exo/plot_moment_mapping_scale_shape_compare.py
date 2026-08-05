"""Compare the Linear moment-to-torque mapping against the Peak-avg
mapping across a grid of scale x shape combinations -- see moment_mapping_sim.py."""
import matplotlib.pyplot as plt

from .moment_mapping_sim import simulate_linear_moment_mapping, simulate_peak_avg_moment_mapping


def plot_moment_mapping_scale_shape_compare(df, sim, side="left", scales=(0.2, 0.4), shapes=(0.5, 2.0),
                                             time_range=None, formula=simulate_peak_avg_moment_mapping,
                                             **peak_avg_params):
    """
    sim: simulate_gait_cycle(df, side) result, for istride (Peak-avg formula's stride boundary).
    side: defaults to "left" -- moment-mapping exploration is standardized on one leg.
    scales, shapes: swept as their full cartesian product -- each (scale, shape) combo gets
        its own trace. Defaults (0.2/0.4, 0.5/2.0) bracket the current moment_control.c
        default (scale=0.4, shape=1.0) to show the range of the redesign's effect.
    formula: which simulate_*_moment_mapping function to sweep -- defaults to Peak-avg,
        matching moment_control.c's actual cap_mode='hard'/ceiling_mode='matched'.
    peak_avg_params: overrides shared by all traces (e.g. cap_mode='soft').
    """
    t = df["time"]
    mp = df[f"{side}_momentPredicted"]

    linear = simulate_linear_moment_mapping(mp)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, mp, color="0.3", linewidth=1)
    axes[0].set_ylabel("Joint Moment (Nm/kg)")

    axes[1].axhline(0, color="gray", linewidth=1, zorder=0)
    axes[1].plot(t, linear, label="Linear", linewidth=1.5, color="0.2", linestyle="--")
    for scale in scales:
        for shape in shapes:
            peak_avg = formula(mp, sim["istride"], scale=scale, shape=shape, **peak_avg_params)
            axes[1].plot(t, peak_avg, label=f"Peak-avg (scale={scale}, shape={shape})", linewidth=1)
    axes[1].set_ylabel("Desired Torque (Nm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="best", fontsize=8)

    if time_range:
        for ax in axes:
            ax.set_xlim(time_range)

    fig.suptitle(f"{side} — Linear vs Peak-avg torque mapping across scale x shape")
    fig.tight_layout()
    return fig, axes
