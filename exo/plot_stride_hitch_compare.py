"""Compare how often each leg's swing shows the stride-to-stride hitch found by
detect_stride_hitch() -- angle trace with detected hitches marked, plus motor
current on a secondary axis to see how much current (if any) is present when
the hitch happens."""
import matplotlib.pyplot as plt

from .detect_stride_hitch import detect_stride_hitch


def plot_stride_hitch_compare(df, angle_col="motorAngle", depth_thresh=5.0, time_range=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for ax, side in zip(axes, ("left", "right")):
        line_angle, = ax.plot(df["time"], df[f"{side}_{angle_col}"], linewidth=1, color="0.3",
                               label=f"{side}_{angle_col}")

        hitches = detect_stride_hitch(df, side, angle_col=angle_col)
        sig = hitches[hitches["depth_deg"] > depth_thresh]
        hitch_scatter = ax.scatter(sig["t_notch"], sig["angle_notch"], color="tab:red", zorder=3, s=15,
                                    label=f"hitch >{depth_thresh:.0f}° (n={len(sig)}/{len(hitches)} candidates)")

        ax_current = ax.twinx()
        line_current, = ax_current.plot(df["time"], df[f"{side}_motorCurrent"], linewidth=1,
                                         color="tab:blue", alpha=0.5, label=f"{side}_motorCurrent")
        ax_current.set_ylabel("Current (A)", color="tab:blue", fontsize=8)
        ax_current.tick_params(axis="y", labelcolor="tab:blue")

        ax.set_ylabel(f"{side}_{angle_col}")
        ax.legend(handles=[line_angle, hitch_scatter, line_current], loc="best", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    if time_range:
        for ax in axes:
            ax.set_xlim(time_range)

    fig.suptitle(f"Stride hitch comparison ({angle_col}) with motor current")
    fig.tight_layout()
    return fig, axes
