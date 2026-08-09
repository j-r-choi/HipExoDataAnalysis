"""Plot motor current, desired/measured torque and (optionally) predicted moment from a log CSV.

Opens an interactive matplotlib window (toolbar: pan, box-zoom, scroll, home to reset) so
nothing is lost to raster resolution. Edit the CONFIG block below, then just run the file:

    python misc_files/plot_torque_current.py
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- CONFIG ----
CSV = Path(__file__).parent / "filter_expl" / "HIP012.csv"

SIDES = ["right", "left"]      # ["right"], ["left"] or ["right", "left"]
SHOW_CURRENT = True            # motorCurrent in its own subplot under the torque plot
SHOW_MOMENT = False            # <side>_momentPredicted on a right-hand axis of the torque plot

# eStop trip thresholds, mirroring Core/Src/main.c:181-185. Set an entry to None to hide it.
SHOW_LIMITS = True
TORQUE_UPPER_LIMIT = 27.0      # [Nm]  torqueUpperLimit
TORQUE_LOWER_LIMIT = -48.0     # [Nm]  torqueLowerLimit
MOTOR_CURRENT_LIMIT = 60.0     # [A]   motorCurLimit, applied as +/- (fabsf in the firmware)
LIMITS_IN_VIEW = False         # True: y-range always spans the caps (squashes the signal)
                               # False: y-range follows the data, caps may sit off-screen

TMIN = None                    # start time [s], None = from the beginning
TMAX = None                    # end time [s],   None = to the end

SAVE_PNG = None                # e.g. Path("fig.png") to also write a file; None = window only
# -----------------------------------------------------------------------------

C_DESIRED = "#ff7f0e"
C_MEASURED = "#1f77b4"
C_CURRENT = "#d62728"
C_MOMENT = "#2ca02c"
C_LIMIT = "#555555"


def draw_limits(ax, values, label):
    """Dotted eStop caps. Keeps the data-driven y-range unless LIMITS_IN_VIEW is set."""
    values = [v for v in values if v is not None]
    if not SHOW_LIMITS or not values:
        return []
    ylim = ax.get_ylim()
    lines = [ax.axhline(v, color=C_LIMIT, ls=":", lw=1.2) for v in values]
    if not LIMITS_IN_VIEW:
        ax.set_ylim(ylim)
    lines[0].set_label(label)  # one legend entry for the whole pair
    return [lines[0]]


def plot_torque(ax, t, df, side):
    """Desired vs measured torque, with predicted moment on a twin axis if enabled."""
    handles = [
        ax.plot(t, df[f"{side}_torqueDesired"], color=C_DESIRED,
                lw=1.2, label="torque desired")[0],
        ax.plot(t, df[f"{side}_torqueMeasured"], color=C_MEASURED,
                lw=1.0, label="torque measured")[0],
    ]
    ax.set_ylabel(f"{side} torque [Nm]")
    ax.grid(alpha=0.25)
    handles += draw_limits(ax, [TORQUE_UPPER_LIMIT, TORQUE_LOWER_LIMIT], "eStop torque cap")

    if SHOW_MOMENT:
        ax_m = ax.twinx()
        handles.append(ax_m.plot(t, df[f"{side}_momentPredicted"], color=C_MOMENT,
                                 lw=1.0, label="moment predicted")[0])
        ax_m.set_ylabel("moment [Nm/kg]", color=C_MOMENT)
        ax_m.tick_params(axis="y", labelcolor=C_MOMENT)

    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)


def plot_current(ax, t, df, side):
    """Motor current in its own subplot."""
    handles = [ax.plot(t, df[f"{side}_motorCurrent"], color=C_CURRENT,
                       lw=0.8, label="motor current")[0]]
    ax.set_ylabel(f"{side} current [A]")
    ax.grid(alpha=0.25)
    lim = MOTOR_CURRENT_LIMIT
    handles += draw_limits(ax, [lim, -lim if lim is not None else None], "eStop current cap")
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)


def main():
    df = pd.read_csv(CSV)
    if TMIN is not None:
        df = df[df["time"] >= TMIN]
    if TMAX is not None:
        df = df[df["time"] <= TMAX]
    if df.empty:
        raise SystemExit("no samples in the requested time window")
    t = df["time"].values

    # one torque row per side, each followed by its own current row
    rows = [(side, kind) for side in SIDES
            for kind in (["torque", "current"] if SHOW_CURRENT else ["torque"])]
    heights = [3 if kind == "torque" else 1.6 for _, kind in rows]

    fig, axes = plt.subplots(len(rows), 1, figsize=(16, sum(heights) * 1.2),
                             sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": heights})
    for ax, (side, kind) in zip(axes[:, 0], rows):
        if kind == "torque":
            plot_torque(ax, t, df, side)
        else:
            plot_current(ax, t, df, side)
    axes[-1, 0].set_xlabel("time [s]")
    fig.suptitle(f"{CSV.stem} - torque"
                 + (", motor current" if SHOW_CURRENT else "")
                 + (" and predicted moment" if SHOW_MOMENT else ""))
    fig.tight_layout()

    if SAVE_PNG:
        fig.savefig(SAVE_PNG, dpi=150)
        print(f"saved {SAVE_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
