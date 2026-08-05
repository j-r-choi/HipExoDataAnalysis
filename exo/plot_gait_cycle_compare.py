"""Python port of the 'Gait Cycle' section of reference/plot_data_sim.m:
MCU-logged gaitCycle/torqueDesired vs an offline re-simulation from raw IMU data."""
import matplotlib.pyplot as plt


def plot_gait_cycle_compare(df, side, sim, time_range=None, sim_fixed=None, fixed_label="Simulation (fixed)"):
    t = df["time"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, df[f"{side}_gaitCycle"], label="MCU")
    axes[0].plot(sim["time"], sim["gaitCycleNorm"], "--", label="Simulation")
    if sim_fixed is not None:
        axes[0].plot(sim_fixed["time"], sim_fixed["gaitCycleNorm"], ":", label=fixed_label)
    axes[0].set_ylabel("Gait Cycle (%)")
    axes[0].legend(loc="best")

    axes[1].plot(t, df[f"{side}_torqueDesired"], label="MCU")
    axes[1].plot(sim["time"], sim["torqueDesired"], "--", label="Simulation")
    if sim_fixed is not None:
        axes[1].plot(sim_fixed["time"], sim_fixed["torqueDesired"], ":", label=fixed_label)
    axes[1].set_ylabel("Desired Torque (Nm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="best")

    if time_range:
        for ax in axes:
            ax.set_xlim(time_range)

    fig.suptitle(f"{side} — MCU vs offline simulation")
    fig.tight_layout()
    return fig, axes
