"""Python port of reference/plot_sim_phase_angle.m: angle, angular velocity, phase
angle, and gait cycle from one simulate_gait_cycle() run, stacked on one figure."""
import matplotlib.pyplot as plt


def plot_gait_algorithm(sim, side, time_range=None):
    t = sim["time"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(t, sim["angleSmooth"])
    axes[0].set_ylabel("Angle (deg)")

    axes[1].plot(t, sim["velocitySmooth"])
    axes[1].set_ylabel("Angular Velocity (deg/s)")

    axes[2].plot(t, sim["phaseAngleSmooth"])
    axes[2].set_ylabel("Phase Angle (rad)")

    axes[3].plot(t, sim["gaitCycleNorm"])
    axes[3].set_ylabel("Gait Cycle (%)")
    axes[3].set_xlabel("Time (s)")

    if time_range:
        for ax in axes:
            ax.set_xlim(time_range)

    fig.suptitle(f"{side} — gait detection algorithm")
    fig.tight_layout()
    return fig, axes
