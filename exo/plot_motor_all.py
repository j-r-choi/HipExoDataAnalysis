"""Python port of reference/plot_motor_all.m"""
import matplotlib.pyplot as plt


def plot_motor_all(df, side, time_range=None):
    t = df["time"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(t, df[f"{side}_motorAngle"])
    axes[0].set_ylabel("Angle (deg)")

    motor_rpm = df[f"{side}_motorVelocity"] * (32 / 190)  # motorVelocity (raw/32) -> motorRPM (raw/190), CSV has no raw column
    axes[1].plot(t, motor_rpm)
    axes[1].set_ylabel("RPM")

    axes[2].plot(t, df[f"{side}_motorCurrent"])
    axes[2].set_ylabel("Current (A)")
    axes[2].set_xlabel("Time (s)")

    if time_range:
        for ax in axes:
            ax.set_xlim(time_range)

    fig.tight_layout()
    return fig, axes
