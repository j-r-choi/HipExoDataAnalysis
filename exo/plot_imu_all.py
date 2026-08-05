"""Python port of reference/plot_IMU_all.m"""
import matplotlib.pyplot as plt


def plot_imu_all(df, side, time_range=None):
    t = df["time"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(t, df[f"{side}_IMUroll"], label="Roll")
    axes[0].plot(t, df[f"{side}_IMUpitch"], label="Pitch")
    axes[0].plot(t, df[f"{side}_IMUyaw"], label="Yaw")
    axes[0].set_title("Euler Angle Data")
    axes[0].set_ylabel("Angles (deg)")
    axes[0].legend(loc="best")

    axes[1].plot(t, df[f"{side}_IMUaccX"], label="Acc X")
    axes[1].plot(t, df[f"{side}_IMUaccY"], label="Acc Y")
    axes[1].plot(t, df[f"{side}_IMUaccZ"], label="Acc Z")
    axes[1].set_title("Acceleration Data")
    axes[1].set_ylabel("Acceleration (m/s^2)")
    axes[1].legend(loc="best")

    axes[2].plot(t, df[f"{side}_IMUgyroX"], label="Gyro X")
    axes[2].plot(t, df[f"{side}_IMUgyroY"], label="Gyro Y")
    axes[2].plot(t, df[f"{side}_IMUgyroZ"], label="Gyro Z")
    axes[2].set_title("Gyroscope Data")
    axes[2].set_ylabel("Angular Velocity (deg/s)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="best")

    if time_range:
        for ax in axes:
            ax.set_xlim(time_range)

    fig.tight_layout()
    return fig, axes
