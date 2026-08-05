"""Python port of reference/plot_imu_motor.m"""
import numpy as np
import matplotlib.pyplot as plt


def _lowpass_filtfilt(x, fs, cutoff_hz):
    """Zero-phase single-pole IIR low-pass (forward-backward pass, no scipy dependency)."""
    dt = 1.0 / fs
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    alpha = dt / (rc + dt)

    def _forward(sig):
        out = np.empty_like(sig)
        out[0] = sig[0]
        for i in range(1, len(sig)):
            out[i] = out[i - 1] + alpha * (sig[i] - out[i - 1])
        return out

    x = np.asarray(x, dtype=float)
    return _forward(_forward(x)[::-1])[::-1]


def plot_imu_motor(df, side, time_range=None):
    t = df["time"]
    fs = 1.0 / t.diff().median()
    motor_velocity = df[f"{side}_motorVelocity"]
    motor_velocity_filt = _lowpass_filtfilt(motor_velocity, fs, cutoff_hz=10.0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, df[f"{side}_motorAngle"], label="Motor")
    axes[0].plot(t, df[f"{side}_IMUroll"] + 100, label="IMU")  # +100 offset just to separate the traces visually
    axes[0].set_ylabel("Angle (deg)")
    axes[0].legend(loc="best")

    axes[1].plot(t, motor_velocity / 100, label="Motor", alpha=0.4)  # /100 scales into the same range as gyro
    axes[1].plot(t, motor_velocity_filt / 100, label="Motor (10 Hz LPF)")
    axes[1].set_ylabel("Velocity (scaled)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="best")

    if time_range:
        for ax in axes:
            ax.set_xlim(time_range)

    fig.suptitle(f"{side} — motor vs IMU")
    fig.tight_layout()
    return fig, axes
