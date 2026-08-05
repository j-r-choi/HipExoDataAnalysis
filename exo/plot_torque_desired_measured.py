"""Desired vs measured torque, one leg."""
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Load-cell calibration constants, mirroring filter_adc_values() (control_logic.c).
_GAIN = 375.53
_EXCITATION = 3.3
_SENSITIVITY_MV_PER_V = {"left": 2.1662, "right": 2.1674}
_MAX_LOAD_KGF = 18.1437
_KGF_TO_N = 9.80665
_MOMENT_ARM_M = 0.36


def _sensitivity_v_per_n(side):
    max_load_n = _MAX_LOAD_KGF * _KGF_TO_N
    return (_SENSITIVITY_MV_PER_V[side] / 1000.0) * _GAIN * _EXCITATION / max_load_n


def _recompute_measured_from_raw_load(df, side, filter_cutoff_hz=5.0):
    """torqueMeasured recomputed straight from the raw load-cell ADC count,
    skipping filter_adc_values()'s 4Hz Butterworth low-pass. meanVoltage (the
    per-recording zero-load calibration constant) isn't logged, so it's
    recovered from the fact that a low-pass filter preserves the DC/mean
    value: it must match the already-logged, filtered torqueMeasured's mean.

    Denoised with a zero-phase (filtfilt) low-pass instead of the firmware's
    causal one -- cuts noise with no added delay, since this only needs to
    look right on a plot, not run in real time.
    """
    raw_load = df[f"{side}_load"]
    measured = df[f"{side}_torqueMeasured"]
    sens = _sensitivity_v_per_n(side)

    output_volt = raw_load * 3.3 / 4095.0
    mean_voltage = output_volt.mean() - measured.mean() * sens / _MOMENT_ARM_M
    force = (output_volt - mean_voltage) / sens
    torque = force * _MOMENT_ARM_M

    if filter_cutoff_hz:
        fs = 1.0 / df["time"].diff().median()
        b, a = butter(2, filter_cutoff_hz / (fs / 2), btype="low")
        torque = filtfilt(b, a, torque)
    return torque


def plot_torque_desired_measured(df, side, time_range=None, measured2_cutoff_hz=5.0):
    t = df["time"]
    measured2 = _recompute_measured_from_raw_load(df, side, filter_cutoff_hz=measured2_cutoff_hz)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.axhline(0, color="gray", linewidth=1, zorder=0)
    ax.plot(t, df[f"{side}_torqueDesired"], label="Desired", linewidth=1)
    ax.plot(t, df[f"{side}_torqueMeasured"], label="Measured", linewidth=1, alpha=0.8)
    ax.plot(t, measured2, label="Measured2 (raw load cell, zero-phase filter)", linewidth=1, alpha=0.6)
    ax.set_ylabel("Torque (Nm)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="best")

    if time_range:
        ax.set_xlim(time_range)

    fig.suptitle(f"{side} — desired vs measured torque")
    fig.tight_layout()
    return fig, ax
