"""Joint moment (momentPredicted) vs desired torque (torqueDesired)."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, filtfilt, group_delay

from .torque_rate_limit_sim import simulate_rate_limit_torque


def _causal_lowpass_delay_ms(order, cutoff_hz, fs, at_hz=1.0):
    """Group delay (ms) of a causal Butterworth low-pass at `at_hz` (the gait
    fundamental is ~1-1.5 Hz) -- what this filter would actually cost on-device."""
    b, a = butter(order, cutoff_hz / (fs / 2), btype="low")
    _, gd = group_delay((b, a), w=[2 * np.pi * at_hz / fs], fs=fs)
    return gd[0] / fs * 1000


def plot_moment_torque(df, side, time_range=None, compare_filters=None, offline_reference=False,
                        show_rate_limit=False):
    """
    compare_filters: optional list of (order, cutoff_hz) Butterworth low-pass filters
        to overlay on torqueDesired, applied causally (lfilter) the same way a
        real-time firmware filter would run -- each labeled with its actual added
        group delay so smoothness and lag can be judged together, not separately.
    offline_reference: if True, also overlay a zero-phase (filtfilt) version of
        torqueDesired at the same cutoffs -- NOT realizable in real time (needs
        future samples), but shows the smoothness ceiling with zero added lag, as
        a benchmark for how much of the raggedness a causal filter can actually remove.
    show_rate_limit: if True, overlay the new rate_limit_torque() firmware behavior
        (simulated from the logged, pre-limit torqueDesired -- the rate-limited value
        itself isn't logged, see main.c) against the existing/recorded profile.
    """
    t = df["time"]
    fs = 1.0 / t.diff().median()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, df[f"{side}_momentPredicted"])
    axes[0].set_ylabel("Joint Moment (Nm/kg)")

    td = df[f"{side}_torqueDesired"]
    axes[1].plot(t, td, label="Existing (recorded, pre-limit)", color="0.3", linewidth=1)

    if show_rate_limit:
        axes[1].plot(t, simulate_rate_limit_torque(td), label="New (rate-limited, simulated)",
                     color="tab:red", linewidth=1.5)

    for order, cutoff_hz in compare_filters or []:
        b, a = butter(order, cutoff_hz / (fs / 2), btype="low")
        delay_ms = _causal_lowpass_delay_ms(order, cutoff_hz, fs)
        axes[1].plot(t, lfilter(b, a, td),
                     label=f"order={order} fc={cutoff_hz}Hz (causal, +{delay_ms:.0f}ms lag)")
        if offline_reference:
            axes[1].plot(t, filtfilt(b, a, td), "--",
                         label=f"order={order} fc={cutoff_hz}Hz (offline zero-phase, not realizable live)")

    axes[1].set_ylabel("Desired Torque (Nm)")
    axes[1].set_xlabel("Time (s)")
    if compare_filters or show_rate_limit:
        axes[1].legend(loc="best", fontsize=7)

    if time_range:
        for ax in axes:
            ax.set_xlim(time_range)

    fig.suptitle(f"{side} — joint moment & desired torque")
    fig.tight_layout()
    return fig, axes
