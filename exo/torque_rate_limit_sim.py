"""Python port of rate_limit_torque() in Core/Src/main.c."""
import numpy as np

TORQUE_RATE_LIMIT = 3.0  # [Nm] per 10ms sample -- must match main.c's torqueRateLimit

# The firmware calls rate_limit_torque() in a 500 Hz loop, holding `desired` fixed
# across 5 calls per 100 Hz torqueDesired update -- sub-step the same way here.
SUBSTEPS_PER_SAMPLE = 5


def simulate_rate_limit_torque(torque_desired, torque_rate_limit=TORQUE_RATE_LIMIT,
                                substeps=SUBSTEPS_PER_SAMPLE):
    max_delta_per_tick = torque_rate_limit / substeps

    torque_desired = np.asarray(torque_desired, dtype=float)
    out = np.empty_like(torque_desired)

    filtered = 0.0
    for i, desired in enumerate(torque_desired):
        if not np.isfinite(desired):
            desired = 0.0
        for _ in range(substeps):
            delta = desired - filtered
            delta = min(max(delta, -max_delta_per_tick), max_delta_per_tick)
            filtered += delta
        out[i] = filtered

    return out


def compute_rate_limit_metrics(raw, filtered, fs):
    """Quantify how much smoother `filtered` is than `raw`, and what it costs in lag.

    d(torque)/dt is where rate-limiting's effect actually shows up -- the value
    trace alone barely looks different at full-recording zoom.
    """
    raw = np.asarray(raw, dtype=float)
    filtered = np.asarray(filtered, dtype=float)
    d_raw = np.diff(raw) * fs
    d_filt = np.diff(filtered) * fs
    max_rate_raw = np.max(np.abs(d_raw))
    max_rate_filtered = np.max(np.abs(d_filt))

    return {
        "max_rate_raw_Nm_s": max_rate_raw,
        "max_rate_filtered_Nm_s": max_rate_filtered,
        "max_rate_reduction_pct": 100 * (1 - max_rate_filtered / max_rate_raw) if max_rate_raw else 0.0,
        "rms_rate_raw_Nm_s": np.sqrt(np.mean(d_raw ** 2)),
        "rms_rate_filtered_Nm_s": np.sqrt(np.mean(d_filt ** 2)),
        "mean_abs_lag_Nm": np.mean(np.abs(filtered - raw)),
        "max_abs_lag_Nm": np.max(np.abs(filtered - raw)),
    }
