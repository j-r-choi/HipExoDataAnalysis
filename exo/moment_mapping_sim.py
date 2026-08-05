"""Python port of the moment_control.c torque-mapping formulas:
Linear (-scale*weight*m) and Peak-avg (peak-normalized, shape/PEAK_TRACK_NUM).

Note: this package used to also have _shape_compare/_scale_compare/_delay_compare
plotting helpers; they were deleted since they only demonstrated things visually
that turned out not to be needed. The simulate_* functions below are unaffected
and still exported, so that exploration can be recreated later if needed (e.g.
to re-verify peak-invariance before a C port)."""
from collections import deque

import numpy as np

PUSH_INTERVAL_MS = 10.0  # matches MOMENT_PUSH_INTERVAL_MS; CSV logging is also 100Hz
PEAK_TRACK_NUM = 3


def _delay(m, delay_ms, fs):
    m = np.asarray(m, dtype=float)
    delay_samples = int(round(delay_ms / (1000.0 / fs)))
    if delay_samples <= 0:
        return m.copy()
    return np.concatenate([np.zeros(delay_samples), m])[:len(m)]


def simulate_linear_moment_mapping(moment_predicted, scale=0.4, weight=60.0, delay_ms=200.0,
                                    fs=100.0, torque_upper_limit=27.0, torque_lower_limit=-48.0):
    m = _delay(moment_predicted, delay_ms, fs)
    torque = -scale * weight * m
    return np.clip(torque, torque_lower_limit, torque_upper_limit)


def soft_cap(ratio, cap=1.0, sharpness=6.0):
    """Smooth approximation of min(ratio, cap) -- close to identity below cap,
    asymptotes to cap above it, with no hard corner. Higher sharpness -> closer
    to a hard clamp (in the limit sharpness->inf, converges to min(ratio, cap))."""
    ratio = np.asarray(ratio, dtype=float)
    return ratio / (1.0 + (ratio / cap) ** sharpness) ** (1.0 / sharpness)


def simulate_peak_avg_moment_mapping(moment_predicted, istride, scale=0.4, weight=60.0, shape=1.0,
                                      delay_ms=200.0, fs=100.0, torque_upper_limit=27.0,
                                      torque_lower_limit=-48.0, peak_track_num=PEAK_TRACK_NUM,
                                      cap_mode="hard", soft_sharpness=6.0, ceiling_mode="matched"):
    """cap_mode: 'hard' clamps ratio=|m|/peak at 1; 'soft' uses soft_cap() instead.
    ceiling_mode: 'matched' re-multiplies by peak after shaping, so shape=1 recovers
        the Linear formula exactly (but the achieved peak still drifts slightly
        with shape, since ratio rarely lands exactly at 1). 'isolated' skips the
        peak re-multiply -- scale*weight is always the ceiling regardless of shape,
        but shape=1 no longer matches the Linear formula's amplitude."""
    m = _delay(moment_predicted, delay_ms, fs)
    istride = np.asarray(istride)
    n = len(m)
    torque = np.zeros(n)

    flexion_peaks = [0.0] * peak_track_num
    extension_peaks = [0.0] * peak_track_num
    current_flexion_peak = 0.0
    current_extension_peak = 0.0
    istride_prev = istride[0] if n else 0
    stride_count = 0

    for i in range(n):
        mi = m[i]

        if istride[i] != istride_prev:
            flexion_peaks = flexion_peaks[1:] + [current_flexion_peak]
            extension_peaks = extension_peaks[1:] + [current_extension_peak]
            current_flexion_peak = 0.0
            current_extension_peak = 0.0
            istride_prev = istride[i]
            stride_count = min(stride_count + 1, peak_track_num)

        if np.isfinite(mi):
            if mi >= 0.0:
                current_flexion_peak = max(current_flexion_peak, mi)
            else:
                current_extension_peak = max(current_extension_peak, abs(mi))

        if stride_count == 0:
            continue

        # Progressive warm-up: average only the strides banked so far (1, 2, or 3).
        peak_flexion = sum(flexion_peaks[-stride_count:]) / stride_count
        peak_extension = sum(extension_peaks[-stride_count:]) / stride_count
        peak = peak_flexion if mi >= 0.0 else peak_extension

        if not np.isfinite(peak) or peak < 1e-3:
            continue

        # Normalize to a 0-1(ish) fraction of peak, then shape the curve.
        raw_ratio = abs(mi) / peak
        ratio = min(raw_ratio, 1.0) if cap_mode == "hard" else float(soft_cap(raw_ratio, sharpness=soft_sharpness))
        shaped_ratio = ratio ** shape
        reshaped = peak * shaped_ratio if ceiling_mode == "matched" else shaped_ratio
        t = (-reshaped if mi >= 0.0 else reshaped) * scale * weight
        torque[i] = min(max(t, torque_lower_limit), torque_upper_limit)

    return torque


def simulate_peak_avg_moment_mapping_istride_buffered(moment_predicted, istride, scale=0.4, weight=60.0,
                                                       shape=1.0, delay_ms=200.0, fs=100.0,
                                                       torque_upper_limit=27.0, torque_lower_limit=-48.0,
                                                       peak_track_num=PEAK_TRACK_NUM, cap_mode="hard",
                                                       soft_sharpness=6.0, ceiling_mode="matched"):
    """Same math as simulate_peak_avg_moment_mapping, but ring-buffers istride instead
    of the moment signal. The non-buffered version delays only the moment (via _delay),
    leaving istride real-time, so m and istride disagree about "now" by delay_samples
    right around stride boundaries -- ring-buffering istride too keeps them paired
    correctly. Explicit sample-by-sample streaming loop with real ring buffers
    (collections.deque), matching how real embedded C firmware would implement this."""
    moment_predicted = np.asarray(moment_predicted, dtype=float)
    istride = np.asarray(istride)
    n = len(moment_predicted)
    torque = np.zeros(n)
    delay_samples = int(round(delay_ms / (1000.0 / fs)))

    moment_ring = deque([0.0] * delay_samples, maxlen=delay_samples) if delay_samples > 0 else None
    istride_ring = deque([istride[0] if n else 0] * delay_samples, maxlen=delay_samples) if delay_samples > 0 else None

    flexion_peaks = [0.0] * peak_track_num
    extension_peaks = [0.0] * peak_track_num
    current_flexion_peak = 0.0
    current_extension_peak = 0.0
    istride_prev = None
    stride_count = 0

    for i in range(n):
        if delay_samples > 0:
            mi = moment_ring[0]
            istride_i = istride_ring[0]
            moment_ring.append(moment_predicted[i])
            istride_ring.append(istride[i])
        else:
            mi = moment_predicted[i]
            istride_i = istride[i]

        if istride_prev is None:
            istride_prev = istride_i

        if istride_i != istride_prev:
            flexion_peaks = flexion_peaks[1:] + [current_flexion_peak]
            extension_peaks = extension_peaks[1:] + [current_extension_peak]
            current_flexion_peak = 0.0
            current_extension_peak = 0.0
            istride_prev = istride_i
            stride_count = min(stride_count + 1, peak_track_num)

        if np.isfinite(mi):
            if mi >= 0.0:
                current_flexion_peak = max(current_flexion_peak, mi)
            else:
                current_extension_peak = max(current_extension_peak, abs(mi))

        if stride_count == 0:
            continue

        peak_flexion = sum(flexion_peaks[-stride_count:]) / stride_count
        peak_extension = sum(extension_peaks[-stride_count:]) / stride_count
        peak = peak_flexion if mi >= 0.0 else peak_extension

        if not np.isfinite(peak) or peak < 1e-3:
            continue

        raw_ratio = abs(mi) / peak
        ratio = min(raw_ratio, 1.0) if cap_mode == "hard" else float(soft_cap(raw_ratio, sharpness=soft_sharpness))
        shaped_ratio = ratio ** shape
        reshaped = peak * shaped_ratio if ceiling_mode == "matched" else shaped_ratio
        t = (-reshaped if mi >= 0.0 else reshaped) * scale * weight
        torque[i] = min(max(t, torque_lower_limit), torque_upper_limit)

    return torque


def simulate_local_peak_moment_mapping(moment_predicted, istride, scale=0.4, weight=60.0, shape=1.0,
                                        delay_ms=200.0, fs=100.0, torque_upper_limit=27.0,
                                        torque_lower_limit=-48.0, peak_track_num=PEAK_TRACK_NUM,
                                        cap_mode="hard", soft_sharpness=6.0, ceiling_mode="matched"):
    """Like simulate_peak_avg_moment_mapping, but shape reshapes only the decline from each
    stride's own running peak, never the climb -- so a stride's own peak torque is
    shape-invariant even when that stride is weaker than history (its ratio vs peak_avg
    never reaches 1). local_ratio = |m| / running_peak_since_stride_start is exactly 1.0
    at every new-record sample, so shape (which only exponentiates local_ratio) can never
    move that peak. peak_avg keeps its original role -- across-stride smoothing/adaptation
    and the cap -- now applied to the running peak via stride_ratio."""
    m = _delay(moment_predicted, delay_ms, fs)
    istride = np.asarray(istride)
    n = len(m)
    torque = np.zeros(n)

    flexion_peaks = [0.0] * peak_track_num
    extension_peaks = [0.0] * peak_track_num
    current_flexion_peak = 0.0
    current_extension_peak = 0.0
    istride_prev = istride[0] if n else 0
    stride_count = 0

    for i in range(n):
        mi = m[i]

        if istride[i] != istride_prev:
            flexion_peaks = flexion_peaks[1:] + [current_flexion_peak]
            extension_peaks = extension_peaks[1:] + [current_extension_peak]
            current_flexion_peak = 0.0
            current_extension_peak = 0.0
            istride_prev = istride[i]
            stride_count = min(stride_count + 1, peak_track_num)

        if np.isfinite(mi):
            if mi >= 0.0:
                current_flexion_peak = max(current_flexion_peak, mi)
            else:
                current_extension_peak = max(current_extension_peak, abs(mi))

        if stride_count == 0:
            continue

        # Same progressive peak_avg as simulate_peak_avg_moment_mapping -- unchanged role.
        peak_flexion = sum(flexion_peaks[-stride_count:]) / stride_count
        peak_extension = sum(extension_peaks[-stride_count:]) / stride_count
        peak_avg = peak_flexion if mi >= 0.0 else peak_extension

        if not np.isfinite(peak_avg) or peak_avg < 1e-3:
            continue

        # Causal running peak since this stride started (post-update above).
        running_peak = current_flexion_peak if mi >= 0.0 else current_extension_peak
        if running_peak < 1e-3:
            continue

        # 1.0 exactly at every new-record sample, regardless of shape.
        local_ratio = abs(mi) / running_peak
        raw_stride_ratio = running_peak / peak_avg
        stride_ratio = min(raw_stride_ratio, 1.0) if cap_mode == "hard" else float(soft_cap(raw_stride_ratio, sharpness=soft_sharpness))

        shaped = stride_ratio * local_ratio ** shape
        reshaped = peak_avg * shaped if ceiling_mode == "matched" else shaped
        t = (-reshaped if mi >= 0.0 else reshaped) * scale * weight
        torque[i] = min(max(t, torque_lower_limit), torque_upper_limit)

    return torque


def simulate_local_peak_moment_mapping_istride_buffered(moment_predicted, istride, scale=0.4, weight=60.0,
                                                         shape=1.0, delay_ms=200.0, fs=100.0,
                                                         torque_upper_limit=27.0, torque_lower_limit=-48.0,
                                                         peak_track_num=PEAK_TRACK_NUM, cap_mode="hard",
                                                         soft_sharpness=6.0, ceiling_mode="matched"):
    """Same math as simulate_local_peak_moment_mapping, but ring-buffers istride instead
    of the moment signal. The non-buffered version delays only the moment (via _delay),
    leaving istride real-time, so m and istride disagree about "now" by delay_samples
    right around stride boundaries -- ring-buffering istride too keeps them paired
    correctly. Explicit sample-by-sample streaming loop with real ring buffers
    (collections.deque), matching how real embedded C firmware would implement this."""
    moment_predicted = np.asarray(moment_predicted, dtype=float)
    istride = np.asarray(istride)
    n = len(moment_predicted)
    torque = np.zeros(n)
    delay_samples = int(round(delay_ms / (1000.0 / fs)))

    moment_ring = deque([0.0] * delay_samples, maxlen=delay_samples) if delay_samples > 0 else None
    istride_ring = deque([istride[0] if n else 0] * delay_samples, maxlen=delay_samples) if delay_samples > 0 else None

    flexion_peaks = [0.0] * peak_track_num
    extension_peaks = [0.0] * peak_track_num
    current_flexion_peak = 0.0
    current_extension_peak = 0.0
    istride_prev = None
    stride_count = 0

    for i in range(n):
        if delay_samples > 0:
            mi = moment_ring[0]
            istride_i = istride_ring[0]
            moment_ring.append(moment_predicted[i])
            istride_ring.append(istride[i])
        else:
            mi = moment_predicted[i]
            istride_i = istride[i]

        if istride_prev is None:
            istride_prev = istride_i

        if istride_i != istride_prev:
            flexion_peaks = flexion_peaks[1:] + [current_flexion_peak]
            extension_peaks = extension_peaks[1:] + [current_extension_peak]
            current_flexion_peak = 0.0
            current_extension_peak = 0.0
            istride_prev = istride_i
            stride_count = min(stride_count + 1, peak_track_num)

        if np.isfinite(mi):
            if mi >= 0.0:
                current_flexion_peak = max(current_flexion_peak, mi)
            else:
                current_extension_peak = max(current_extension_peak, abs(mi))

        if stride_count == 0:
            continue

        peak_flexion = sum(flexion_peaks[-stride_count:]) / stride_count
        peak_extension = sum(extension_peaks[-stride_count:]) / stride_count
        peak_avg = peak_flexion if mi >= 0.0 else peak_extension

        if not np.isfinite(peak_avg) or peak_avg < 1e-3:
            continue

        running_peak = current_flexion_peak if mi >= 0.0 else current_extension_peak
        if running_peak < 1e-3:
            continue

        local_ratio = abs(mi) / running_peak
        raw_stride_ratio = running_peak / peak_avg
        stride_ratio = min(raw_stride_ratio, 1.0) if cap_mode == "hard" else float(soft_cap(raw_stride_ratio, sharpness=soft_sharpness))

        shaped = stride_ratio * local_ratio ** shape
        reshaped = peak_avg * shaped if ceiling_mode == "matched" else shaped
        t = (-reshaped if mi >= 0.0 else reshaped) * scale * weight
        torque[i] = min(max(t, torque_lower_limit), torque_upper_limit)

    return torque


def shift_for_delivery(torque, delay_samples):
    """simulate_delayed_rise_moment_mapping writes each sample's shaped result back
    to its OWN original index (for easy same-timestamp shape-vs-shape comparison) --
    that's timing-dishonest about when the torque would actually be deliverable. This
    shifts it forward by delay_samples (zero-padding the front) to show what a real
    output-delay ring buffer would actually be commanding at each point in time."""
    torque = np.asarray(torque, dtype=float)
    if delay_samples <= 0:
        return torque.copy()
    return np.concatenate([np.zeros(delay_samples), torque])[:len(torque)]


# +150ms on top of delay_ms. Chosen from real onset-to-peak timing (well under
# typical flexion/extension rise durations) and a false-record-rate elbow: the rate
# drops sharply out to ~15 samples, then flattens -- little gained by going higher.
RISE_DELAY_SAMPLES = 15


def simulate_delayed_rise_moment_mapping(moment_predicted, istride, scale=0.4, weight=60.0, shape=1.0,
                                          delay_ms=200.0, fs=100.0, torque_upper_limit=27.0,
                                          torque_lower_limit=-48.0, peak_track_num=PEAK_TRACK_NUM,
                                          cap_mode="hard", soft_sharpness=6.0, ceiling_mode="matched",
                                          rise_delay_samples=RISE_DELAY_SAMPLES):
    """Like simulate_local_peak_moment_mapping, but shape also reshapes the RISE, at
    the cost of a small fixed extra output delay: release of sample i is held until
    either rise_delay_samples more samples arrive or its stride ends (whichever first),
    using the BEST peak known by then -- so a later, larger sample can retroactively
    make an earlier point's local_ratio honestly <1, letting shape reshape the climb too.

    Peak invariance stays exact for any rise_delay_samples >= 0: the running peak used
    to release a sample is always bounded between that sample's own value and the
    stride's true max, so the stride's actual peak sample is always released at
    local_ratio==1 regardless of delay length. One residual effect: a sample that looks
    like a record within its own window but is beaten only after the window closes is
    also released at local_ratio==1, so more than one sample per stride/phase can hit
    the ceiling -- a visibility/latency trade-off, not a peak-invariance violation.

    Zero added delay cannot do this for any causal function, not just power-laws: two
    strides with identical history up to sample i but diverging only afterward are
    indistinguishable to a causal algorithm at i, so exact peak invariance forces
    shape invariance at every candidate-record sample -- i.e. the whole climb. Bounded
    delay is the only way to turn "is this the peak?" from an unobservable predicate
    into an observable one at record time.

    Known gaps if this is ported to C: (a) real firmware needs an explicit
    output-delay ring buffer -- this function writes each result back to its own
    sample's index for easy shape-vs-shape comparison (use shift_for_delivery to see
    the honest delivery-time-shifted torque); (b) the last ~rise_delay_samples of a
    run never get released since the FIFO never empties -- firmware needs a
    flush-on-stop path; (c) unlike the current C map_moment(), this doesn't guard
    !isfinite(m) up front.
    """
    m = _delay(moment_predicted, delay_ms, fs)
    istride = np.asarray(istride)
    n = len(m)
    torque = np.zeros(n)

    flexion_peaks = [0.0] * peak_track_num
    extension_peaks = [0.0] * peak_track_num
    current_flexion_peak = 0.0
    current_extension_peak = 0.0
    istride_prev = istride[0] if n else 0
    stride_count = 0

    # FIFO of not-yet-released samples: [orig_index, mi, istride_of_mi, peak_avg,
    # release_at_i, frozen_peak]. frozen_peak stays None until either its own stride
    # ends (frozen early, using the just-banked peak) or its timeout is reached (then
    # release() reads the live current_*_peak instead of freezing anything).
    pending = []

    def release(entry):
        idx, mi, _, peak_avg, _, frozen_peak = entry
        running_peak = frozen_peak if frozen_peak is not None else (
            current_flexion_peak if mi >= 0.0 else current_extension_peak)
        if running_peak < 1e-3:
            return

        local_ratio = abs(mi) / running_peak
        raw_stride_ratio = running_peak / peak_avg
        stride_ratio = min(raw_stride_ratio, 1.0) if cap_mode == "hard" else float(soft_cap(raw_stride_ratio, sharpness=soft_sharpness))

        shaped = stride_ratio * local_ratio ** shape
        reshaped = peak_avg * shaped if ceiling_mode == "matched" else shaped
        t = (-reshaped if mi >= 0.0 else reshaped) * scale * weight
        torque[idx] = min(max(t, torque_lower_limit), torque_upper_limit)

    for i in range(n):
        mi = m[i]

        if istride[i] != istride_prev:
            # Stride just ended: its final peaks are now known, so any of its samples
            # still waiting in the queue can be released now instead of at timeout.
            for entry in pending:
                if entry[5] is None and entry[2] == istride_prev:
                    entry[5] = current_flexion_peak if entry[1] >= 0.0 else current_extension_peak

            flexion_peaks = flexion_peaks[1:] + [current_flexion_peak]
            extension_peaks = extension_peaks[1:] + [current_extension_peak]
            current_flexion_peak = 0.0
            current_extension_peak = 0.0
            istride_prev = istride[i]
            stride_count = min(stride_count + 1, peak_track_num)

        if np.isfinite(mi):
            if mi >= 0.0:
                current_flexion_peak = max(current_flexion_peak, mi)
            else:
                current_extension_peak = max(current_extension_peak, abs(mi))

        if stride_count > 0:
            peak_flexion = sum(flexion_peaks[-stride_count:]) / stride_count
            peak_extension = sum(extension_peaks[-stride_count:]) / stride_count
            peak_avg = peak_flexion if mi >= 0.0 else peak_extension

            if np.isfinite(peak_avg) and peak_avg >= 1e-3:
                pending.append([i, mi, istride[i], peak_avg, i + rise_delay_samples, None])

        # Release everything due: frozen (stride just ended) or timed out. Both are
        # always at the front first, since the queue is in strictly increasing index
        # order and both conditions are monotonic along it.
        while pending and (pending[0][5] is not None or pending[0][4] <= i):
            release(pending.pop(0))

    return torque


def simulate_delayed_rise_moment_mapping_istride_buffered(moment_predicted, istride, scale=0.4, weight=60.0,
                                                            shape=1.0, delay_ms=200.0, fs=100.0,
                                                            torque_upper_limit=27.0, torque_lower_limit=-48.0,
                                                            peak_track_num=PEAK_TRACK_NUM, cap_mode="hard",
                                                            soft_sharpness=6.0, ceiling_mode="matched",
                                                            rise_delay_samples=RISE_DELAY_SAMPLES):
    """Same math as simulate_delayed_rise_moment_mapping, but ring-buffers istride
    instead of the moment signal, so a delayed-but-just-arrived moment estimate still
    gets paired with the istride/phase that was live when it was measured -- otherwise
    the two disagree by delay_samples right at stride boundaries and a sample gets
    bucketed into the wrong stride/phase. Explicit sample-by-sample streaming loop
    with real ring buffers (collections.deque), matching how real embedded C firmware
    would implement this. Verified numerically identical to delay_ms=0 shifted by
    delay_samples.
    """
    moment_predicted = np.asarray(moment_predicted, dtype=float)
    istride = np.asarray(istride)
    n = len(moment_predicted)
    torque = np.zeros(n)
    delay_samples = int(round(delay_ms / (1000.0 / fs)))

    # Two fixed-depth ring buffers: moment_ring mirrors _delay() as an explicit FIFO;
    # istride_ring is the new one this variant adds, so the delay_samples-old istride
    # pairs with the just-arrived (already-stale) moment estimate.
    moment_ring = deque([0.0] * delay_samples, maxlen=delay_samples) if delay_samples > 0 else None
    istride_ring = deque([istride[0] if n else 0] * delay_samples, maxlen=delay_samples) if delay_samples > 0 else None

    flexion_peaks = [0.0] * peak_track_num
    extension_peaks = [0.0] * peak_track_num
    current_flexion_peak = 0.0
    current_extension_peak = 0.0
    istride_prev = None
    stride_count = 0

    pending = []

    def release(entry):
        idx, mi, _, peak_avg, _, frozen_peak = entry
        running_peak = frozen_peak if frozen_peak is not None else (
            current_flexion_peak if mi >= 0.0 else current_extension_peak)
        if running_peak < 1e-3:
            return

        local_ratio = abs(mi) / running_peak
        raw_stride_ratio = running_peak / peak_avg
        stride_ratio = min(raw_stride_ratio, 1.0) if cap_mode == "hard" else float(soft_cap(raw_stride_ratio, sharpness=soft_sharpness))

        shaped = stride_ratio * local_ratio ** shape
        reshaped = peak_avg * shaped if ceiling_mode == "matched" else shaped
        t = (-reshaped if mi >= 0.0 else reshaped) * scale * weight
        torque[idx] = min(max(t, torque_lower_limit), torque_upper_limit)

    for i in range(n):
        if delay_samples > 0:
            # Read each ring's oldest (delay_samples-ago) entry before pushing this
            # tick's true value in -- same push/pop order a real circular buffer uses.
            mi = moment_ring[0]
            istride_i = istride_ring[0]
            moment_ring.append(moment_predicted[i])
            istride_ring.append(istride[i])
        else:
            mi = moment_predicted[i]
            istride_i = istride[i]

        if istride_prev is None:
            istride_prev = istride_i

        if istride_i != istride_prev:
            # Stride just ended: its final peaks are now known, so any of its samples
            # still waiting in the queue can be released now instead of at timeout.
            for entry in pending:
                if entry[5] is None and entry[2] == istride_prev:
                    entry[5] = current_flexion_peak if entry[1] >= 0.0 else current_extension_peak

            flexion_peaks = flexion_peaks[1:] + [current_flexion_peak]
            extension_peaks = extension_peaks[1:] + [current_extension_peak]
            current_flexion_peak = 0.0
            current_extension_peak = 0.0
            istride_prev = istride_i
            stride_count = min(stride_count + 1, peak_track_num)

        if np.isfinite(mi):
            if mi >= 0.0:
                current_flexion_peak = max(current_flexion_peak, mi)
            else:
                current_extension_peak = max(current_extension_peak, abs(mi))

        if stride_count > 0:
            peak_flexion = sum(flexion_peaks[-stride_count:]) / stride_count
            peak_extension = sum(extension_peaks[-stride_count:]) / stride_count
            peak_avg = peak_flexion if mi >= 0.0 else peak_extension

            if np.isfinite(peak_avg) and peak_avg >= 1e-3:
                pending.append([i, mi, istride_i, peak_avg, i + rise_delay_samples, None])

        while pending and (pending[0][5] is not None or pending[0][4] <= i):
            release(pending.pop(0))

    return torque
