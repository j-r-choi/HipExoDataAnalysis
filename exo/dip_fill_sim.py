"""Pre-processing filter that removes small mid-phase "double-bump" dips from
momentPredicted before it reaches any of the moment-mapping formulas in
moment_mapping_sim.py.

Some strides show a small secondary wiggle: the signal rises toward a peak,
dips down partway, rises again to a smaller secondary local peak, then
genuinely declines the rest of the way to zero (confirmed genuine/recurring
on HIP006.csv, not noise). filter_dip_fill() bridges exactly that kind of
recoverable dip while leaving genuine single-peaked strides and genuine
declines untouched.

This is a hybrid of two independently-designed candidates: a segment-and-merge
filter's genuine-peak/prominence gate, combined with a bounded-causal filter's
D-sample FIFO streaming delivery (segment-and-merge's own implementation
needs unbounded whole-phase lookahead, which isn't implementable as bounded
real-time streaming state).

Known, deliberate limitations -- tried alternatives and rejected them rather
than overlooked them (see defaults below for what was tried): dips whose
secondary bump recovers to just under `recover_frac` of the pre-dip peak are
left unfilled by design; a handful of dips in the noisy pre/post-walk tails
are also missed, either because their prominence never clears
`min_seg_height` or because `confirmed_peak` is pinned to a stale,
much-earlier reference peak the signal never climbs back to. Most, but not
all, dips fill -- the remaining gaps are accepted tradeoffs, not bugs to
chase further."""
import collections

import numpy as np


def filter_dip_fill(moment_predicted, istride, D=20, min_seg_height=0.025, recover_frac=0.85):
    """Explicit sample-by-sample streaming filter (bounded D-sample FIFO plus
    a handful of scalar peak-tracking state) -- matches this project's
    Python/C fidelity convention, not a vectorized lookahead trick.

    Mechanism: work in magnitude space, grouped by sign-of-moment only (a
    genuine flexion<->extension transition). `istride` is accepted as a
    parameter for caller/API parity but is NOT used internally -- grouping
    on istride too caused genuine dips that straddle an istride transition
    to be split and missed. Track a causal running_peak (max |m|
    so far this group) and a confirmed_peak (the last running_peak the
    signal has since fallen at least min_seg_height below -- a genuine,
    locally-prominent peak, not just "the biggest sample seen yet"). Every
    sample is held in the FIFO for D samples (or until its group ends, if
    sooner) before being released: if it was genuinely below recover_frac
    of its confirmed peak AND some sample within the D-sample window climbs
    back to recover_frac of that peak, release the confirmed_peak value
    instead (bridging the dip); otherwise release the sample's own raw
    value unchanged (a genuine decline, or nothing recovers in time).

    Defaults, each tuned against known problem cases: D=20 (200ms @ 100Hz)
    matches this project's existing delay_ms=200 / rise_delay_samples
    buffering convention; larger D (tested up to 200 samples) was rejected
    because it doesn't fix the remaining stale-confirmed_peak dips (a
    reference-peak problem, not a window-length one) and only adds
    real-time latency. min_seg_height=0.025 Nm/kg is a prominence floor,
    relaxed from an original 0.035 to rescue a few small dips in the noisy
    pre-walk period, at negligible cost elsewhere. recover_frac=0.85 is
    deliberately not relaxed further -- doing so was found to wrongly
    bridge a genuine decline-to-a-smaller-hump elsewhere in the data.
    """
    m = np.asarray(moment_predicted, dtype=float)
    istride = np.asarray(istride)  # accepted for caller/API parity; unused below
    n = len(m)
    out = np.zeros(n)
    if n == 0:
        return out

    sign_prev = None
    running_peak = 0.0
    confirmed_peak = 0.0
    pending = collections.deque()  # FIFO of [sample_idx, |m|, sign, confirmed_peak_at_arrival]

    def release_entry(idx, mag, sgn, peak_ref, future_vals):
        is_dip_candidate = (mag < recover_frac * peak_ref) and (peak_ref - mag) >= min_seg_height
        recovered = is_dip_candidate and len(future_vals) > 0 and max(future_vals) >= recover_frac * peak_ref
        released_mag = peak_ref if (is_dip_candidate and recovered) else mag
        out[idx] = released_mag if sgn >= 0 else -released_mag

    def flush_all(entries):
        vals = [e[1] for e in entries]
        for k, (idx, mag, sgn, peak_ref) in enumerate(entries):
            release_entry(idx, mag, sgn, peak_ref, vals[k + 1:k + 1 + D])

    for i in range(n):
        mi = m[i]
        mag = abs(mi) if np.isfinite(mi) else 0.0
        sgn = 1 if mi >= 0.0 else -1

        if sign_prev is None:
            sign_prev = sgn

        if sgn != sign_prev:
            # Genuine flexion<->extension sign flip: resolve everything
            # pending from the OLD phase using only that phase's own known
            # tail, then reset peak state. An istride change with no sign
            # flip does not reset anything -- resetting on every istride
            # tick used to split and miss dips straddling that transition.
            flush_all(list(pending))
            pending.clear()
            running_peak = 0.0
            confirmed_peak = 0.0
            sign_prev = sgn

        if mag > running_peak:
            running_peak = mag
        if running_peak - mag >= min_seg_height:
            confirmed_peak = running_peak
        peak_ref = confirmed_peak if confirmed_peak > 0.0 else running_peak

        pending.append([i, mag, sgn, peak_ref])

        while pending and (i - pending[0][0]) >= D:
            idx, mag0, sgn0, peak_ref0 = pending.popleft()
            future_vals = [e[1] for e in list(pending)[:D]]
            release_entry(idx, mag0, sgn0, peak_ref0, future_vals)

    flush_all(list(pending))
    return out
