"""Detect a stride-to-stride 'catch and release' hitch in an angle trace: a local
peak immediately followed (within `gap_max` seconds) by a second, smaller local
peak, with a notch between them.

Found while investigating why left_momentPredicted's '+' region fluctuates
(HIP055): left_motorAngle and left_IMUroll -- two independent sensors -- both
show this same notch on most strides, near-zero left_motorCurrent, so it's a
real kinematic hitch during swing, not sensor noise or a torque-command
artifact.
"""
import numpy as np
import pandas as pd


EMPTY_COLUMNS = ["t_peak1", "angle_peak1", "t_notch", "angle_notch",
                 "t_peak2", "angle_peak2", "depth_deg", "gap_s"]


def detect_stride_hitch(df, side, angle_col="motorAngle", peak_thresh=None, gap_max=0.3):
    """peak_thresh: only look for hitches near the top of the swing peak, i.e.
    above this value. Defaults to the midpoint of the column's own min/max range,
    since different columns (motorAngle: ~-30..55, IMUroll: ~-120..-55) don't
    share one absolute threshold.
    """
    t = df["time"].to_numpy()
    angle = df[f"{side}_{angle_col}"].to_numpy()
    if peak_thresh is None:
        peak_thresh = (np.nanmin(angle) + np.nanmax(angle)) / 2

    d = np.diff(angle)
    sign = np.sign(d)
    sign[sign == 0] = 1
    maxima = np.where(np.diff(sign) < 0)[0]
    minima = np.where(np.diff(sign) > 0)[0]
    maxima = maxima[angle[maxima] > peak_thresh]

    rows = []
    for i in range(len(maxima) - 1):
        if t[maxima[i + 1]] - t[maxima[i]] < gap_max:
            between = minima[(minima > maxima[i]) & (minima < maxima[i + 1])]
            if len(between):
                m = between[np.argmin(angle[between])]
                rows.append({
                    "t_peak1": t[maxima[i]], "angle_peak1": angle[maxima[i]],
                    "t_notch": t[m], "angle_notch": angle[m],
                    "t_peak2": t[maxima[i + 1]], "angle_peak2": angle[maxima[i + 1]],
                    "depth_deg": min(angle[maxima[i]], angle[maxima[i + 1]]) - angle[m],
                    "gap_s": t[maxima[i + 1]] - t[maxima[i]],
                })
    return pd.DataFrame(rows, columns=EMPTY_COLUMNS)
