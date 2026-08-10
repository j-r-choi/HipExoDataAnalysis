"""
Every bench condition in one sheet: torqueDesired / torqueMeasured / current,
with both safety limits drawn, and TRIPPED vs OK marked.

Rate labels come from the operator's log book; the 10-90% rise time printed on
each panel is measured from the data and is the objective check on them.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
CUR, TQ_HI, TQ_LO = 60.0, 27.0, -48.0
BLUE, RED, GREEN, ORANGE, GREY = "#1f6feb", "#d1242f", "#1a7f37", "#bf5b04", "#8c959f"

def runs(m):
    m = np.asarray(m).astype(int); d = np.diff(np.r_[0, m, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))

DATA = {}
for fn in ["HIP006", "HIP007"]:
    df = pd.read_csv(os.path.join(HERE, fn + ".csv")); t = df.time.values
    frozen = (np.r_[False, np.diff(df.left_motorCommand.values) == 0] &
              np.r_[False, np.diff(df.right_motorCommand.values) == 0])
    trips = [(a, b) for a, b in runs(frozen) if (b - a) >= 2 and 0.5 < (t[b] - t[a]) < 2.0]
    DATA[fn] = (df, t, trips)

# (file, t_start, t_end, label amplitude, torqueRateLimit from log book)
COND = [
    # --- same amplitude, different rate: the decisive comparison ---
    ("HIP007",  20, 33,  "-45 Nm", "1.0"),
    ("HIP006", 171, 188, "-45 Nm", "1.0"),
    ("HIP007", 126, 140, "-45 Nm", "2.5"),
    # --- extension, rate sweep at other amplitudes ---
    ("HIP006", 126, 140, "-35 Nm", "1.0"),
    ("HIP006", 150, 164, "-40 Nm", "1.0"),
    ("HIP006",  40, 56,  "-30 Nm", "2.5"),
    # --- flexion ---
    ("HIP007", 455, 492, "+20 Nm", "up to 3"),
    ("HIP007", 344, 364, "+25 Nm", "1.5"),
    ("HIP007", 283, 296, "+30 Nm", "1.0"),
]

NC = 3
fig, axes = plt.subplots(6, NC, figsize=(17, 15),
                         gridspec_kw={"height_ratios": [1.15, 1] * 3})

for k, (fn, t0, t1, amp, trl) in enumerate(COND):
    df, t, trips = DATA[fn]
    r, c = (k // NC) * 2, k % NC
    axT, axI = axes[r, c], axes[r + 1, c]
    w = (t >= t0) & (t < t1)
    tw = t[w]
    ntrip = sum(1 for a, b in trips if t0 <= t[a] <= t1)

    # measured 10-90% rise time of |torqueMeasured| on the first burst
    td = df.left_torqueDesired.values
    rise = np.nan
    bs = [(a, b) for a, b in runs(np.abs(td) > 1.0) if (b - a) >= 5 and t0 <= t[a] <= t1]
    if bs:
        a, b = bs[0]
        seg = np.abs(df.left_torqueMeasured.values[a:b + 1]); pk = seg.max()
        if pk > 2:
            try:
                i1 = np.where(seg > 0.1 * pk)[0][0]; i2 = np.where(seg > 0.9 * pk)[0][0]
                rise = (i2 - i1) * 0.01
            except Exception: pass

    axT.plot(tw, df.left_torqueDesired[w], color="k", lw=1.3, ls=":", label="torqueDesired")
    axT.plot(tw, df.left_torqueMeasured[w], color=BLUE, lw=1.5, label="torqueMeasured L")
    axT.plot(tw, df.right_torqueMeasured[w], color=GREEN, lw=1.3, label="torqueMeasured R")
    axT.axhline(TQ_HI, color=RED, ls="--", lw=1.6)
    axT.axhline(TQ_LO, color=ORANGE, ls="--", lw=1.6)
    axT.set_ylim(-58, 40); axT.set_ylabel("torque [Nm]", fontsize=9)

    axI.plot(tw, df.left_motorCurrent[w], color=BLUE, lw=1.5, label="current L")
    axI.plot(tw, df.right_motorCurrent[w], color=GREEN, lw=1.3, label="current R")
    axI.axhline(CUR, color=RED, ls="--", lw=1.6); axI.axhline(-CUR, color=RED, ls="--", lw=1.6)
    axI.set_ylim(-80, 90); axI.set_ylabel("current [A]", fontsize=9)
    axI.set_xlabel("time [s]", fontsize=9)

    for a, b in trips:
        if t0 - 2 <= t[a] <= t1:
            for x in (axT, axI): x.axvspan(t[a], min(t[b], t1), color=RED, alpha=0.11)

    ok = ntrip == 0
    axT.set_title(f"{amp}   torqueRateLimit = {trl}\n"
                  f"{'OK - no trip' if ok else f'TRIPPED x{ntrip}'}"
                  f"   ({fn}, rise {rise*1000:.0f} ms)" if np.isfinite(rise) else
                  f"{amp}   torqueRateLimit = {trl}\n{'OK - no trip' if ok else f'TRIPPED x{ntrip}'}   ({fn})",
                  fontsize=10, fontweight="bold", color=(GREEN if ok else RED))
    for x in (axT, axI):
        x.grid(alpha=0.3); x.tick_params(labelsize=8)

leg = [Line2D([], [], color="k", ls=":", lw=1.3, label="torqueDesired"),
       Line2D([], [], color=BLUE, lw=1.5, label="left leg (torque / current)"),
       Line2D([], [], color=GREEN, lw=1.3, label="right leg (torque / current)"),
       Line2D([], [], color=RED, ls="--", lw=1.6, label="torqueUpperLimit +27 Nm  /  current limit ±60 A"),
       Line2D([], [], color=ORANGE, ls="--", lw=1.6, label="torqueLowerLimit −48 Nm"),
       Line2D([], [], color=RED, alpha=0.3, lw=8, label="eStop blackout (1 s)")]
fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=10, frameon=False,
           bbox_to_anchor=(0.5, -0.004))
fig.suptitle("Bench rate test - every condition.  Extension trips on CURRENT (60 A); "
             "flexion trips on TORQUE (+27 Nm) at only ~30 A.",
             fontweight="bold", fontsize=13.5)
fig.tight_layout(rect=[0, 0.035, 1, 0.985])
out = os.path.join(HERE, "all_runs_overview.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)

for fn, t0, t1, amp, trl in COND:
    df, t, trips = DATA[fn]
    w = (t >= t0) & (t < t1)
    n = sum(1 for a, b in trips if t0 <= t[a] <= t1)
    print(f"  {amp:>7}  trl={trl:>7}  {fn}  t={t0}-{t1}s  "
          f"peak|I| L{df.left_motorCurrent[w].abs().max():5.1f} R{df.right_motorCurrent[w].abs().max():5.1f} A  "
          f"tqMeas {df.left_torqueMeasured[w].min():6.1f}..{df.left_torqueMeasured[w].max():5.1f} Nm  "
          f"-> {'OK' if n==0 else f'TRIPPED x{n}'}")
