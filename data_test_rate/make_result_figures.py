"""Presentation figures for the bench rate test (HIP006 / HIP007)."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CUR, TQ_HI, TQ_LO = 60.0, 27.0, -48.0
BLUE, RED, GREY, ORANGE, GREEN = "#1f6feb", "#d1242f", "#8c959f", "#bf5b04", "#1a7f37"

def runs(m):
    m = np.asarray(m).astype(int); d = np.diff(np.r_[0, m, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))

def load(fn):
    df = pd.read_csv(os.path.join(HERE, fn)); t = df.time.values
    frozen = (np.r_[False, np.diff(df.left_motorCommand.values) == 0] &
              np.r_[False, np.diff(df.right_motorCommand.values) == 0])
    trips = [(a, b) for a, b in runs(frozen) if (b - a) >= 2 and 0.5 < (t[b] - t[a]) < 2.0]
    return df, t, trips

# measured ramp rate of the actual demand, per trip
def ramp_rate(df, a):
    w = df.left_torqueDesired.values[max(0, a - 60):a]
    if len(w) < 5: return np.nan
    d = np.abs(np.diff(w)) / 0.01
    return np.percentile(d[d > 5], 75) if (d > 5).any() else 0.0

rows = []
for fn in ["HIP006.csv", "HIP007.csv"]:
    df, t, trips = load(fn)
    for a, b in trips:
        w = df.iloc[max(0, a - 6):a]
        iL, iR = w.left_motorCurrent.abs().max(), w.right_motorCurrent.abs().max()
        tqmax = max(w.left_torqueMeasured.max(), w.right_torqueMeasured.max())
        tqmin = min(w.left_torqueMeasured.min(), w.right_torqueMeasured.min())
        amp = w.left_torqueDesired.values[np.argmax(np.abs(w.left_torqueDesired.values))]
        sign = "FLEXION (+)" if amp > 0 else "EXTENSION (-)"
        if tqmax >= TQ_HI - 0.3: verdict = "TORQUE +27 Nm"
        elif max(iL, iR) > CUR - 3: verdict = "CURRENT 60 A"
        elif amp > 0: verdict = "TORQUE +27 Nm (aliased)"
        else: verdict = "CURRENT 60 A (aliased)"
        rows.append(dict(file=fn[:7], t=t[a], amp=amp, dir=sign, rate=ramp_rate(df, a),
                         iPk=max(iL, iR), tqPk=tqmax if amp > 0 else tqmin, verdict=verdict))
R = pd.DataFrame(rows)
print(R.to_string(index=False, float_format=lambda x: f"{x:8.1f}"))
print("\nTALLY:", dict(R.verdict.value_counts()))
R.to_csv(os.path.join(HERE, "trip_summary.csv"), index=False)

# ---------------- FIG 1: the two mechanisms side by side --------------------
df7, t7, tr7 = load("HIP007.csv")
_negs = [x for x in tr7 if df7.left_torqueDesired[x[0] - 3] < 0]
# pick the one whose breach is actually visible at 100 Hz, else the highest peak
neg = max(_negs, key=lambda x: max(df7.left_motorCurrent.abs()[x[0]-6:x[0]].max(),
                                   df7.right_motorCurrent.abs()[x[0]-6:x[0]].max()))
pos = [x for x in tr7 if df7.left_torqueDesired[x[0] - 3] > 0][-1]

fig, ax = plt.subplots(2, 2, figsize=(13.5, 7.5))
for col, (a, b), title in [(0, neg, "EXTENSION (negative torque)"), (1, pos, "FLEXION (positive torque)")]:
    s = slice(a - 70, min(len(df7) - 1, b + 20))
    ax[0, col].plot(t7[s], df7.left_motorCurrent[s], color=BLUE, lw=1.6, label="left_motorCurrent")
    ax[0, col].plot(t7[s], df7.right_motorCurrent[s], color=GREEN, lw=1.4, label="right_motorCurrent")
    ax[0, col].axhline(CUR, color=RED, ls="--", lw=1.5, label="current limit 60 A")
    ax[0, col].axhline(-CUR, color=RED, ls="--", lw=1.5)
    ax[0, col].set_ylabel("current [A]"); ax[0, col].set_ylim(-75, 85)
    ax[0, col].set_title(f"{title}\n(HIP007, t={t7[a]:.0f}s)", fontsize=11, fontweight="bold")
    ax[1, col].plot(t7[s], df7.left_torqueDesired[s], color=BLUE, lw=1.6, ls=":", label="left_torqueDesired")
    ax[1, col].plot(t7[s], df7.left_torqueMeasured[s], color=BLUE, lw=1.8, label="left_torqueMeasured")
    ax[1, col].plot(t7[s], df7.right_torqueMeasured[s], color=GREEN, lw=1.4, label="right_torqueMeasured")
    ax[1, col].axhline(TQ_HI, color=RED, ls="--", lw=1.5, label="torque limit +27 Nm")
    ax[1, col].axhline(TQ_LO, color=ORANGE, ls="--", lw=1.5, label="torque limit -48 Nm")
    ax[1, col].set_ylabel("torque [Nm]"); ax[1, col].set_xlabel("time [s]"); ax[1, col].set_ylim(-55, 40)
    for r in range(2):
        ax[r, col].axvspan(t7[a], t7[b], color=RED, alpha=0.10); ax[r, col].grid(alpha=0.3)
        ax[r, col].legend(fontsize=7, loc="lower left")
ax[0, 0].annotate("current hits 60 A\n-> CURRENT trip", xy=(t7[neg[0]], 62), fontsize=9, color=RED, ha="right")
ax[1, 1].annotate("torque hits +27 Nm\n-> TORQUE trip\n(current only ~30 A)", xy=(t7[pos[0]], 29), fontsize=9,
                  color=RED, ha="right")
fig.suptitle("Bench rate test: TWO different limits are firing, not one", fontweight="bold", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig1_two_mechanisms.png"), dpi=150)

# ---------------- FIG 2: the asymmetric safety envelope ---------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].axhspan(TQ_LO, TQ_HI, color=GREEN, alpha=0.12)
ax[0].axhline(TQ_HI, color=RED, lw=2.5); ax[0].axhline(TQ_LO, color=ORANGE, lw=2.5)
ax[0].text(0.5, (TQ_HI + TQ_LO) / 2, "allowed torque band", ha="center", fontsize=11, color=GREEN)
ax[0].text(0.02, TQ_HI + 1.5, f"torqueUpperLimit = +{TQ_HI:.0f} Nm", color=RED, fontsize=10)
ax[0].text(0.02, TQ_LO - 4, f"torqueLowerLimit = {TQ_LO:.0f} Nm", color=ORANGE, fontsize=10)
ax[0].annotate("", xy=(0.75, TQ_HI), xytext=(0.75, 0), arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
ax[0].annotate("", xy=(0.9, TQ_LO), xytext=(0.9, 0), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=2))
ax[0].text(0.77, TQ_HI / 2, "27 Nm", color=RED, fontsize=10)
ax[0].text(0.92, TQ_LO / 2, "48 Nm", color=ORANGE, fontsize=10)
ax[0].set_xlim(0, 1); ax[0].set_ylim(-58, 38); ax[0].set_xticks([])
ax[0].set_ylabel("torqueMeasured [Nm]")
ax[0].set_title("The limits are asymmetric: flexion has 44% less room", fontsize=11)

for v, c, m in [("TORQUE +27 Nm", RED, "o"), ("TORQUE +27 Nm (aliased)", RED, "o"),
                ("CURRENT 60 A", BLUE, "s"), ("CURRENT 60 A (aliased)", BLUE, "s")]:
    q = R[R.verdict == v]
    if len(q): ax[1].scatter(q.amp.abs(), q.iPk, c=c, marker=m, s=90, alpha=0.85,
                             label=v if "aliased" not in v else None, edgecolor="k", lw=0.5)
ax[1].axhline(CUR, color=BLUE, ls="--", lw=1.5, label="current limit 60 A")
ax[1].set_xlabel("|torqueDesired| at the trip [Nm]"); ax[1].set_ylabel("peak current at the trip [A]")
ax[1].set_title("Flexion trips at ~30 A — current was never the issue there", fontsize=11)
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3); ax[1].set_ylim(0, 80)
fig.suptitle("Why flexion stopped sooner than extension", fontweight="bold", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig2_asymmetry.png"), dpi=150)
print("\nwrote fig1_two_mechanisms.png, fig2_asymmetry.png, trip_summary.csv")
