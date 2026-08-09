"""
Anatomy of a measured-current spike: every relevant signal through one breach
of the 60 A limit, plus all breaches aligned on the breach sample.

Run:  python exo/make_spike_exhibits.py [--log HIP002] [--row N] [--limit 60]
      --log    HIPxxx.csv under data/<subject>/<date>/exo/ (or a path to one)
      --row    CSV line number of the breach to detail; default: the largest one
      --limit  breach threshold on |left_motorCurrent| [A]
The two PNGs are written next to the CSV, in the session's exo/ folder.

Left-leg sign convention: extension torque is NEGATIVE, and the current that
produces it is POSITIVE (main.c command is negated for the left leg).
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .data_paths import exo_csv
except ImportError:            # run as a plain script, not as part of the package
    from data_paths import exo_csv

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--log", default="HIP002", help="log name or CSV path (default: HIP002)")
ap.add_argument("--row", type=int, default=None, help="CSV row of the breach to detail")
ap.add_argument("--limit", type=float, default=60.0, help="breach threshold [A] (default: 60)")
args = ap.parse_args()

CSV = exo_csv(args.log)
OUT = CSV.parent
print(f"reading {CSV}")
df = pd.read_csv(CSV)
t = df.time.values
CUR_LIMIT, K_T, G, K_P, K_D = args.limit, 0.12, 9.0, 1.6, 1.2
KTG = K_T * G

def runs(m):
    m = np.asarray(m).astype(int); d = np.diff(np.r_[0, m, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))

frozen = (np.r_[False, np.diff(df.left_motorCommand.values) == 0] &
          np.r_[False, np.diff(df.right_motorCommand.values) == 0])
coast = (df.left_motorCommand.abs() < 1e-6) & (df.right_motorCommand.abs() < 1e-6)
in_coast = np.zeros(len(df), bool)
for a, b in runs(coast):
    if t[b] - t[a] > 2: in_coast[a:b + 1] = True
trips = [(a, b) for a, b in runs(frozen) if (b - a) >= 2 and not in_coast[a]]
blackout = np.zeros(len(df), bool)
for a, b in trips: blackout[a:b + 1] = True

# controller decomposition (left leg), only meaningful where the loop ran
ff   = (-df.left_torqueDesired / KTG).values
kp_t = (-K_P * (df.left_torqueDesired - df.left_torqueMeasured)).values
kd_t = (-K_D * df.left_motorVelocity / 1000.0).values
total = ff + kp_t + kd_t
live_mask = np.where(blackout, np.nan, 1.0)

BREACHES = np.where(df.left_motorCurrent.abs() > CUR_LIMIT)[0]
if not len(BREACHES):
    raise SystemExit(f"no |left_motorCurrent| > {CUR_LIMIT:g} A in {CSV.name} "
                     f"(peak {df.left_motorCurrent.abs().max():.1f} A) -- "
                     f"pass a lower --limit or another --log")
print(f"{len(BREACHES)} breach(es) > {CUR_LIMIT:g} A, peak {df.left_motorCurrent.abs().max():.1f} A"
      + (f", at CSV rows: {' '.join(str(k + 2) for k in BREACHES)}" if len(BREACHES) <= 25 else ""))
BLUE, RED, GREY, ORANGE, GREEN, PURPLE = "#1f6feb", "#d1242f", "#8c959f", "#bf5b04", "#1a7f37", "#8250df"

# ---------------------------------------------------------------- detail ---
i = (args.row - 2) if args.row is not None else -1
if i not in BREACHES:
    i = int(BREACHES[np.argmax(df.left_motorCurrent.abs().values[BREACHES])])
PRE, POST = 45, 55
lo, hi = i - PRE, min(len(df) - 1, i + POST)
s = slice(lo, hi)
tb = t[i]
trip = next(((a, b) for a, b in trips if a - 3 <= i <= b), None)

fig, ax = plt.subplots(5, 1, figsize=(11.5, 13), sharex=True,
                       gridspec_kw={"height_ratios": [1.25, 1.25, 1.25, 1, 1]})
fig.suptitle(f"Exhibit G - anatomy of a current spike ({CSV.stem}, CSV row {i+2}, t = {tb:.2f} s)",
             fontweight="bold", y=0.995)

# 1 - measured vs commanded current
ax[0].axhline(CUR_LIMIT, color=RED, ls="--", lw=1.3, label=f"eStop trip / clamp = {CUR_LIMIT:.0f} A")
ax[0].plot(t[s], df.left_motorCurrent[s], color=BLUE, lw=1.9, marker="o", ms=2.6,
           label="left_motorCurrent (measured)")
ax[0].plot(t[s], df.left_motorCommand[s], color=GREEN, lw=1.5, ls="-",
           label="left_motorCommand (logged, 100 Hz)")
ax[0].scatter([tb], [df.left_motorCurrent[i]], color=RED, s=90, zorder=6,
              label=f"breach: {df.left_motorCurrent[i]:.1f} A")
ax[0].set_ylabel("current [A]")
ax[0].legend(fontsize=8, loc="upper left", ncol=2)

# 2 - where the command comes from
ax[1].axhline(CUR_LIMIT, color=RED, ls="--", lw=1.3)
ax[1].plot(t[s], (ff * live_mask)[s], color=GREY, lw=1.7, label=r"feedforward  $\tau_d/(k_tG)$")
ax[1].plot(t[s], (kp_t * live_mask)[s], color=ORANGE, lw=1.7, label=r"$k_p$ term  $-k_p(\tau_d-\tau_m)$")
ax[1].plot(t[s], (kd_t * live_mask)[s], color=PURPLE, lw=1.3, label=r"$k_d$ term")
ax[1].plot(t[s], (total * live_mask)[s], color="k", lw=2.0, ls="--", label="sum (before clamp)")
ax[1].set_ylabel("current contribution [A]")
ax[1].legend(fontsize=8, loc="upper left", ncol=2)
ax[1].annotate(f"at the breach:\nff {ff[i-1]:.0f} A  +  $k_p$ {kp_t[i-1]:.0f} A  =  {total[i-1]:.0f} A",
               xy=(tb, max(total[i-1], CUR_LIMIT) * 0.62), fontsize=9, color=ORANGE, ha="left")

# 3 - torque desired vs measured, error shaded
ax[2].plot(t[s], df.left_torqueDesired[s], color=BLUE, lw=1.9, label="left_torqueDesired")
ax[2].plot(t[s], df.left_torqueMeasured[s], color=RED, lw=1.7, label="left_torqueMeasured")
ax[2].fill_between(t[s], df.left_torqueDesired[s], df.left_torqueMeasured[s],
                   color=ORANGE, alpha=0.22, label="tracking error (drives $k_p$)")
ax[2].axhline(0, color="k", lw=0.6)
ax[2].set_ylabel("torque [Nm]")
ax[2].legend(fontsize=8, loc="lower left")
ax[2].annotate(f"error = {df.left_torqueDesired[i-1]-df.left_torqueMeasured[i-1]:.1f} Nm",
               xy=(tb, df.left_torqueDesired[i-1]), fontsize=9, color=ORANGE)

# 4 - motor velocity
ax[3].plot(t[s], df.left_motorVelocity[s], color=PURPLE, lw=1.5, label="left_motorVelocity")
ax[3].axhline(0, color="k", lw=0.6)
ax[3].set_ylabel("motor speed [rpm]"); ax[3].legend(fontsize=8, loc="upper left")

# 5 - gait phase + the moment estimate driving the demand
ax[4].plot(t[s], df.left_gaitCycle[s], color=BLUE, lw=1.5, label="left_gaitCycle [%]")
ax[4].plot(t[s], df.right_gaitCycle[s], color=GREY, lw=1.2, label="right_gaitCycle [%]")
a4 = ax[4].twinx()
a4.plot(t[s], df.left_momentPredicted[s], color=GREEN, lw=1.5, label="left_momentPredicted")
a4.set_ylabel("moment estimate", color=GREEN)
ax[4].set_ylabel("gait cycle [%]"); ax[4].set_xlabel("time [s]")
h1, l1 = ax[4].get_legend_handles_labels(); h2, l2 = a4.get_legend_handles_labels()
ax[4].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

for x in ax:
    x.grid(alpha=0.3)
    x.axvline(tb, color=RED, lw=1.0, alpha=0.7)
    if trip: x.axvspan(t[trip[0]], t[min(trip[1], hi - 1)], color=RED, alpha=0.08)
fig.tight_layout()
fig.savefig(OUT / f"{CSV.stem}_exhibit_G_spike_detail.png", dpi=150)

print(f"EXHIBIT G - CSV row {i+2}, t={tb:.2f}s")
print(f"  {'row':>6} {'t':>7} | {'I meas':>7} {'cmd log':>8} | {'tqDes':>7} {'tqMeas':>7} {'err':>7} | "
      f"{'ff':>6} {'kp':>6} {'kd':>6} {'sum':>7} | {'spd':>7} {'gc%':>5}")
for j in range(i - 8, min(i + 6, len(df))):
    mk = "  <-- BREACH" if j == i else ("  (blackout)" if blackout[j] else "")
    print(f"  {j+2:6d} {t[j]:7.2f} | {df.left_motorCurrent[j]:7.2f} {df.left_motorCommand[j]:8.2f} | "
          f"{df.left_torqueDesired[j]:7.2f} {df.left_torqueMeasured[j]:7.2f} "
          f"{df.left_torqueDesired[j]-df.left_torqueMeasured[j]:7.2f} | "
          f"{ff[j]:6.1f} {kp_t[j]:6.1f} {kd_t[j]:6.1f} {total[j]:7.1f} | "
          f"{df.left_motorVelocity[j]:7.0f} {df.left_gaitCycle[j]:5.1f}{mk}")

# ------------------------------------------------------- all breaches aligned ---
W = 30
fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
rel = np.arange(-W, W + 1) * 10.0   # ms relative to the breach sample
for n, k in enumerate(BREACHES):
    a, b = k - W, k + W + 1
    if a < 0 or b > len(df): continue
    m = np.where(blackout[a:b], np.nan, 1.0)
    lab = f"row {k+2}"
    ax[0].plot(rel, df.left_motorCurrent[a:b], lw=1.2, alpha=0.85, label=lab)
    ax[1].plot(rel, (df.left_torqueDesired - df.left_torqueMeasured)[a:b] * m, lw=1.2, alpha=0.85)
    ax[2].plot(rel, (kp_t[a:b]) * m, lw=1.2, alpha=0.85)
ax[0].axhline(CUR_LIMIT, color=RED, ls="--", lw=1.4)
ax[0].set_title("measured current [A]", fontsize=10)
ax[1].set_title("tracking error $\\tau_d-\\tau_m$ [Nm]", fontsize=10)
ax[2].set_title("$k_p$ contribution [A]", fontsize=10)
for x in ax:
    x.axvline(0, color=RED, lw=1.0); x.grid(alpha=0.3); x.set_xlabel("ms relative to breach")
if len(BREACHES) <= 12:                 # one legend entry per breach is only readable when few
    ax[0].legend(fontsize=6, ncol=2)
fig.suptitle(f"Exhibit H - all {len(BREACHES)} breaches, aligned on the breach sample: "
             f"one mechanism, not {len(BREACHES)} accidents", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / f"{CSV.stem}_exhibit_H_all_spikes.png", dpi=150)
print(f"\nwrote {CSV.stem}_exhibit_G_spike_detail.png and "
      f"{CSV.stem}_exhibit_H_all_spikes.png in {OUT}")
