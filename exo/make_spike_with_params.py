"""
Current spikes in an exo log annotated with the HILO optimisation parameters
that were live at that moment, and with the moment estimate that produced the
demand. The HILO_state.json is taken from the same session as the CSV, i.e.
data/<subject>/<date>/optimization/*/HILO_state.json.

Run:  python exo/make_spike_with_params.py [--log HIP002] [--weight 74]
                                           [--limit 60] [--first-cond 2]
The PNG is written next to the CSV, in the session's exo/ folder.

Condition <-> time mapping is INFERRED (HILO_state.json carries no timestamps):
measured moment->torque lag minus each condition's delayMsec is constant
(43 +/- 5 ms = the RISE_DELAY_SAMPLES queue) only for the assignment
segment k -> Pilot(k+1), i.e. Pilot2..Pilot7 in order.

Sign/unit note: momentPredicted is Nm/kg in the estimator's convention;
torqueDesired is Nm in the exoskeleton's convention, opposite sign
(moment_control.c:207 negates). Row 1 puts both on one Nm axis by plotting
    -momentPredicted * scale * weight
which is exactly the mapping compute_release_torque() applies when shape=1 and
the stride's running peak equals the N-stride peak average. Divergence from
torqueDesired is therefore the shape exponent plus peak-normalisation.

For the original Subject 2 log: mass 74 kg (confirmed by operator), and Pilot1
was not recorded to SD, so the logged segments were Pilot2..Pilot7 in order --
hence --first-cond 2. Check both against the session you are plotting.
"""
import argparse, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .data_paths import exo_csv, hilo_state
except ImportError:            # run as a plain script, not as part of the package
    from data_paths import exo_csv, hilo_state

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--log", default="HIP002", help="log name or CSV path (default: HIP002)")
ap.add_argument("--weight", type=float, default=74.0, help="subject mass [kg] (default: 74)")
ap.add_argument("--limit", type=float, default=60.0, help="breach threshold [A] (default: 60)")
ap.add_argument("--first-cond", type=int, default=2, dest="first_cond",
                help="pilot condition the first logged segment corresponds to (default: 2)")
args = ap.parse_args()

CUR, K_T, G, K_P, K_D = args.limit, 0.12, 9.0, 1.6, 1.2
KTG = K_T * G
WEIGHT_KG = args.weight

CSV = exo_csv(args.log)
STATE = hilo_state(CSV)
OUT = CSV.parent
print(f"reading {CSV}\n    and {STATE}")

st = json.load(open(STATE))
ped = np.array(st["pilot_exploring_data"])     # [pca_x, pca_y, scale, delayMsec, shape, emg]
H = st["torque_params_history"]

df = pd.read_csv(CSV); t = df.time.values

def runs(m):
    m = np.asarray(m).astype(int); d = np.diff(np.r_[0, m, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))

coast = (df.left_motorCommand.abs() < 1e-6) & (df.right_motorCommand.abs() < 1e-6)
gaps = [(a, b) for a, b in runs(coast) if t[b] - t[a] > 2]
segs = []; prev = 0
for a, b in gaps:
    if t[a] - t[prev] > 5: segs.append((prev, a))
    prev = b
if t[-1] - t[prev] > 5: segs.append((prev, len(df) - 1))
SEG2COND = {k: k + args.first_cond for k in range(len(segs))}   # seg0->Pilot<first_cond>, ...

def cond_at(i):
    for k, (a, b) in enumerate(segs):
        if a <= i <= b:
            c = SEG2COND[k]
            return c if c <= len(ped) else None    # segment beyond the logged conditions
    return None

ff   = (-df.left_torqueDesired / KTG).values
kp_t = (-K_P * (df.left_torqueDesired - df.left_torqueMeasured)).values
kd_t = (-K_D * df.left_motorVelocity / 1000.0).values
frozen = (np.r_[False, np.diff(df.left_motorCommand.values) == 0] &
          np.r_[False, np.diff(df.right_motorCommand.values) == 0])
blackout = np.zeros(len(df), bool)
for a, b in runs(frozen):
    if (b - a) >= 2 and not coast[a]: blackout[a:b + 1] = True
live = np.where(blackout, np.nan, 1.0)

BREACH = np.where(df.left_motorCurrent.abs() > CUR)[0]
# one representative spike per condition
chosen = {}
for i in BREACH:
    c = cond_at(i)
    if c is None: continue
    if c not in chosen or df.left_motorCurrent[i] > df.left_motorCurrent[chosen[c]]:
        chosen[c] = i
picks = sorted(chosen.items())
if not picks:
    raise SystemExit(f"no |left_motorCurrent| > {CUR:g} A inside a walking segment of "
                     f"{CSV.name} (peak {df.left_motorCurrent.abs().max():.1f} A) -- "
                     f"pass a lower --limit or another --log")

BLUE, RED, GREEN, ORANGE, GREY, PURPLE = "#1f6feb", "#d1242f", "#1a7f37", "#bf5b04", "#8c959f", "#8250df"
N = len(picks)
fig, ax = plt.subplots(3, N, figsize=(4.9 * N, 11.5), squeeze=False)

for col, (cond, i) in enumerate(picks):
    scale, delayMs, shape = ped[cond - 1, 2], ped[cond - 1, 3], ped[cond - 1, 4]
    hs, hd, hsh = H["scale"][cond - 1], H["delay"][cond - 1], H["shape"][cond - 1]
    lo, hi = i - 90, min(len(df) - 1, i + 40)
    s = slice(lo, hi); tw = t[s]
    shift = int(round(delayMs / 10.0))          # log is 100 Hz -> 10 ms/sample

    # ---- row 0: biological moment, scaled moment, and torque demand -- all in Nm ----
    a0 = ax[0, col]
    bio_nm = (-df.left_momentPredicted.values * WEIGHT_KG)            # Nm, no scale
    mom_nm = bio_nm * scale                                           # Nm, after scale
    a0.plot(tw, bio_nm[s], color=PURPLE, lw=1.5, alpha=0.65,
            label=f"biological moment x {WEIGHT_KG:.0f} kg (no scale)")
    a0.plot(tw, mom_nm[s], color=GREEN, lw=1.5, alpha=0.70,
            label=f"x scale {scale:.3f} (as received)")
    mshift = mom_nm[lo - shift:hi - shift] if lo - shift >= 0 else None
    if mshift is not None and len(mshift) == len(tw):
        a0.plot(tw, mshift, color=GREEN, lw=2.4, ls="--",
                label=f"same, delayed {delayMs:.0f} ms  (what the controller reads)")
    a0.plot(tw, df.left_torqueDesired[s], color=BLUE, lw=2.2, label="torqueDesired [Nm]")

    # mark the 50 ms rise-delay hold (RISE_DELAY_SAMPLES) on the rising edge
    tdw = df.left_torqueDesired.values[s]
    fl = np.r_[False, np.diff(tdw) == 0]
    k0 = None
    for aa, bb in [(x, y) for x, y in
                   zip(*[np.where(np.diff(np.r_[0, fl.astype(int), 0]) == d)[0] for d in (1, -1)])]:
        if (bb - aa) >= 3 and tdw[aa] < -3:
            k0 = (aa - 1, bb); break
    if k0:
        a0.plot(tw[k0[0]:k0[1] + 1], tdw[k0[0]:k0[1] + 1], color=RED, lw=4.0, alpha=0.55,
                solid_capstyle="butt", label="50 ms rise-delay hold")

    a0.axhline(0, color="k", lw=0.6)
    ymin = min(bio_nm[s].min(), df.left_torqueDesired[s].min())
    a0.set_ylabel("torque [Nm]"); a0.set_ylim(ymin * 1.15, max(bio_nm[s].max(), 12) * 1.25)
    a0.legend(fontsize=7.0, loc="lower left")
    a0.set_title(f"Pilot{cond}   scale {scale:.3f}   delay {delayMs:.0f} ms   shape {shape:.2f}\n"
                 f"(HILO {hs:.2f} / {hd:.2f} / {hsh:.2f})   EMG score {ped[cond-1,5]:.3f}",
                 fontsize=10.5, fontweight="bold")

    # ---- row 1: torque tracking ----
    a = ax[1, col]
    a.plot(tw, df.left_torqueDesired[s], color=BLUE, lw=1.5, ls=":", label="torqueDesired")
    a.plot(tw, df.left_torqueMeasured[s], color=RED, lw=1.8, label="torqueMeasured")
    a.fill_between(tw, df.left_torqueDesired[s], df.left_torqueMeasured[s],
                   color=ORANGE, alpha=0.22, label="tracking error")
    a.axhline(0, color="k", lw=0.6); a.set_ylabel("torque [Nm]"); a.set_ylim(-52, 30)
    a.legend(fontsize=7.5, loc="lower left")

    # ---- row 2: current + decomposition ----
    a = ax[2, col]
    a.axhline(CUR, color=RED, ls="--", lw=1.4, label="60 A limit")
    a.plot(tw, df.left_motorCurrent[s], color=BLUE, lw=1.8, label="motorCurrent (measured)")
    a.plot(tw, (ff * live)[s], color=GREY, lw=1.4, label="feedforward")
    a.plot(tw, ((ff + kp_t + kd_t) * live)[s], color="k", lw=1.7, ls="--", label="command (unclamped)")
    a.scatter([t[i]], [df.left_motorCurrent[i]], color=RED, s=80, zorder=6,
              label=f"breach {df.left_motorCurrent[i]:.0f} A")
    a.set_ylabel("current [A]"); a.set_xlabel("time [s]"); a.set_ylim(-20, 100)
    a.legend(fontsize=7.5, loc="upper left")

    for r in range(3):
        ax[r, col].grid(alpha=0.3); ax[r, col].axvline(t[i], color=RED, lw=1.0, alpha=0.6)
        for aa, bb in runs(blackout):
            if lo <= aa <= hi: ax[r, col].axvspan(t[aa], t[min(bb, hi)], color=RED, alpha=0.07)

fig.suptitle(f"{CSV.stem} - current spikes with the HILO parameters live at that moment  —  "
             "moment estimate, its delay, and the resulting torque demand",
             fontweight="bold", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.965])
out = OUT / f"{CSV.stem}_exhibit_I_spikes_with_params.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
for cond, i in picks:
    print(f"  Pilot{cond}: spike at t={t[i]:7.2f}s (row {i+2}), {df.left_motorCurrent[i]:5.1f} A | "
          f"scale {ped[cond-1,2]:.3f}  delay {ped[cond-1,3]:6.1f} ms  shape {ped[cond-1,4]:.2f}")
