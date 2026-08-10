"""Which limit actually fired in the bench rate test? HIP006 / HIP007."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
CUR, TQ_HI, TQ_LO, TEMP = 60.0, 27.0, -48.0, 55.0

def runs(m):
    m = np.asarray(m).astype(int); d = np.diff(np.r_[0, m, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))

for fn in ["HIP006.csv", "HIP007.csv"]:
    df = pd.read_csv(os.path.join(HERE, fn)); t = df.time.values
    print("=" * 96); print(f"{fn}   {len(df)} rows, {t[-1]:.0f} s"); print("=" * 96)

    frozen = (np.r_[False, np.diff(df.left_motorCommand.values) == 0] &
              np.r_[False, np.diff(df.right_motorCommand.values) == 0])
    trips = [(a, b) for a, b in runs(frozen) if (b - a) >= 2 and (t[b] - t[a]) < 5]
    print(f"trips (command frozen >=30 ms): {len(trips)}")
    if not trips: continue
    d = np.array([t[b] - t[a] for a, b in trips])
    print(f"durations: median {np.median(d):.2f}s  min {d.min():.2f}  max {d.max():.2f}  "
          f"({(np.abs(d-1.0)<0.15).sum()}/{len(trips)} are the 1.0 s eStop signature)\n")

    # attribute each trip: worst value of each limit in the 60 ms before the freeze
    cause = []
    print(f"{'#':>3} {'t':>7} {'dur':>5} | {'peak|I|L':>8} {'peak|I|R':>8} | {'tqM_L':>15} {'tqM_R':>15} | {'temp':>5} | VERDICT")
    for n, (a, b) in enumerate(trips):
        w = df.iloc[max(0, a - 6):a]
        iL, iR = w.left_motorCurrent.abs().max(), w.right_motorCurrent.abs().max()
        tLmin, tLmax = w.left_torqueMeasured.min(), w.left_torqueMeasured.max()
        tRmin, tRmax = w.right_torqueMeasured.min(), w.right_torqueMeasured.max()
        tp = max(w.left_motorTemp.max(), w.right_motorTemp.max())
        hits = []
        if iL > CUR: hits.append("I_L>60")
        if iR > CUR: hits.append("I_R>60")
        if tLmax >= TQ_HI: hits.append("tqL>+27")
        if tRmax >= TQ_HI: hits.append("tqR>+27")
        if tLmin <= TQ_LO: hits.append("tqL<-48")
        if tRmin <= TQ_LO: hits.append("tqR<-48")
        if tp >= TEMP: hits.append("TEMP")
        v = ",".join(hits) if hits else "none visible @100Hz"
        cause.append(v)
        if n < 40:
            print(f"{n:3d} {t[a]:7.2f} {t[b]-t[a]:5.2f} | {iL:8.1f} {iR:8.1f} | "
                  f"{tLmin:6.1f}..{tLmax:<7.1f} {tRmin:6.1f}..{tRmax:<7.1f} | {tp:5.0f} | {v}")
    if len(trips) > 40: print(f"    ... {len(trips)-40} more")

    from collections import Counter
    print("\nCAUSE TALLY:", dict(Counter(cause)))
    # global extremes
    print(f"\nwhole-file extremes:")
    for s in ["left", "right"]:
        print(f"  {s:5s}: |I| max {df[f'{s}_motorCurrent'].abs().max():6.1f} A  | "
              f"tqMeas {df[f'{s}_torqueMeasured'].min():7.2f} .. {df[f'{s}_torqueMeasured'].max():6.2f} Nm | "
              f"tqDes {df[f'{s}_torqueDesired'].min():7.2f} .. {df[f'{s}_torqueDesired'].max():6.2f} | "
              f"temp {df[f'{s}_motorTemp'].max():.0f} C")
    print(f"  samples |I|>60: L {(df.left_motorCurrent.abs()>CUR).sum()}  R {(df.right_motorCurrent.abs()>CUR).sum()}")
    print(f"  samples tqMeas>=+27: L {(df.left_torqueMeasured>=TQ_HI).sum()}  R {(df.right_torqueMeasured>=TQ_HI).sum()}")
    print(f"  samples tqMeas<=-48: L {(df.left_torqueMeasured<=TQ_LO).sum()}  R {(df.right_torqueMeasured<=TQ_LO).sum()}")
    print()
