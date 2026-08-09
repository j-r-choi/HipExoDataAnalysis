"""Locating session data under `data/` (layout documented in README.md).

    data/<subject>/<date>/exo/HIPxxx.csv            exo log (data_convert.py output)
    data/<subject>/<date>/optimization/<subj>/...   HILO run output of that session
"""
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def exo_csv(log, data_root=DATA_ROOT):
    """Path of a HIPxxx.csv exo log.

    `log` is either a log name ("HIP002" or "HIP002.csv"), searched for in any
    `exo/` folder under data/, or a path to a CSV, used as given.
    """
    p = Path(log)
    if p.suffix.lower() == ".csv" and p.exists():
        return p

    hits = [c for c in sorted(data_root.rglob(f"{p.stem}.csv")) if c.parent.name == "exo"]
    if not hits:
        raise SystemExit(f"no {p.stem}.csv found under {data_root}")
    if len(hits) > 1:
        print(f"warning: {len(hits)} logs named {p.stem}.csv, using {hits[0]}")
    return hits[0]


def hilo_state(csv_path, data_root=DATA_ROOT):
    """HILO_state.json from the same session (same <subject>/<date>) as `csv_path`."""
    session = Path(csv_path).resolve().parent.parent      # .../<subject>/<date>
    hits = sorted((session / "optimization").glob("*/HILO_state.json"))
    if not hits:
        raise SystemExit(f"no optimization/*/HILO_state.json next to {csv_path}")
    if len(hits) > 1:
        print(f"warning: {len(hits)} HILO_state.json in {session}, using {hits[0]}")
    return hits[0]
