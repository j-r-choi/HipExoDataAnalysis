#!/usr/bin/env python3
"""
Python port of data_convert.m (+ importFileExo.m / label_hip_exo.m) for the
Hip Exo Controller. Reads a HIP*.BIN log, labels the fields, and writes a CSV
with the same columns as the MATLAB pipeline.

Binary record format (reverse-engineered from HIP043.BIN, matches the
dataNum == 51 branch of label_hip_exo.m):
    uint16 0xAAAA sync | N x float32 (little-endian) fields | uint16 0xBBBB sync
Record length (and therefore N) is auto-detected from the sync markers so
logs from other firmware versions (34/35/37/39/41/47/49/55 fields) still
parse correctly.

Usage:
    python3 data_convert.py                     # convert every *.BIN under data/ that has no .csv yet
    python3 data_convert.py path/to/HIP043.BIN
    python3 data_convert.py path/to/HIP043.BIN --out HIP043.csv

Requires: numpy, pandas
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SYNC_START = b"\xaa\xaa"
SYNC_END = b"\xbb\xbb"
HEADER_BYTES = 2
FOOTER_BYTES = 2
FIELD_BYTES = 4  # float32


def import_file_exo(file_path):
    """Parse a HIP*.BIN log into a dict of data1..dataN float64 arrays."""
    raw = Path(file_path).read_bytes()

    first = raw.find(SYNC_START)
    if first == -1:
        raise ValueError(f"no {SYNC_START.hex()} sync marker found in {file_path}")

    boundary = raw.find(SYNC_END + SYNC_START, first)
    if boundary == -1:
        raise ValueError("could not determine record length (no footer+header boundary found)")
    record_len = (boundary + FOOTER_BYTES) - first
    n_fields = (record_len - HEADER_BYTES - FOOTER_BYTES) // FIELD_BYTES
    fmt = f"<{n_fields}f"

    n_records = (len(raw) - first) // record_len
    columns = [[] for _ in range(n_fields)]
    n_bad = 0
    for i in range(n_records):
        offset = first + i * record_len
        rec = raw[offset:offset + record_len]
        if (rec[:HEADER_BYTES] != SYNC_START
                or rec[-FOOTER_BYTES:] != SYNC_END):
            n_bad += 1
            continue
        fields = struct.unpack_from(fmt, rec, HEADER_BYTES)
        for col, val in zip(columns, fields):
            col.append(val)

    if n_bad:
        print(f"warning: skipped {n_bad}/{n_records} out-of-sync record(s)", file=sys.stderr)

    return {f"data{i + 1}": np.array(col, dtype=np.float64) for i, col in enumerate(columns)}


def _reconstruct_time(raw_ms):
    """timeMsecCounter resets to 0 if Init_System() re-runs mid-recording (BLE
    "Initialize System" command) -- rebase each reset onto a continuous
    monotonic timeline instead of letting it go negative and overlap the
    previous segment's time range. Returns (time_seconds, new_session_mask).
    """
    t = np.asarray(raw_ms, dtype=np.float64) / 1000.0
    dt = np.diff(t)
    sample_interval = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.01
    correction = np.where(dt < 0, -dt + sample_interval, 0.0)
    offset = np.concatenate(([0.0], np.cumsum(correction)))
    new_session = np.concatenate(([False], dt < 0))

    if new_session.sum():
        print(f"warning: {int(new_session.sum())} mid-recording time reset(s) detected "
              f"(BLE Initialize System re-run) -- time axis rebased to stay monotonic",
              file=sys.stderr)

    return t - t[0] + offset, new_session


def label_hip_exo(data_raw):
    """Python port of label_hip_exo.m — maps dataRaw['dataN'] arrays to named signals."""
    n = len(data_raw)
    d = lambda i: data_raw[f"data{i}"]
    data = {"right": {}, "left": {}}

    time, new_session = _reconstruct_time(d(1))
    data["new_session"] = new_session

    if n == 34:
        data["right"]["Time"] = time
        data["right"]["motorAngle"] = d(2)
        data["right"]["motorVelocity"] = d(3) / 32
        data["right"]["motorRPM"] = d(3) / 190
        data["right"]["motorCurrent"] = d(4)
        data["right"]["IMUroll"] = d(5)
        data["right"]["IMUpitch"] = d(6)
        data["right"]["IMUyaw"] = d(7)
        data["right"]["IMUaccX"] = d(8)
        data["right"]["IMUaccY"] = d(9)
        data["right"]["IMUaccZ"] = d(10)
        data["right"]["IMUgyroX"] = d(11)
        data["right"]["IMUgyroY"] = d(12)
        data["right"]["IMUgyroZ"] = d(13)
        data["right"]["gaitCycle"] = d(14)
        data["right"]["torqueDesired"] = d(15)
        data["right"]["torqueMeasured"] = d(16)
        data["right"]["motorCommand"] = d(17)
        data["left"]["Time"] = time
        data["left"]["motorAngle"] = d(18)
        data["left"]["motorVelocity"] = d(19) / 32
        data["left"]["motorRPM"] = d(19) / 190
        data["left"]["motorCurrent"] = d(20)
        data["left"]["IMUroll"] = d(21)
        data["left"]["IMUpitch"] = d(22)
        data["left"]["IMUyaw"] = d(23)
        data["left"]["IMUaccX"] = d(24)
        data["left"]["IMUaccY"] = d(25)
        data["left"]["IMUaccZ"] = d(26)
        data["left"]["IMUgyroX"] = d(27)
        data["left"]["IMUgyroY"] = d(28)
        data["left"]["IMUgyroZ"] = d(29)
        data["left"]["gaitCycle"] = d(30)
        data["left"]["torqueDesired"] = d(31)
        data["left"]["torqueMeasured"] = d(32)
        data["left"]["motorCommand"] = d(33)
        data["trailingLimbAngle"] = d(34)

    elif n == 35:
        data["right"]["Time"] = time
        data["right"]["motorAngle"] = d(2)
        data["right"]["motorVelocity"] = d(3) / 32
        data["right"]["motorRPM"] = d(3) / 190
        data["right"]["motorCurrent"] = d(4)
        data["right"]["IMUroll"] = d(5)
        data["right"]["IMUpitch"] = d(6)
        data["right"]["IMUyaw"] = d(7)
        data["right"]["IMUaccX"] = d(8)
        data["right"]["IMUaccY"] = d(9)
        data["right"]["IMUaccZ"] = d(10)
        data["right"]["IMUgyroX"] = d(11)
        data["right"]["IMUgyroY"] = d(12)
        data["right"]["IMUgyroZ"] = d(13)
        data["right"]["gaitCycle"] = d(14)
        data["right"]["torqueDesired"] = d(15)
        data["right"]["torqueMeasured"] = d(16)
        data["right"]["motorCommand"] = d(17)
        data["right"]["trailingLimbAngle"] = d(18)
        data["left"]["Time"] = time
        data["left"]["motorAngle"] = d(19)
        data["left"]["motorVelocity"] = d(20) / 32
        data["left"]["motorRPM"] = d(20) / 190
        data["left"]["motorCurrent"] = d(21)
        data["left"]["IMUroll"] = d(22)
        data["left"]["IMUpitch"] = d(23)
        data["left"]["IMUyaw"] = d(24)
        data["left"]["IMUaccX"] = d(25)
        data["left"]["IMUaccY"] = d(26)
        data["left"]["IMUaccZ"] = d(27)
        data["left"]["IMUgyroX"] = d(28)
        data["left"]["IMUgyroY"] = d(29)
        data["left"]["IMUgyroZ"] = d(30)
        data["left"]["gaitCycle"] = d(31)
        data["left"]["torqueDesired"] = d(32)
        data["left"]["torqueMeasured"] = d(33)
        data["left"]["motorCommand"] = d(34)
        data["left"]["trailingLimbAngle"] = d(35)

    elif n == 37:
        data["right"]["Time"] = time
        data["right"]["motorAngle"] = d(2)
        data["right"]["motorVelocity"] = d(3) / 32
        data["right"]["motorRPM"] = d(3) / 190
        data["right"]["motorCurrent"] = d(4)
        data["right"]["load"] = d(5)
        data["right"]["IMUroll"] = d(6)
        data["right"]["IMUpitch"] = d(7)
        data["right"]["IMUyaw"] = d(8)
        data["right"]["IMUaccX"] = d(9)
        data["right"]["IMUaccY"] = d(10)
        data["right"]["IMUaccZ"] = d(11)
        data["right"]["IMUgyroX"] = d(12)
        data["right"]["IMUgyroY"] = d(13)
        data["right"]["IMUgyroZ"] = d(14)
        data["right"]["gaitCycle"] = d(15)
        data["right"]["torqueDesired"] = d(16)
        data["right"]["torqueMeasured"] = d(17)
        data["right"]["motorCommand"] = d(18)
        data["right"]["trailingLimbAngle"] = d(19)
        data["left"]["Time"] = time
        data["left"]["motorAngle"] = d(20)
        data["left"]["motorVelocity"] = d(21) / 32
        data["left"]["motorRPM"] = d(21) / 190
        data["left"]["motorCurrent"] = d(22)
        data["left"]["load"] = d(23)
        data["left"]["IMUroll"] = d(24)
        data["left"]["IMUpitch"] = d(25)
        data["left"]["IMUyaw"] = d(26)
        data["left"]["IMUaccX"] = d(27)
        data["left"]["IMUaccY"] = d(28)
        data["left"]["IMUaccZ"] = d(29)
        data["left"]["IMUgyroX"] = d(30)
        data["left"]["IMUgyroY"] = d(31)
        data["left"]["IMUgyroZ"] = d(32)
        data["left"]["gaitCycle"] = d(33)
        data["left"]["torqueDesired"] = d(34)
        data["left"]["torqueMeasured"] = d(35)
        data["left"]["motorCommand"] = d(36)
        data["left"]["trailingLimbAngle"] = d(37)

    elif n == 39:
        data["right"]["Time"] = time
        data["right"]["motorAngle"] = d(2)
        data["right"]["motorVelocity"] = d(3) / 32
        data["right"]["motorRPM"] = d(3) / 190
        data["right"]["motorCurrent"] = d(4)
        data["right"]["motorTemp"] = d(5)
        data["right"]["load"] = d(6)
        data["right"]["IMUroll"] = d(7)
        data["right"]["IMUpitch"] = d(8)
        data["right"]["IMUyaw"] = d(9)
        data["right"]["IMUaccX"] = d(10)
        data["right"]["IMUaccY"] = d(11)
        data["right"]["IMUaccZ"] = d(12)
        data["right"]["IMUgyroX"] = d(13)
        data["right"]["IMUgyroY"] = d(14)
        data["right"]["IMUgyroZ"] = d(15)
        data["right"]["gaitCycle"] = d(16)
        data["right"]["torqueDesired"] = d(17)
        data["right"]["torqueMeasured"] = d(18)
        data["right"]["motorCommand"] = d(19)
        data["right"]["trailingLimbAngle"] = d(20)
        data["left"]["Time"] = time
        data["left"]["motorAngle"] = d(21)
        data["left"]["motorVelocity"] = d(22) / 32
        data["left"]["motorRPM"] = d(22) / 190
        data["left"]["motorCurrent"] = d(23)
        data["left"]["motorTemp"] = d(24)
        data["left"]["load"] = d(25)
        data["left"]["IMUroll"] = d(26)
        data["left"]["IMUpitch"] = d(27)
        data["left"]["IMUyaw"] = d(28)
        data["left"]["IMUaccX"] = d(29)
        data["left"]["IMUaccY"] = d(30)
        data["left"]["IMUaccZ"] = d(31)
        data["left"]["IMUgyroX"] = d(32)
        data["left"]["IMUgyroY"] = d(33)
        data["left"]["IMUgyroZ"] = d(34)
        data["left"]["gaitCycle"] = d(35)
        data["left"]["torqueDesired"] = d(36)
        data["left"]["torqueMeasured"] = d(37)
        data["left"]["motorCommand"] = d(38)
        data["left"]["trailingLimbAngle"] = d(39)

    elif n == 41:
        data["right"]["Time"] = time
        data["right"]["motorAngle"] = d(2)
        data["right"]["motorVelocity"] = d(3) / 32
        data["right"]["motorRPM"] = d(3) / 190
        data["right"]["motorCurrent"] = d(4)
        data["right"]["motorTemp"] = d(5)
        data["right"]["load"] = d(6)
        data["pelvis"] = {
            "IMUroll": d(7), "IMUpitch": d(8), "IMUyaw": d(9),
            "IMUaccX": d(10), "IMUaccY": d(11), "IMUaccZ": d(12),
            "IMUgyroX": d(13), "IMUgyroY": d(14), "IMUgyroZ": d(15),
        }
        data["right"]["gaitCycle"] = d(16)
        data["right"]["torqueDesired"] = d(17)
        data["right"]["torqueMeasured"] = d(18)
        data["right"]["motorCommand"] = d(19)
        data["right"]["trailingLimbAngle"] = d(20)
        data["right"]["momentPredicted"] = d(21)
        data["left"]["Time"] = time
        data["left"]["motorAngle"] = d(22)
        data["left"]["motorVelocity"] = d(23) / 32
        data["left"]["motorRPM"] = d(23) / 190
        data["left"]["motorCurrent"] = d(24)
        data["left"]["motorTemp"] = d(25)
        data["left"]["load"] = d(26)
        data["left"]["IMUroll"] = d(27)
        data["left"]["IMUpitch"] = d(28)
        data["left"]["IMUyaw"] = d(29)
        data["left"]["IMUaccX"] = d(30)
        data["left"]["IMUaccY"] = d(31)
        data["left"]["IMUaccZ"] = d(32)
        data["left"]["IMUgyroX"] = d(33)
        data["left"]["IMUgyroY"] = d(34)
        data["left"]["IMUgyroZ"] = d(35)
        data["left"]["gaitCycle"] = d(36)
        data["left"]["torqueDesired"] = d(37)
        data["left"]["torqueMeasured"] = d(38)
        data["left"]["motorCommand"] = d(39)
        data["left"]["trailingLimbAngle"] = d(40)
        data["left"]["momentPredicted"] = d(41)

    elif n == 47:
        data["right"]["Time"] = time
        data["right"]["motorAngle"] = d(2)
        data["right"]["motorVelocity"] = d(3) / 32
        data["right"]["motorRPM"] = d(3) / 190
        data["right"]["motorCurrent"] = d(4)
        data["right"]["load"] = d(5)
        data["right"]["IMUroll"] = d(6)
        data["right"]["IMUpitch"] = d(7)
        data["right"]["IMUyaw"] = d(8)
        data["right"]["IMUaccX"] = d(9)
        data["right"]["IMUaccY"] = d(10)
        data["right"]["IMUaccZ"] = d(11)
        data["right"]["IMUgyroX"] = d(12)
        data["right"]["IMUgyroY"] = d(13)
        data["right"]["IMUgyroZ"] = d(14)
        data["right"]["gaitCycle"] = d(15)
        data["right"]["torqueDesired"] = d(16)
        data["right"]["torqueMeasured"] = d(17)
        data["right"]["motorCommand"] = d(18)
        data["right"]["trailingLimbAngle"] = d(19)
        data["right"]["ankleCommand"] = d(20)
        data["right"]["ankleAngle"] = d(21)
        data["right"]["ankleLoad"] = d(22)
        data["right"]["ankleDisplacement"] = d(23)
        data["right"]["ankleActuation"] = d(24)
        data["left"]["Time"] = time
        data["left"]["motorAngle"] = d(25)
        data["left"]["motorVelocity"] = d(26) / 32
        data["left"]["motorRPM"] = d(26) / 190
        data["left"]["motorCurrent"] = d(27)
        data["left"]["load"] = d(28)
        data["left"]["IMUroll"] = d(29)
        data["left"]["IMUpitch"] = d(30)
        data["left"]["IMUyaw"] = d(31)
        data["left"]["IMUaccX"] = d(32)
        data["left"]["IMUaccY"] = d(33)
        data["left"]["IMUaccZ"] = d(34)
        data["left"]["IMUgyroX"] = d(35)
        data["left"]["IMUgyroY"] = d(36)
        data["left"]["IMUgyroZ"] = d(37)
        data["left"]["gaitCycle"] = d(38)
        data["left"]["torqueDesired"] = d(39)
        data["left"]["torqueMeasured"] = d(40)
        data["left"]["motorCommand"] = d(41)
        data["left"]["trailingLimbAngle"] = d(42)
        data["left"]["ankleCommand"] = d(43)
        data["left"]["ankleAngle"] = d(44)
        data["left"]["ankleLoad"] = d(45)
        data["left"]["ankleDisplacement"] = d(46)
        data["left"]["ankleActuation"] = d(47)

    elif n == 49:
        data["right"]["Time"] = time
        data["right"]["motorAngle"] = d(2)
        data["right"]["motorVelocity"] = d(3) / 32
        data["right"]["motorRPM"] = d(3) / 190
        data["right"]["motorCurrent"] = d(4)
        data["right"]["motorTemp"] = d(5)
        data["right"]["load"] = d(6)
        data["right"]["IMUroll"] = d(7)
        data["right"]["IMUpitch"] = d(8)
        data["right"]["IMUyaw"] = d(9)
        data["right"]["IMUaccX"] = d(10)
        data["right"]["IMUaccY"] = d(11)
        data["right"]["IMUaccZ"] = d(12)
        data["right"]["IMUgyroX"] = d(13)
        data["right"]["IMUgyroY"] = d(14)
        data["right"]["IMUgyroZ"] = d(15)
        data["right"]["gaitCycle"] = d(16)
        data["right"]["torqueDesired"] = d(17)
        data["right"]["torqueMeasured"] = d(18)
        data["right"]["motorCommand"] = d(19)
        data["right"]["trailingLimbAngle"] = d(20)
        data["right"]["ankleCommand"] = d(21)
        data["right"]["ankleAngle"] = d(22)
        data["right"]["ankleLoad"] = d(23)
        data["right"]["ankleDisplacement"] = d(24)
        data["right"]["ankleActuation"] = d(25)
        data["left"]["Time"] = time
        data["left"]["motorAngle"] = d(26)
        data["left"]["motorVelocity"] = d(27) / 32
        data["left"]["motorRPM"] = d(27) / 190
        data["left"]["motorCurrent"] = d(28)
        data["left"]["motorTemp"] = d(29)
        data["left"]["load"] = d(30)
        data["left"]["IMUroll"] = d(31)
        data["left"]["IMUpitch"] = d(32)
        data["left"]["IMUyaw"] = d(33)
        data["left"]["IMUaccX"] = d(34)
        data["left"]["IMUaccY"] = d(35)
        data["left"]["IMUaccZ"] = d(36)
        data["left"]["IMUgyroX"] = d(37)
        data["left"]["IMUgyroY"] = d(38)
        data["left"]["IMUgyroZ"] = d(39)
        data["left"]["gaitCycle"] = d(40)
        data["left"]["torqueDesired"] = d(41)
        data["left"]["torqueMeasured"] = d(42)
        data["left"]["motorCommand"] = d(43)
        data["left"]["trailingLimbAngle"] = d(44)
        data["left"]["ankleCommand"] = d(45)
        data["left"]["ankleAngle"] = d(46)
        data["left"]["ankleLoad"] = d(47)
        data["left"]["ankleDisplacement"] = d(48)
        data["left"]["ankleActuation"] = d(49)

    elif n in (51, 55):
        data["right"]["Time"] = time
        data["right"]["motorAngle"] = d(2)
        data["right"]["motorVelocity"] = d(3) / 32
        data["right"]["motorRPM"] = d(3) / 190
        data["right"]["motorCurrent"] = d(4)
        data["right"]["motorTemp"] = d(5)
        data["right"]["load"] = d(6)
        data["right"]["IMUroll"] = d(7)
        data["right"]["IMUpitch"] = d(8)
        data["right"]["IMUyaw"] = d(9)
        data["right"]["IMUaccX"] = d(10)
        data["right"]["IMUaccY"] = d(11)
        data["right"]["IMUaccZ"] = d(12)
        data["right"]["IMUgyroX"] = d(13)
        data["right"]["IMUgyroY"] = d(14)
        data["right"]["IMUgyroZ"] = d(15)
        data["right"]["gaitCycle"] = d(16)
        data["right"]["torqueDesired"] = d(17)
        data["right"]["torqueMeasured"] = d(18)
        data["right"]["motorCommand"] = d(19)
        data["right"]["trailingLimbAngle"] = d(20)
        data["right"]["momentPredicted"] = d(21)
        data["left"]["Time"] = time
        data["left"]["motorAngle"] = d(22)
        data["left"]["motorVelocity"] = d(23) / 32
        data["left"]["motorRPM"] = d(23) / 190
        data["left"]["motorCurrent"] = d(24)
        data["left"]["motorTemp"] = d(25)
        data["left"]["load"] = d(26)
        data["left"]["IMUroll"] = d(27)
        data["left"]["IMUpitch"] = d(28)
        data["left"]["IMUyaw"] = d(29)
        data["left"]["IMUaccX"] = d(30)
        data["left"]["IMUaccY"] = d(31)
        data["left"]["IMUaccZ"] = d(32)
        data["left"]["IMUgyroX"] = d(33)
        data["left"]["IMUgyroY"] = d(34)
        data["left"]["IMUgyroZ"] = d(35)
        data["left"]["gaitCycle"] = d(36)
        data["left"]["torqueDesired"] = d(37)
        data["left"]["torqueMeasured"] = d(38)
        data["left"]["motorCommand"] = d(39)
        data["left"]["trailingLimbAngle"] = d(40)
        data["left"]["momentPredicted"] = d(41)
        data["pelvis"] = {
            "load": d(42),
            "IMUroll": d(43), "IMUpitch": d(44), "IMUyaw": d(45),
            "IMUaccX": d(46), "IMUaccY": d(47), "IMUaccZ": d(48),
            "IMUgyroX": d(49), "IMUgyroY": d(50), "IMUgyroZ": d(51),
        }
        if n == 55:
            # Thigh-IMU CAN RX-rate diagnostic (appended to TxData; updated ~1 Hz)
            data["left"]["thighRxHz"] = d(52)
            data["right"]["thighRxHz"] = d(53)
            data["left"]["thighRxMaxGapMs"] = d(54)
            data["right"]["thighRxMaxGapMs"] = d(55)

    else:
        raise ValueError(f"unsupported record layout: {n} fields")

    return data


def find_unconverted_bins(data_root):
    """All *.BIN files anywhere under data_root that don't have a same-named .csv sibling yet."""
    return [b for b in sorted(data_root.rglob("*.BIN")) if not b.with_suffix(".csv").exists()]


def convert_bin_to_csv(bin_path, out_path=None):
    data_raw = import_file_exo(bin_path)
    data = label_hip_exo(data_raw)

    time = data["right"]["Time"]
    r, l, p = data["right"], data["left"], data["pelvis"]

    df = pd.DataFrame({
        "time": time,
        "new_session": data["new_session"],
        "right_motorAngle": r["motorAngle"], "right_motorVelocity": r["motorVelocity"],
        "right_motorCurrent": r["motorCurrent"], "right_motorTemp": r["motorTemp"],
        "right_load": r["load"],
        "right_IMUroll": r["IMUroll"], "right_IMUpitch": r["IMUpitch"], "right_IMUyaw": r["IMUyaw"],
        "right_IMUaccX": r["IMUaccX"], "right_IMUaccY": r["IMUaccY"], "right_IMUaccZ": r["IMUaccZ"],
        "right_IMUgyroX": r["IMUgyroX"], "right_IMUgyroY": r["IMUgyroY"], "right_IMUgyroZ": r["IMUgyroZ"],
        "right_gaitCycle": r["gaitCycle"], "right_torqueDesired": r["torqueDesired"],
        "right_torqueMeasured": r["torqueMeasured"], "right_motorCommand": r["motorCommand"],
        "right_trailingLimbAngle": r["trailingLimbAngle"], "right_momentPredicted": r["momentPredicted"],
        "left_motorAngle": l["motorAngle"], "left_motorVelocity": l["motorVelocity"],
        "left_motorCurrent": l["motorCurrent"], "left_motorTemp": l["motorTemp"],
        "left_load": l["load"],
        "left_IMUroll": l["IMUroll"], "left_IMUpitch": l["IMUpitch"], "left_IMUyaw": l["IMUyaw"],
        "left_IMUaccX": l["IMUaccX"], "left_IMUaccY": l["IMUaccY"], "left_IMUaccZ": l["IMUaccZ"],
        "left_IMUgyroX": l["IMUgyroX"], "left_IMUgyroY": l["IMUgyroY"], "left_IMUgyroZ": l["IMUgyroZ"],
        "left_gaitCycle": l["gaitCycle"], "left_torqueDesired": l["torqueDesired"],
        "left_torqueMeasured": l["torqueMeasured"], "left_motorCommand": l["motorCommand"],
        "left_trailingLimbAngle": l["trailingLimbAngle"], "left_momentPredicted": l["momentPredicted"],
        "pelvis_load": p["load"],
        "pelvis_IMUroll": p["IMUroll"], "pelvis_IMUpitch": p["IMUpitch"], "pelvis_IMUyaw": p["IMUyaw"],
        "pelvis_IMUaccX": p["IMUaccX"], "pelvis_IMUaccY": p["IMUaccY"], "pelvis_IMUaccZ": p["IMUaccZ"],
        "pelvis_IMUgyroX": p["IMUgyroX"], "pelvis_IMUgyroY": p["IMUgyroY"], "pelvis_IMUgyroZ": p["IMUgyroZ"],
    })

    out_path = Path(out_path) if out_path else bin_path.with_suffix(".csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bin_file", nargs="?", default=None,
                    help="BIN file to convert. If omitted, converts every *.BIN under data/ "
                         "that doesn't already have a matching .csv")
    ap.add_argument("--out", default=None, help="output CSV path (only valid together with an explicit bin_file)")
    args = ap.parse_args()

    data_root = Path(__file__).resolve().parent / "data"

    if args.bin_file:
        bin_path = Path(args.bin_file).expanduser().resolve()
        if not bin_path.exists():
            sys.exit(f"BIN file not found: {bin_path}")
        convert_bin_to_csv(bin_path, args.out)
    else:
        if args.out:
            sys.exit("--out requires an explicit bin_file")
        bin_paths = find_unconverted_bins(data_root)
        if not bin_paths:
            sys.exit(f"no unconverted .BIN files found under {data_root}")
        for bin_path in bin_paths:
            convert_bin_to_csv(bin_path)


if __name__ == "__main__":
    main()
