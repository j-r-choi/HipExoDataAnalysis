# Hip Exo Data Analysis

Analysis code and notebooks for the hip exoskeleton: raw exo sensor logs and
Hip-Exo-Optimization (HILO) run results.

## Layout

```
exo/                        exo-data plotting + simulation functions
optimization/               HILO optimization analysis code (TODO, not added yet)
data_convert.py             converts HIP*.BIN logs to CSV
plot_hip_exo.ipynb          IMU/motor/gait-cycle/load/torque plots
moment_mapping_params.ipynb moment-to-torque mapping parameter sweeps
data/                       session data, gitignored (OneDrive-backed, not git)
  <subject>/<date>/exo/            raw BIN + converted CSV
  <subject>/<date>/optimization/   HILO run output: HILO_state.json, GP_model.pkl,
                                    frames/Condition_*.csv (+ _EMG/_GYRO/.png)
                                    (exact files depend on which HILO mode was run)
```

`exo/` and `optimization/` mirror the `data/` subfolder names — code for a given
data type lives in the top-level folder of the same name.

**Organizing new data:** one folder per subject (currently just `pilot/`), one
folder per date under that. Each date gets its own `exo/` and `optimization/` —
file exo logs and optimization output under the date they were recorded/run,
even if one optimization run draws on multiple exo sessions.

## Converting BIN → CSV

```
python3 data_convert.py                     # convert every unconverted *.BIN under data/
python3 data_convert.py path/to/HIP043.BIN  # convert one file
```

Needs the `hipexo` conda env (numpy/pandas).
