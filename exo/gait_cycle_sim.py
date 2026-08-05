"""Offline re-run of the on-MCU gait-cycle detector against logged IMU data.

Python port of `process_leg()` in Core/Src/control_logic.c, retuned to match the
constants actually compiled into the firmware (IMU_SMOOTH_NUM=30, offset -5). Feeding
it the same `{side}_IMUroll` / `{side}_IMUgyroX` columns the MCU itself read lets you
compare the simulated gaitCycle against the logged `{side}_gaitCycle` and see exactly
where (if anywhere) the two diverge.

Matches the fixed process_leg(): the phaseRadius-based Walking Stop Check stays gated on
istride>1 (it needs calibrated angle/velocity center & size factors to mean anything,
and is not reachable before that), debouncing a phaseRadius dip for `stop_confirm_s`
before confirming a real stop and fully resetting the normalization state when it does.
A leg stuck at istride<=1 (which used to be able to latch walkingFlag=True forever, since
the phaseRadius check was unreachable there) instead recovers via a plain elapsed-time
check: `single_stride_timeout_s` since the one confirmed stride with no second stride.
startMoveFlag auto-clears after `start_move_timeout_s` if gait never confirms, and
peak/trough tracking is gated on startMoveFlag and only runs while not mid-debounce --
all ported from reference/hip_exo_controller_v4.m.
"""
import math

import numpy as np

IMU_SMOOTH_NUM = 30
STEP_TRACK_NUM = 3

START_MOVE_VEL_THRESHOLD = 0.7
NO_MOVE_VEL_THRESHOLD = 0.3
NO_MOVE_TIME_THRESHOLD = 0.5
VELOCITY_PEAK_THRESHOLD = 1.0
RADIUS_THRESHOLD = 0.4
PAUSE_TIME_THRESHOLD = 1.0

FREQ_SAMPLING = 100.0
FREQ_CUTOFF = 2.0

# Default moment_control.c MomentParams (BLE can override these at runtime; the CSV
# doesn't log the live values, so torqueDesired reproduction is only as good as these).
MOMENT_SCALE = 0.4
MOMENT_WEIGHT = 60.0
MOMENT_DELAY_SAMPLES = 20  # 200 ms delay / 10 ms push interval, both at 100 Hz


def simulate_gait_cycle(df, side, calibration_range=(2.0, 5.0),
                         stop_confirm_s=0.3, start_move_timeout_s=1.0,
                         single_stride_timeout_s=3.0):
    """Re-run the firmware's gait-cycle state machine over logged IMU data.

    `calibration_range` mirrors main.c's (initializationTime, initializationTime +
    calibrationTime) window used to compute imuOffset = mean(roll) over that window.
    `stop_confirm_s`/`start_move_timeout_s`/`single_stride_timeout_s` match
    control_logic.c's stopConfirmTime/startMoveTimeout/singleStrideTimeout constants --
    exposed as arguments for what-if experiments, but the defaults are what's actually
    compiled into the firmware.
    """
    t = df["time"].to_numpy()
    roll = df[f"{side}_IMUroll"].to_numpy()
    gyro_x = df[f"{side}_IMUgyroX"].to_numpy()
    n = len(t)

    cal_mask = (t > calibration_range[0]) & (t <= calibration_range[1])
    imu_offset = roll[cal_mask].mean() if cal_mask.any() else 0.0

    angle_raw = roll - imu_offset - 5.0
    velocity_raw = -gyro_x

    T = 1.0 / FREQ_SAMPLING
    cutoff = 2.0 * math.pi * FREQ_CUTOFF
    a = cutoff / (2.0 / T + cutoff)
    b = a
    c = (2.0 / T - cutoff) / (2.0 / T + cutoff)

    angle_filt = 0.0
    velocity_filt = 0.0
    phase_angle_filt = 0.0
    phase_angle_filt_prev = 0.0
    phase_angle_smooth = 0.0
    phase_angle_smooth_prev = 0.0

    angle_smooth_hist = [0.0, 0.0, 0.0]  # [i-2, i-1, i]
    velocity_smooth_hist = [0.0, 0.0, 0.0]

    istride = 0
    istride_prev = 0
    step_prev_time = 0.0
    step_durations = [0.0] * STEP_TRACK_NUM

    start_move_flag = False
    start_move_time = 0.0
    walking_flag = False
    walking_stop_time = 0.0
    no_move_flag = False
    no_move_time = 0.0
    phase_wrap_flag = False
    phase_wrap_time = 0.0
    radius_above_flag = True
    radius_above_time = 0.0

    angle_min_flag = True
    angle_max_flag = False
    velocity_min_flag = True
    velocity_max_flag = False

    angle_min = angle_max = 0.0
    angle_center_factor = 0.0
    angle_size_factor = 100.0

    velocity_min = velocity_max = 0.0
    velocity_center_factor = 0.0
    velocity_size_factor = 100.0

    # Rolling 30-sample sums via cumulative sum, matching the zero-padded ring buffer
    cum_angle = np.concatenate([[0.0], np.cumsum(angle_raw)])
    cum_velocity = np.concatenate([[0.0], np.cumsum(velocity_raw)])

    gait_cycle_norm = np.zeros(n)
    walking_flag_out = np.zeros(n, dtype=bool)
    istride_out = np.zeros(n, dtype=int)
    phase_radius_out = np.zeros(n)
    angle_smooth_out = np.zeros(n)
    velocity_smooth_out = np.zeros(n)
    phase_angle_smooth_out = np.zeros(n)

    def reset_state():
        nonlocal walking_flag, start_move_flag, no_move_flag
        nonlocal phase_wrap_flag, radius_above_flag, istride
        nonlocal angle_min_flag, angle_max_flag, velocity_min_flag, velocity_max_flag
        nonlocal angle_min, angle_max, velocity_min, velocity_max
        nonlocal angle_center_factor, velocity_center_factor, angle_size_factor, velocity_size_factor
        walking_flag = False
        start_move_flag = False
        no_move_flag = False
        phase_wrap_flag = False
        radius_above_flag = True
        istride = 0
        angle_min_flag = True
        angle_max_flag = False
        velocity_min_flag = True
        velocity_max_flag = False
        angle_min = angle_max = 0.0
        velocity_min = velocity_max = 0.0
        angle_center_factor = 0.0
        velocity_center_factor = 0.0
        angle_size_factor = 100.0
        velocity_size_factor = 100.0

    for i in range(n):
        time_sec = t[i]

        raw_prev_angle = angle_raw[i - 1] if i > 0 else 0.0
        raw_prev_velocity = velocity_raw[i - 1] if i > 0 else 0.0

        angle_filt = a * angle_raw[i] + b * raw_prev_angle + c * angle_filt
        velocity_filt = a * velocity_raw[i] + b * raw_prev_velocity + c * velocity_filt

        phase_angle_filt = math.atan2(velocity_filt, angle_filt - angle_center_factor)
        if phase_angle_filt < 0:
            phase_angle_filt += 2 * math.pi

        window_start = max(0, i + 1 - IMU_SMOOTH_NUM)
        angle_smooth = (cum_angle[i + 1] - cum_angle[window_start]) / IMU_SMOOTH_NUM
        velocity_smooth = (cum_velocity[i + 1] - cum_velocity[window_start]) / IMU_SMOOTH_NUM

        angle_smooth_hist = [angle_smooth_hist[1], angle_smooth_hist[2], angle_smooth]
        velocity_smooth_hist = [velocity_smooth_hist[1], velocity_smooth_hist[2], velocity_smooth]

        phase_angle_smooth = math.atan2(velocity_smooth, angle_smooth - angle_center_factor)
        if phase_angle_smooth < 0:
            phase_angle_smooth += 2 * math.pi

        # Walking Start Detection
        if abs(velocity_smooth) > START_MOVE_VEL_THRESHOLD and (time_sec - walking_stop_time) > PAUSE_TIME_THRESHOLD:
            start_move_flag = True
            start_move_time = time_sec

        # A motion blip that never turns into a confirmed gait shouldn't leave
        # start_move_flag armed forever.
        if start_move_flag and (time_sec - start_move_time) > start_move_timeout_s:
            start_move_flag = False

        # Real Walking (Gait) Detection
        if start_move_flag and (time_sec - step_prev_time) > 0.5:
            if not phase_wrap_flag and (phase_angle_filt_prev - phase_angle_filt) > 4.0:
                phase_wrap_flag = True
                phase_wrap_time = time_sec

            if (phase_wrap_flag and (time_sec - phase_wrap_time) > 0.25) or (
                (phase_angle_smooth_prev - phase_angle_smooth) > 4.0
            ):
                walking_flag = True

                if istride >= 1:
                    step_durations = step_durations[1:] + [time_sec - step_prev_time]

                istride += 1
                step_prev_time = time_sec
                phase_wrap_flag = False

        phase_angle_filt_prev = phase_angle_filt
        phase_angle_smooth_prev = phase_angle_smooth

        # Calculating Normalization Parameters (peak detection on the smoothed signal) --
        # only while a gait attempt is in progress and not mid-debounce.
        if start_move_flag and radius_above_flag:
            if angle_max_flag:
                if (angle_smooth_hist[1] < angle_smooth_hist[2] and angle_smooth_hist[1] <= angle_smooth_hist[0]
                        and angle_smooth_hist[1] < angle_max - 10.0):
                    angle_min = angle_smooth_hist[1]
                    angle_min_flag = True
                    angle_max_flag = False
            if angle_min_flag:
                if (angle_smooth_hist[1] > angle_smooth_hist[2] and angle_smooth_hist[1] >= angle_smooth_hist[0]
                        and angle_smooth_hist[1] > angle_min + 10.0):
                    angle_max = angle_smooth_hist[1]
                    angle_min_flag = False
                    angle_max_flag = True
            if velocity_max_flag:
                if (velocity_smooth_hist[1] < velocity_smooth_hist[2] and velocity_smooth_hist[1] <= velocity_smooth_hist[0]
                        and velocity_smooth_hist[1] < -VELOCITY_PEAK_THRESHOLD):
                    velocity_min = velocity_smooth_hist[1]
                    velocity_min_flag = True
                    velocity_max_flag = False
            if velocity_min_flag:
                if (velocity_smooth_hist[1] > velocity_smooth_hist[2] and velocity_smooth_hist[1] >= velocity_smooth_hist[0]
                        and velocity_smooth_hist[1] > VELOCITY_PEAK_THRESHOLD):
                    velocity_max = velocity_smooth_hist[1]
                    velocity_min_flag = False
                    velocity_max_flag = True

        # Update Normalization Parameter
        if istride > 1:
            angle_center_factor = (angle_max + angle_min) / 2.0
            velocity_center_factor = (velocity_max + velocity_min) / 2.0
            angle_size_factor = (angle_max - angle_min) / 2.0
            velocity_size_factor = (velocity_max - velocity_min) / 2.0

            if angle_size_factor < 0.1:
                angle_size_factor = 1.0
            if velocity_size_factor < 0.1:
                velocity_size_factor = 1.0

        angle_norm = (angle_smooth - angle_center_factor) / angle_size_factor
        velocity_norm = (velocity_smooth - velocity_center_factor) / velocity_size_factor

        # phase_angle_norm must be computed BEFORE the No Movement Detection zeroing
        # below -- control_logic.c computes *phaseAngleNorm right here, and only
        # *phaseRadius (after) sees the zeroed angle_norm/velocity_norm.
        phase_angle_norm = math.atan2(velocity_norm, angle_norm)
        if phase_angle_norm < 0:
            phase_angle_norm += 2 * math.pi

        # No Movement Detection
        if abs(velocity_smooth) < NO_MOVE_VEL_THRESHOLD:
            if not no_move_flag:
                no_move_time = time_sec
            elif (time_sec - no_move_time) > NO_MOVE_TIME_THRESHOLD:
                angle_norm = 0.0
                velocity_norm = 0.0
            no_move_flag = True
        else:
            no_move_flag = False

        phase_radius = math.sqrt(angle_norm**2 + velocity_norm**2)

        # Gait Cycle Estimation.
        if walking_flag:
            if istride > 1:
                if istride > STEP_TRACK_NUM:
                    mean_step_duration = sum(step_durations) / STEP_TRACK_NUM
                    if (time_sec - step_prev_time) > mean_step_duration:
                        cycle = 100.0
                    else:
                        cycle = 100.0 * math.fmod(time_sec - step_prev_time, mean_step_duration) / mean_step_duration
                else:
                    cycle = phase_angle_norm * 100.0 / (2 * math.pi)

                # Debounced Walking Stop Check. Stays gated on istride>1 -- phase_radius
                # isn't calibrated before that (see istride<=1 case below).
                if istride_prev <= 1:
                    radius_above_time = time_sec  # fresh debounce clock for a new calibration

                if phase_radius < RADIUS_THRESHOLD:
                    radius_above_flag = False
                    if (time_sec - radius_above_time) > stop_confirm_s:  # confirmed stop
                        walking_stop_time = time_sec
                        cycle = 0.0
                        reset_state()
                else:  # still above threshold -- push the debounce clock forward
                    radius_above_flag = True
                    radius_above_time = time_sec
            else:
                # istride<=1: not calibrated yet, so fall back to a plain timeout
                # instead of the phase_radius check (this used to latch forever here).
                cycle = 0.0
                if (time_sec - step_prev_time) > single_stride_timeout_s:
                    walking_stop_time = time_sec
                    reset_state()
        else:
            cycle = 0.0

        istride_prev = istride

        gait_cycle_norm[i] = cycle
        walking_flag_out[i] = walking_flag
        istride_out[i] = istride
        phase_radius_out[i] = phase_radius
        angle_smooth_out[i] = angle_smooth
        velocity_smooth_out[i] = velocity_smooth
        phase_angle_smooth_out[i] = phase_angle_smooth

    delayed_moment = df[f"{side}_momentPredicted"].shift(MOMENT_DELAY_SAMPLES, fill_value=0.0).to_numpy()
    torque_desired = np.where(walking_flag_out, -MOMENT_SCALE * MOMENT_WEIGHT * delayed_moment, 0.0)

    return {
        "time": t,
        "gaitCycleNorm": gait_cycle_norm,
        "walkingFlag": walking_flag_out,
        "istride": istride_out,
        "phaseRadius": phase_radius_out,
        "torqueDesired": torque_desired,
        "imuOffset": imu_offset,
        "angleSmooth": angle_smooth_out,
        "velocitySmooth": velocity_smooth_out,
        "phaseAngleSmooth": phase_angle_smooth_out,
    }
