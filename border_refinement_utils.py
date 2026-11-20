import numpy as np
import pandas as pd


_SHELL_RADII = [10, 25, 50, 100, 150]
_BASELINE_OFFSET_NEAR = 150
_BASELINE_OFFSET_FAR  = 200


def determine_event_polarity(current_values, dataset_timestamps, event_start, event_end):
    """Determine event polarity from the peak absolute current in its window."""
    event_mask = (dataset_timestamps >= event_start) & (dataset_timestamps <= event_end)
    event_data = current_values[event_mask]

    if len(event_data) == 0:
        return None

    max_abs_idx = np.argmax(np.abs(event_data))
    max_abs_value = event_data[max_abs_idx]

    return 'positive' if max_abs_value >= 0 else 'negative'


def _shell_stats(current_values, center_idx, radius):
    """Return mean, std, energy for current values in [center-radius, center+radius]."""
    lo = max(0, center_idx - radius)
    hi = min(len(current_values), center_idx + radius + 1)
    window = current_values[lo:hi]
    if len(window) == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(window)), float(np.std(window)), float(np.mean(window ** 2))


def _ring_stats(current_values, center_idx, r_inner, r_outer):
    """Return mean, std, energy for the ring between r_inner and r_outer (exclusive of inner)."""
    lo_outer = max(0, center_idx - r_outer)
    hi_outer = min(len(current_values), center_idx + r_outer + 1)
    lo_inner = max(0, center_idx - r_inner)
    hi_inner = min(len(current_values), center_idx + r_inner + 1)

    left_ring  = current_values[lo_outer:lo_inner]
    right_ring = current_values[hi_inner:hi_outer]
    ring = np.concatenate([left_ring, right_ring])
    if len(ring) == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(ring)), float(np.std(ring)), float(np.mean(ring ** 2))


def extract_features(predicted_event, op_timestamps, op_values, dataset_timestamps, current_values):
    """Extract features for a predicted event to train the ML model."""
    pred_start = pd.to_datetime(predicted_event[0])
    pred_end   = pd.to_datetime(predicted_event[1])

    op_timestamps_pd = pd.to_datetime(op_timestamps)

    event_mask_op = (op_timestamps_pd >= pred_start) & (op_timestamps_pd <= pred_end)
    event_op_values     = op_values[event_mask_op]
    event_op_timestamps = op_timestamps_pd[event_mask_op]

    if len(event_op_values) == 0:
        return None

    peak_local_idx  = int(np.argmax(event_op_values))
    peak_op_value   = float(event_op_values[peak_local_idx])
    peak_timestamp  = event_op_timestamps[peak_local_idx]

    event_global_indices = np.where(event_mask_op)[0]
    peak_global_op_idx   = int(event_global_indices[peak_local_idx])

    peak_position_relative = float(peak_local_idx / len(event_op_values)) if len(event_op_values) > 0 else 0.5

    half_max = 0.5 * peak_op_value
    left_hw = 0
    for i in range(peak_global_op_idx - 1, max(-1, peak_global_op_idx - len(op_values) - 1), -1):
        if i < 0 or op_values[i] < half_max:
            break
        left_hw += 1
    right_hw = 0
    for i in range(peak_global_op_idx + 1, min(len(op_values), peak_global_op_idx + len(op_values) + 1)):
        if op_values[i] < half_max:
            break
        right_hw += 1
    op_half_width = float(left_hw + right_hw)

    decay_window = 5
    left_end_idx  = max(0, peak_global_op_idx - decay_window)
    right_end_idx = min(len(op_values) - 1, peak_global_op_idx + decay_window)

    left_vals  = op_values[left_end_idx : peak_global_op_idx]
    right_vals = op_values[peak_global_op_idx + 1 : right_end_idx + 1]

    if len(left_vals) > 0:
        op_left_decay_rate  = float((peak_op_value - float(left_vals[0])) / len(left_vals))
    else:
        op_left_decay_rate  = 0.0

    if len(right_vals) > 0:
        op_right_decay_rate = float((peak_op_value - float(right_vals[-1])) / len(right_vals))
    else:
        op_right_decay_rate = 0.0

    op_asymmetry = op_left_decay_rate - op_right_decay_rate

    time_diffs_current = np.abs((dataset_timestamps - peak_timestamp).total_seconds())
    peak_current_idx   = int(np.argmin(time_diffs_current))

    shell_features = {}
    shell_energies = []
    for i, radius in enumerate(_SHELL_RADII):
        m, s, e = _shell_stats(current_values, peak_current_idx, radius)
        shell_features[f'current_shell_{i}_mean']   = m
        shell_features[f'current_shell_{i}_std']    = s
        shell_features[f'current_shell_{i}_energy'] = e
        shell_energies.append(e)

    for i in range(len(_SHELL_RADII) - 1):
        r_inner = _SHELL_RADII[i]
        r_outer = _SHELL_RADII[i + 1]
        m, s, e = _ring_stats(current_values, peak_current_idx, r_inner, r_outer)
        shell_features[f'current_ring_{i}_mean']   = m
        shell_features[f'current_ring_{i}_std']    = s
        shell_features[f'current_ring_{i}_energy'] = e

    pre_lo  = max(0, peak_current_idx - _BASELINE_OFFSET_FAR)
    pre_hi  = max(0, peak_current_idx - _BASELINE_OFFSET_NEAR)
    post_lo = min(len(current_values), peak_current_idx + _BASELINE_OFFSET_NEAR)
    post_hi = min(len(current_values), peak_current_idx + _BASELINE_OFFSET_FAR)

    pre_window  = current_values[pre_lo:pre_hi]
    post_window = current_values[post_lo:post_hi]

    if len(pre_window) > 0:
        baseline_pre_mean   = float(np.mean(pre_window))
        baseline_pre_std    = float(np.std(pre_window))
        baseline_pre_energy = float(np.mean(pre_window ** 2))
    else:
        baseline_pre_mean = baseline_pre_std = baseline_pre_energy = 0.0

    if len(post_window) > 0:
        baseline_post_mean   = float(np.mean(post_window))
        baseline_post_std    = float(np.std(post_window))
        baseline_post_energy = float(np.mean(post_window ** 2))
    else:
        baseline_post_mean = baseline_post_std = baseline_post_energy = 0.0

    baseline_energy_avg = (baseline_pre_energy + baseline_post_energy) / 2.0
    _EPS = 1e-30

    ratio_features = {}
    for i, e in enumerate(shell_energies):
        ratio_features[f'shell_{i}_to_baseline_energy_ratio'] = e / (baseline_energy_avg + _EPS)

    event_mask_current = (dataset_timestamps >= pred_start) & (dataset_timestamps <= pred_end)
    event_current_values = current_values[event_mask_current]

    if len(event_current_values) > 0:
        current_mean_in_event = float(np.mean(event_current_values))
        current_std_in_event  = float(np.std(event_current_values))
        current_amplitude     = float(np.max(np.abs(event_current_values)))
    else:
        current_mean_in_event = current_std_in_event = current_amplitude = 0.0

    polarity          = determine_event_polarity(current_values, dataset_timestamps, pred_start, pred_end)
    polarity_encoded  = 1 if polarity == 'positive' else 0

    features = {
        'op_peak_value':           peak_op_value,
        'peak_position_relative':  peak_position_relative,
        'op_half_width':           op_half_width,
        'op_left_decay_rate':      op_left_decay_rate,
        'op_right_decay_rate':     op_right_decay_rate,
        'op_asymmetry':            op_asymmetry,
        'current_mean_in_event':   current_mean_in_event,
        'current_std_in_event':    current_std_in_event,
        'current_amplitude':       current_amplitude,
        'polarity':                polarity_encoded,
        'baseline_pre_mean':       baseline_pre_mean,
        'baseline_pre_std':        baseline_pre_std,
        'baseline_pre_energy':     baseline_pre_energy,
        'baseline_post_mean':      baseline_post_mean,
        'baseline_post_std':       baseline_post_std,
        'baseline_post_energy':    baseline_post_energy,
    }
    features.update(shell_features)
    features.update(ratio_features)

    return features


def extract_cnn_window(predicted_event, op_timestamps, op_values,
                       dataset_timestamps, current_values, window_size=300):
    """Extract a 2-channel window (current + interpolated OP) centered on the OP peak."""
    pred_start = pd.to_datetime(predicted_event[0])
    pred_end = pd.to_datetime(predicted_event[1])

    op_timestamps_pd = pd.to_datetime(op_timestamps)

    event_mask_op = (op_timestamps_pd >= pred_start) & (op_timestamps_pd <= pred_end)
    event_op_values = op_values[event_mask_op]

    if len(event_op_values) == 0:
        return None

    peak_local_idx = int(np.argmax(event_op_values))
    peak_timestamp = op_timestamps_pd[event_mask_op][peak_local_idx]

    time_diffs = np.abs((dataset_timestamps - peak_timestamp).total_seconds())
    peak_current_idx = int(np.argmin(time_diffs))

    half = window_size // 2
    lo = peak_current_idx - half
    hi = peak_current_idx + half

    current_window = np.zeros(window_size, dtype=np.float64)
    src_lo = max(0, lo)
    src_hi = min(len(current_values), hi)
    dst_lo = src_lo - lo
    dst_hi = dst_lo + (src_hi - src_lo)
    current_window[dst_lo:dst_hi] = current_values[src_lo:src_hi]

    window_timestamps = dataset_timestamps[src_lo:src_hi]
    op_seconds = (op_timestamps_pd - op_timestamps_pd[0]).total_seconds().values.astype(np.float64)
    window_seconds = (window_timestamps - op_timestamps_pd[0]).total_seconds().values.astype(np.float64)

    op_interp_window = np.zeros(window_size, dtype=np.float64)
    if len(window_seconds) > 0 and len(op_seconds) > 1:
        interp_vals = np.interp(window_seconds, op_seconds, op_values)
        op_interp_window[dst_lo:dst_hi] = interp_vals

    return np.stack([current_window, op_interp_window], axis=-1)
