"""Preprocessing module for event detection data."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.signal import hilbert
from _util_voltage_specific_vars import get_voltage_shift, get_capacitor_current_amp, CAPACITOR_CURRENT_DELAY


def calculate_capacitor_current(t, v, voltage):
    """Calculate capacitor current for given time and voltage arrays."""
    capacitor_current_amp = get_capacitor_current_amp(voltage)
    capacitor_current_delay = CAPACITOR_CURRENT_DELAY
    omega = 2 * np.pi * 30000

    v_peak_idx = np.argmax(v)
    t_v_peak = t[v_peak_idx]
    t_c_peak = t_v_peak + capacitor_current_delay

    phi = np.pi/2 - omega * t_c_peak
    capacitor_current = capacitor_current_amp * np.sin(omega * t + phi)

    return capacitor_current


def process_raw_data_file(t, i, v, voltage, global_time_reference, accumulated_time,
                         store_raw_data=False, sub_capacitor=True):
    """Process a single raw data file: subtract capacitor current and apply voltage shift."""
    if len(t) == 0:
        if store_raw_data:
            return None, None, None, accumulated_time, 0.0, None, None, None, None
        else:
            return None, None, None, accumulated_time, 0.0

    voltage_shift = get_voltage_shift(voltage)

    if sub_capacitor:
        capacitor_current = calculate_capacitor_current(t, v, voltage)
        adjusted_current = i - capacitor_current
    else:
        capacitor_current = np.zeros_like(i)
        adjusted_current = i.copy()

    all_values = []
    all_voltages = []
    all_timestamps = []

    if store_raw_data:
        all_raw_i = []
        all_raw_t = []
        all_raw_v = []

    for idx in range(len(t)):
        time_sec = t[idx] * 1e9
        global_time = global_time_reference + timedelta(seconds=accumulated_time + time_sec)
        all_timestamps.append(global_time)
        adjusted_value = adjusted_current[idx]
        all_values.append(adjusted_value)
        all_voltages.append(v[idx])

        if store_raw_data:
            all_raw_i.append(i[idx])
            all_raw_t.append(t[idx])
            all_raw_v.append(v[idx])

    file_duration = (t[-1] - t[0]) * 1e9
    new_accumulated_time = accumulated_time + file_duration

    if store_raw_data:
        return (all_timestamps, all_values, all_voltages, new_accumulated_time, file_duration,
                all_raw_i, all_raw_t, all_raw_v, capacitor_current)
    else:
        return all_timestamps, all_values, all_voltages, new_accumulated_time, file_duration


def normalize_voltage_relative(dataset, voltage, reference_voltage=2000):
    """Normalize voltage column relative to a reference voltage."""
    dataset = dataset.copy()
    voltage_min = dataset['voltage'].min()
    voltage_max = dataset['voltage'].max()

    if voltage_max > voltage_min:
        dataset['voltage'] = 2 * (dataset['voltage'] - voltage_min) / (voltage_max - voltage_min) - 1
    else:
        dataset['voltage'] = 0.0

    scale = voltage / reference_voltage
    dataset['voltage'] = dataset['voltage'] * scale

    return dataset


def remove_peak_frequency(dataset, bandwidth_ratio=0.01, n_peaks=1):
    """Remove the top N frequencies with highest magnitudes (and small bands around them) using FFT filtering."""
    if n_peaks < 1:
        print("Warning: n_peaks must be >= 1.")
        return dataset

    print(f"\nApplying notch filter to remove top {n_peaks} peak frequency/frequencies...")

    values = dataset['value'].values
    n = len(values)

    if n < 2:
        print("Warning: dataset too short for FFT filtering.")
        return dataset

    timestamps = dataset.index
    time_diffs = (timestamps[1:] - timestamps[:-1]).total_seconds()
    median_dt = np.median(time_diffs)
    sampling_rate = 1.0 / median_dt

    print(f"  Dataset length: {n} points")
    print(f"  Sampling rate: {sampling_rate:.2e} Hz ({sampling_rate/1e9:.2f} GHz)")

    fft_values = np.fft.fft(values)

    frequencies = np.fft.fftfreq(n, d=median_dt)

    magnitudes = np.abs(fft_values)

    positive_freq_idx = frequencies >= 0
    positive_freqs = frequencies[positive_freq_idx]
    positive_magnitudes = magnitudes[positive_freq_idx]

    if len(positive_magnitudes) <= 1:
        print("Warning: no peak frequencies found.")
        return dataset

    available_peaks = min(n_peaks, len(positive_magnitudes) - 1)
    if available_peaks < n_peaks:
        print(f"Warning: Requested {n_peaks} peaks but only {available_peaks} available. Using {available_peaks}.")

    top_indices = np.argsort(positive_magnitudes[1:])[-available_peaks:][::-1] + 1

    peak_freqs = positive_freqs[top_indices]
    peak_magnitudes = positive_magnitudes[top_indices]

    print(f"\n  Found {len(peak_freqs)} peak frequency/frequencies to remove:")
    for i, (peak_freq, peak_mag) in enumerate(zip(peak_freqs, peak_magnitudes), 1):
        print(f"    {i}. {peak_freq:.2e} Hz ({peak_freq/1e6:.2f} MHz) - Magnitude: {peak_mag:.6e}")

    mask = np.ones(n, dtype=bool)
    removed_bands = []

    for peak_freq in peak_freqs:
        bandwidth = peak_freq * bandwidth_ratio
        freq_low = peak_freq - bandwidth
        freq_high = peak_freq + bandwidth
        removed_bands.append((freq_low, freq_high))

        for i, freq in enumerate(frequencies):
            abs_freq = abs(freq)
            if freq_low <= abs_freq <= freq_high:
                mask[i] = False

    print(f"\n  Removing frequency bands (bandwidth ratio: {bandwidth_ratio}):")
    for i, (freq_low, freq_high) in enumerate(removed_bands, 1):
        print(f"    {i}. {freq_low:.2e} Hz to {freq_high:.2e} Hz")

    removed_bins = np.sum(~mask)
    print(f"\n  Removed {removed_bins} frequency bins ({removed_bins/n*100:.2f}% of spectrum)")

    filtered_fft = fft_values.copy()
    filtered_fft[~mask] = 0.0

    filtered_values = np.real(np.fft.ifft(filtered_fft))

    filtered_dataset = dataset.copy()
    filtered_dataset['value'] = filtered_values

    print(f"  Value range: [{filtered_values.min():.6f}, {filtered_values.max():.6f}]")

    return filtered_dataset


def subtract_mean(dataset):
    """Subtract the mean of the 'value' column from itself to eliminate y-axis shift."""
    print("\nSubtracting mean to eliminate y-axis shift...")

    mean_value = np.mean(dataset['value'])

    print(f"  Mean value: {mean_value:.6e}")
    print(f"  Value range before: [{dataset['value'].min():.6e}, {dataset['value'].max():.6e}]")

    dataset_centered = dataset.copy()
    dataset_centered['value'] = dataset_centered['value'] - mean_value

    print(f"  Value range after: [{dataset_centered['value'].min():.6e}, {dataset_centered['value'].max():.6e}]")
    print(f"  New mean: {np.mean(dataset_centered['value']):.6e}")

    return dataset_centered


def estimate_noise_std(dataset, events_df=None, margin_factor=2.0, min_points=1000):
    """Estimate noise standard deviation using robust statistics."""
    values = dataset['value'].values

    percentile_low = 1.0
    percentile_high = 99.0

    q_low = np.percentile(values, percentile_low)
    q_high = np.percentile(values, percentile_high)

    filtered_values = values[(values >= q_low) & (values <= q_high)]

    if len(filtered_values) < min_points:
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        iqr = q75 - q25
        estimated_std = iqr / 1.35 if iqr > 0 else np.std(values)
    else:
        estimated_std = np.std(filtered_values)

    print(f"Estimated noise std (robust percentile filtering): {estimated_std:.6e}")
    return estimated_std


def normalize_noise_level(dataset, events_df=None, target_noise_std=None,
                         margin_factor=2.0, min_points=1000):
    """Normalize dataset noise level to target value."""
    estimated_noise_std = estimate_noise_std(dataset, events_df=None, margin_factor=margin_factor, min_points=min_points)

    if target_noise_std is None or target_noise_std <= 0:
        scale_factor = 1.0
    elif estimated_noise_std > 0:
        scale_factor = target_noise_std / estimated_noise_std
    else:
        scale_factor = 1.0

    print(f"Scale factor: {scale_factor}")

    dataset_scaled = dataset.copy()
    dataset_scaled['value'] = dataset_scaled['value'] * scale_factor

    return dataset_scaled, scale_factor, estimated_noise_std


def apply_wavelet_denoising(dataset, wavelet='db4', mode='symmetric', threshold_mode='soft',
                           noise_std=None, threshold_multiplier=1.0, level=None):
    """Apply wavelet denoising to reduce noise while preserving signal features (events)."""
    try:
        import pywt
    except ImportError:
        print("Error: PyWavelets is not installed.")
        return dataset

    print("\nApplying wavelet denoising...")

    values = dataset['value'].values
    n = len(values)

    if n < 2:
        print("Warning: dataset too short for wavelet denoising.")
        return dataset

    print(f"  Dataset length: {n} points")
    print(f"  Wavelet: {wavelet}")
    print(f"  Mode: {mode}")
    print(f"  Threshold mode: {threshold_mode}")

    if noise_std is None:
        noise_std = estimate_noise_std(dataset, events_df=None, min_points=1000)
        print(f"  Estimated noise std: {noise_std:.6e}")
    else:
        print(f"  Using provided noise std: {noise_std:.6e}")

    if level is None:
        level = pywt.dwt_max_level(n, wavelet)
        print(f"  Maximum decomposition level: {level}")
    else:
        print(f"  Using specified decomposition level: {level}")

    try:
        coeffs = pywt.wavedec(values, wavelet, mode=mode, level=level)
    except Exception as e:
        print(f"Warning: wavelet decomposition failed: {e}")
        return dataset

    threshold = noise_std * np.sqrt(2 * np.log(n)) * threshold_multiplier
    print(f"  Universal threshold: {threshold:.6e} (multiplier: {threshold_multiplier})")

    coeffs_thresh = [coeffs[0]]
    for i in range(1, len(coeffs)):
        coeff_thresh = pywt.threshold(coeffs[i], threshold, mode=threshold_mode)
        coeffs_thresh.append(coeff_thresh)

    try:
        denoised_values = pywt.waverec(coeffs_thresh, wavelet, mode=mode)
    except Exception as e:
        print(f"Warning: wavelet reconstruction failed: {e}")
        return dataset

    if len(denoised_values) != n:
        if len(denoised_values) > n:
            denoised_values = denoised_values[:n]
        else:
            padding = np.full(n - len(denoised_values), denoised_values[-1])
            denoised_values = np.concatenate([denoised_values, padding])

    denoised_dataset = dataset.copy()
    denoised_dataset['value'] = denoised_values

    original_std = np.std(values)
    denoised_std = np.std(denoised_values)
    noise_reduction = (1 - denoised_std / original_std) * 100 if original_std > 0 else 0

    print(f"  Original std: {original_std:.6e}")
    print(f"  Denoised std: {denoised_std:.6e}")
    print(f"  Noise reduction: {noise_reduction:.2f}%")
    print(f"  Value range: [{denoised_values.min():.6f}, {denoised_values.max():.6f}]")

    return denoised_dataset


def apply_voltage_based_scaling(dataset, voltage, reference_voltage=2000):
    """Apply voltage-based scaling using reference_voltage/voltage factor."""
    if voltage <= 0:
        print(f"Warning: invalid voltage {voltage}.")
        return dataset

    scale_factor = reference_voltage / voltage
    print(f"\nApplying voltage-based scaling...")
    print(f"  Voltage: {voltage} V")
    print(f"  Reference voltage: {reference_voltage} V")
    print(f"  Scale factor: {scale_factor:.6f}")
    print(f"  Value range before: [{dataset['value'].min():.6e}, {dataset['value'].max():.6e}]")

    dataset_scaled = dataset.copy()
    dataset_scaled['value'] = dataset_scaled['value'] * scale_factor

    print(f"  Value range after: [{dataset_scaled['value'].min():.6e}, {dataset_scaled['value'].max():.6e}]")

    return dataset_scaled


def create_dataset_from_processed_data(all_timestamps, all_values, all_voltages,
                                       all_raw_i=None, all_raw_t=None, all_raw_v=None,
                                       capacitor_current=None):
    """Create a pandas DataFrame from processed data arrays."""
    data_dict = {
        'value': all_values,
        'voltage': all_voltages
    }

    if all_raw_i is not None:
        data_dict['raw_i'] = all_raw_i
    if all_raw_t is not None:
        data_dict['raw_t'] = all_raw_t
    if all_raw_v is not None:
        data_dict['raw_v'] = all_raw_v
    if capacitor_current is not None:
        data_dict['capacitor_current'] = capacitor_current.tolist()

    dataset = pd.DataFrame(data_dict, index=pd.to_datetime(all_timestamps))
    return dataset


def calculate_signed_charge(dataset, time_per_step=0.5e-9):
    """Calculate signed (net) charge by integrating current over time."""
    values = dataset['value'].values
    n = len(values)

    charge = np.zeros(n)
    for i in range(1, n):
        charge[i] = charge[i-1] + 0.5 * (values[i] + values[i-1]) * time_per_step

    return charge


def calculate_charge_rate(charge, time_per_step=0.5e-9):
    """Calculate the rate of growth of charge (dQ/dt) by taking the difference."""
    charge_diff = np.diff(charge)

    charge_rate = charge_diff / time_per_step

    return charge_rate


def calculate_local_noise_std(dataset, window_size=500):
    """Calculate rolling standard deviation of the current signal."""
    rolling_std = dataset['value'].rolling(window_size, center=True, min_periods=1).std()
    return rolling_std.values


def calculate_derivative(dataset, time_per_step=0.5e-9):
    """Calculate the first derivative of the current signal (dI/dt)."""
    values = dataset['value'].values
    diff = np.diff(values) / time_per_step
    return np.concatenate([[diff[0]], diff])


def calculate_hilbert_envelope(dataset):
    """Calculate the Hilbert envelope (instantaneous amplitude) of the current signal."""
    values = dataset['value'].values
    analytic_signal = hilbert(values)
    return np.abs(analytic_signal)
