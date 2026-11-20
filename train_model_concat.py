import os
os.environ['PYTHONHASHSEED'] = '41'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import random
import numpy as np
random.seed(41)
np.random.seed(41)

import tensorflow as tf
tf.random.set_seed(41)

import pandas as pd
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import preprocessing

TARGET_VOLTAGE = [1800, 2200, 2400]
WINDOW_SIZE_WIDTH = 30
MODE = "Both"

SUB_CAPACITOR = True

REFERENCE_VOLTAGE = 2000

SUBTRACT_MEAN = True

APPLY_WAVELET_FILTER = False
WAVELET_TYPE = 'bior2.2'
WAVELET_MODE = 'symmetric'
WAVELET_THRESHOLD_MODE = 'soft'
WAVELET_THRESHOLD_MULTIPLIER = 0.8
WAVELET_LEVEL = 8

FEATURES = ["i", "v", "H"]

PLOT_INPUTS = True
PLOT_INPUTS_POINTS = 50000
PLOT_INPUTS_BATCHES = 5

LOCAL_NOISE_WINDOW_SIZE = 500

USE_DENOISED_AS_FEATURE = False

COLLECT_BORDER_REFINEMENT_DATA = True
BORDER_REFINEMENT_CHUNK_SIZE = 20000

TRAIN_BORDER_REFINEMENT_ML = True
BR_MODEL = "FFN"
USE_HYPERPARAMETER_TUNING = True
HYPERPARAMETER_CONFIG = {
    'n_estimators': {'min': 50, 'max': 300},
    'max_depth': [None, 10, 15, 20, 25, 30, 35, 40],
    'min_samples_split': {'min': 2, 'max': 20},
    'min_samples_leaf': {'min': 1, 'max': 10},
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}
N_ITER = 50
CV_FOLDS = 5
TEST_SIZE = 0.2

sys.path.append(os.path.join(os.path.dirname(__file__), 'eventdetector'))

from eventdetector_ts.metamodel.meta_model import MetaModel
from eventdetector_ts import FFN, LSTM, CNN, GRU, TRANSFORMER
from eventdetector_ts.prediction.prediction import predict
from read_classes import NpzFileIterator
from border_refinement_utils import extract_features, determine_event_polarity
from train_border_refinement_ml import (
    load_border_refinement_data,
    prepare_training_data,
    prepare_cnn_training_data,
    train_models,
    save_models,
    match_events_efficiently,
    _MODEL_TYPE_CNN,
    _MODEL_TYPE_FFN,
    _MODEL_TYPE_RF,
)

MODELS = [(CNN, 1)]

FEATURE_NAME_MAP = {
    "i": "value",
    "v": "voltage",
    "q": "charge_rate",
    "ln": "local_noise_std",
    "der": "derivative",
    "H": "hilbert_envelope",
}


def get_feature_columns():
    """Build the list of dataset column names from FEATURES and USE_DENOISED_AS_FEATURE."""
    cols = [FEATURE_NAME_MAP[f] for f in FEATURES]
    if USE_DENOISED_AS_FEATURE and APPLY_WAVELET_FILTER:
        cols.append("value_original")
    return cols


def generate_model_name_string(models):
    """Generate a string representation of the model configuration."""
    model_name_map = {
        FFN: "FFN",
        LSTM: "LSTM",
        CNN: "CNN",
        GRU: "GRU",
        TRANSFORMER: "TRANSFORMER"
    }

    model_counts = {}
    for model_type, count in models:
        model_name = model_name_map.get(model_type, str(model_type))
        if model_name not in model_counts:
            model_counts[model_name] = 0
        model_counts[model_name] += count

    parts = [f"{count}{model_name}" for model_name, count in sorted(model_counts.items())]
    return "_".join(parts)


def split_voltage_data(voltage_dataset, voltage_events_df, test_size, val_size):
    """Split a single voltage's dataset and events into train/val/test sets."""
    train_prop = (1 - test_size) * (1 - val_size)
    val_prop = (1 - test_size) * val_size
    test_prop = test_size

    total_prop = train_prop + val_prop + test_prop
    train_prop = train_prop / total_prop
    val_prop = val_prop / total_prop
    test_prop = test_prop / total_prop

    n_points = len(voltage_dataset)

    train_end_idx = int(n_points * train_prop)
    val_end_idx = train_end_idx + int(n_points * val_prop)

    train_df = voltage_dataset.iloc[:train_end_idx].copy()
    val_df = voltage_dataset.iloc[train_end_idx:val_end_idx].copy()
    test_df = voltage_dataset.iloc[val_end_idx:].copy()

    if len(train_df) > 0:
        train_start = train_df.index[0]
        train_end = train_df.index[-1]
    else:
        train_start = None
        train_end = None

    if len(val_df) > 0:
        val_start = val_df.index[0]
        val_end = val_df.index[-1]
    else:
        val_start = None
        val_end = None

    if len(test_df) > 0:
        test_start = test_df.index[0]
        test_end = test_df.index[-1]
    else:
        test_start = None
        test_end = None

    train_events = []
    val_events = []
    test_events = []

    for _, event in voltage_events_df.iterrows():
        event_start = event['start']
        event_end = event['end']
        event_midpoint = event_start + (event_end - event_start) / 2

        if train_start is not None and train_end is not None and train_start <= event_midpoint <= train_end:
            train_events.append(event)
        elif val_start is not None and val_end is not None and val_start <= event_midpoint <= val_end:
            val_events.append(event)
        elif test_start is not None and test_end is not None and test_start <= event_midpoint <= test_end:
            test_events.append(event)
        else:
            if train_start is not None and train_end is not None:
                train_dist = min(abs((train_start - event_midpoint).total_seconds()),
                                abs((train_end - event_midpoint).total_seconds()))
            else:
                train_dist = float('inf')

            if val_start is not None and val_end is not None:
                val_dist = min(abs((val_start - event_midpoint).total_seconds()),
                              abs((val_end - event_midpoint).total_seconds()))
            else:
                val_dist = float('inf')

            if test_start is not None and test_end is not None:
                test_dist = min(abs((test_start - event_midpoint).total_seconds()),
                               abs((test_end - event_midpoint).total_seconds()))
            else:
                test_dist = float('inf')

            if train_dist <= val_dist and train_dist <= test_dist:
                train_events.append(event)
            elif val_dist <= test_dist:
                val_events.append(event)
            else:
                test_events.append(event)

    train_events_df = pd.DataFrame(train_events) if train_events else pd.DataFrame(columns=['start', 'end'])
    val_events_df = pd.DataFrame(val_events) if val_events else pd.DataFrame(columns=['start', 'end'])
    test_events_df = pd.DataFrame(test_events) if test_events else pd.DataFrame(columns=['start', 'end'])

    return train_df, val_df, test_df, train_events_df, val_events_df, test_events_df


def load_data():
    """Load labeled time series data and their events for TARGET_VOLTAGE."""
    if isinstance(TARGET_VOLTAGE, (int, float)):
        target_voltages = [int(TARGET_VOLTAGE)]
    elif isinstance(TARGET_VOLTAGE, (list, tuple)):
        target_voltages = [int(v) for v in TARGET_VOLTAGE]
    else:
        raise ValueError(f"TARGET_VOLTAGE must be int or list of ints, got {type(TARGET_VOLTAGE)}")

    print(f"Loading data for {len(target_voltages)} voltage(s): {target_voltages}")

    global_time_reference = datetime(2024, 1, 1, 0, 0, 0)
    accumulated_time = 0.0

    all_datasets = []
    all_events_dfs = []

    for voltage_idx, target_voltage in enumerate(target_voltages):
        print(f"\n{'='*60}")
        print(f"Processing voltage {target_voltage}V ({voltage_idx + 1}/{len(target_voltages)})")
        print(f"{'='*60}")

        directory = f"example_labeled_data/{target_voltage}"
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist. Skipping voltage {target_voltage}V.")
            continue

        reader = NpzFileIterator(directory)

        voltage_accumulated_time = 0.0

        all_values = []
        all_voltages = []
        all_timestamps = []

        try:
            while True:
                voltage, frequency, data, current_filepath = next(reader)

                t = data['t']
                i = data['i']
                v = data['v']

                if len(t) == 0:
                    continue

                result = preprocessing.process_raw_data_file(
                    t, i, v, target_voltage, global_time_reference, voltage_accumulated_time,
                    store_raw_data=False, sub_capacitor=SUB_CAPACITOR
                )

                if result is None:
                    continue

                file_timestamps, file_values, file_voltages, voltage_accumulated_time, file_duration = result

                all_timestamps.extend(file_timestamps)
                all_values.extend(file_values)
                all_voltages.extend(file_voltages)

        except StopIteration:
            pass

        if len(all_values) == 0:
            print(f"Warning: No data points found for voltage {target_voltage}V. Skipping.")
            continue

        print(f"Loaded {len(all_values)} data points for {target_voltage}V")
        print(f"Data range: [{min(all_values):.3f}, {max(all_values):.3f}]")

        voltage_dataset = preprocessing.create_dataset_from_processed_data(
            all_timestamps, all_values, all_voltages
        )

        voltage_dataset = preprocessing.normalize_voltage_relative(
            voltage_dataset, voltage=target_voltage, reference_voltage=REFERENCE_VOLTAGE
        )
        print(f"Voltage range (scaled): [{voltage_dataset['voltage'].min():.3f}, {voltage_dataset['voltage'].max():.3f}]")

        if SUBTRACT_MEAN:
            voltage_dataset = preprocessing.subtract_mean(voltage_dataset)
        else:
            print("\nMean subtraction disabled (SUBTRACT_MEAN is False)")

        print(f"\nCalculating signed charge and charge rate for {target_voltage}V...")
        TIME_PER_STEP = 0.5e-9

        signed_charge = preprocessing.calculate_signed_charge(voltage_dataset, time_per_step=TIME_PER_STEP)

        charge_rate = preprocessing.calculate_charge_rate(signed_charge, time_per_step=TIME_PER_STEP)

        if len(charge_rate) > 0:
            charge_rate_padded = np.concatenate([[charge_rate[0]], charge_rate, [charge_rate[-1]]])
            charge_rate_padded = charge_rate_padded[:len(voltage_dataset)]
        else:
            charge_rate_padded = np.zeros(len(voltage_dataset))

        voltage_dataset['charge_rate'] = charge_rate_padded
        print(f"  Signed charge range: [{signed_charge.min():.6e}, {signed_charge.max():.6e}] C")
        print(f"  Charge rate range: [{charge_rate.min():.6e}, {charge_rate.max():.6e}] C/s")

        print(f"\nApplying voltage-based scaling for {target_voltage}V...")
        voltage_dataset = preprocessing.apply_voltage_based_scaling(
            voltage_dataset,
            voltage=target_voltage,
            reference_voltage=REFERENCE_VOLTAGE
        )

        if "ln" in FEATURES:
            voltage_dataset['local_noise_std'] = preprocessing.calculate_local_noise_std(
                voltage_dataset, window_size=LOCAL_NOISE_WINDOW_SIZE
            )
            print(f"  local_noise_std range: [{voltage_dataset['local_noise_std'].min():.6e}, {voltage_dataset['local_noise_std'].max():.6e}]")

        if "der" in FEATURES:
            voltage_dataset['derivative'] = preprocessing.calculate_derivative(
                voltage_dataset, time_per_step=TIME_PER_STEP
            )
            print(f"  derivative range: [{voltage_dataset['derivative'].min():.6e}, {voltage_dataset['derivative'].max():.6e}]")

        if "H" in FEATURES:
            voltage_dataset['hilbert_envelope'] = preprocessing.calculate_hilbert_envelope(voltage_dataset)
            print(f"  hilbert_envelope range: [{voltage_dataset['hilbert_envelope'].min():.6e}, {voltage_dataset['hilbert_envelope'].max():.6e}]")

        if APPLY_WAVELET_FILTER:
            print(f"\nApplying wavelet denoising for {target_voltage}V...")

            voltage_dataset['value_original'] = voltage_dataset['value'].copy()
            if USE_DENOISED_AS_FEATURE:
                print("  Saved original current values as 'value_original' feature")

            voltage_dataset = preprocessing.apply_wavelet_denoising(
                voltage_dataset,
                wavelet=WAVELET_TYPE,
                mode=WAVELET_MODE,
                threshold_mode=WAVELET_THRESHOLD_MODE,
                noise_std=None,
                threshold_multiplier=WAVELET_THRESHOLD_MULTIPLIER,
                level=WAVELET_LEVEL
            )

        else:
            print(f"\nWavelet denoising disabled (APPLY_WAVELET_FILTER is False)")

        events_file = f'selections/{target_voltage}/time_events.txt'
        if os.path.exists(events_file):
            print(f"\nLoading events from {events_file}...")
            voltage_events_df = pd.read_csv(
                events_file,
                header=None,
                names=['start', 'end'],
                sep=' ',
                dtype=str
            )

            voltage_events_df['start'] = pd.to_datetime(voltage_events_df['start'])
            voltage_events_df['end'] = pd.to_datetime(voltage_events_df['end'])

            print(f"Found {len(voltage_events_df)} events for {target_voltage}V (before filtering)")

            dataset_start = voltage_dataset.index[0]
            dataset_end = voltage_dataset.index[-1]

            mask = (voltage_events_df['start'] <= dataset_end) & (voltage_events_df['end'] >= dataset_start)
            voltage_events_df = voltage_events_df[mask].copy()

            tolerance = timedelta(seconds=1)
            if len(voltage_events_df) > 0:
                events_before_dataset = (voltage_events_df['start'] < dataset_start - tolerance).sum()
                events_after_dataset = (voltage_events_df['end'] > dataset_end + tolerance).sum()

                if events_before_dataset > 0 or events_after_dataset > 0:
                    print(f"  Warning: Found {events_before_dataset} events starting before dataset, {events_after_dataset} events ending after dataset")
                    voltage_events_df['start'] = voltage_events_df['start'].clip(lower=dataset_start)
                    voltage_events_df['end'] = voltage_events_df['end'].clip(upper=dataset_end)

            print(f"Filtered to {len(voltage_events_df)} events within dataset time range")
            print(f"  Dataset time range: {dataset_start} to {dataset_end}")
            if len(voltage_events_df) > 0:
                print(f"  Events time range: {voltage_events_df['start'].min()} to {voltage_events_df['end'].max()}")
        else:
            print(f"Warning: Events file {events_file} does not exist. Creating empty events DataFrame.")
            voltage_events_df = pd.DataFrame(columns=['start', 'end'])

        test_size = 0.05
        val_size = 10 / 95

        print(f"\nSplitting voltage {target_voltage}V data into train/val/test...")
        train_df, val_df, test_df, train_events, val_events, test_events = split_voltage_data(
            voltage_dataset, voltage_events_df, test_size, val_size
        )

        print(f"  Train: {len(train_df)} data points, {len(train_events)} events")
        print(f"  Val: {len(val_df)} data points, {len(val_events)} events")
        print(f"  Test: {len(test_df)} data points, {len(test_events)} events")

        all_datasets.append((target_voltage, train_df, val_df, test_df, train_events, val_events, test_events))

    if len(all_datasets) == 0:
        raise ValueError(f"No data loaded for TARGET_VOLTAGE={TARGET_VOLTAGE}.")

    print(f"\n{'='*60}")
    print("Interleaving and concatenating datasets from all voltages...")
    print("Order: [all train sets] → [all val sets] → [all test sets]")
    print(f"{'='*60}")

    all_train_datasets = []
    all_val_datasets = []
    all_test_datasets = []
    all_train_events = []
    all_val_events = []
    all_test_events = []

    for target_voltage, train_df, val_df, test_df, train_events, val_events, test_events in all_datasets:
        all_train_datasets.append((target_voltage, train_df, train_events))
        all_val_datasets.append((target_voltage, val_df, val_events))
        all_test_datasets.append((target_voltage, test_df, test_events))

    if len(all_train_datasets) > 0 and len(all_train_datasets[0][1]) > 0:
        first_timestamp = all_train_datasets[0][1].index[0]
    elif len(all_val_datasets) > 0 and len(all_val_datasets[0][1]) > 0:
        first_timestamp = all_val_datasets[0][1].index[0]
    elif len(all_test_datasets) > 0 and len(all_test_datasets[0][1]) > 0:
        first_timestamp = all_test_datasets[0][1].index[0]
    else:
        raise ValueError("No data points found in any split.")

    current_time = first_timestamp
    concatenated_datasets = []
    concatenated_events = []

    print("\nConcatenating train sets from all voltages...")
    for target_voltage, train_df, train_events in all_train_datasets:
        if len(train_df) == 0:
            continue

        dataset_start = train_df.index[0]
        time_offset = current_time - dataset_start

        shifted_dataset = train_df.copy()
        shifted_dataset.index = train_df.index + time_offset

        if len(train_events) > 0:
            shifted_events = train_events.copy()
            shifted_events['start'] = train_events['start'] + time_offset
            shifted_events['end'] = train_events['end'] + time_offset
        else:
            shifted_events = train_events.copy()

        concatenated_datasets.append(shifted_dataset)
        concatenated_events.append(shifted_events)

        shifted_dataset_end = shifted_dataset.index[-1]
        current_time = shifted_dataset_end

        print(f"  Voltage {target_voltage}V train: shifted by {time_offset.total_seconds():.6f} seconds")
        print(f"    Dataset time range: {shifted_dataset.index[0]} to {shifted_dataset.index[-1]}")
        if len(train_events) > 0:
            print(f"    Events: {len(shifted_events)} events")

    print("\nConcatenating val sets from all voltages...")
    for target_voltage, val_df, val_events in all_val_datasets:
        if len(val_df) == 0:
            continue

        dataset_start = val_df.index[0]
        time_offset = current_time - dataset_start

        shifted_dataset = val_df.copy()
        shifted_dataset.index = val_df.index + time_offset

        if len(val_events) > 0:
            shifted_events = val_events.copy()
            shifted_events['start'] = val_events['start'] + time_offset
            shifted_events['end'] = val_events['end'] + time_offset
        else:
            shifted_events = val_events.copy()

        concatenated_datasets.append(shifted_dataset)
        concatenated_events.append(shifted_events)

        shifted_dataset_end = shifted_dataset.index[-1]
        current_time = shifted_dataset_end

        print(f"  Voltage {target_voltage}V val: shifted by {time_offset.total_seconds():.6f} seconds")
        print(f"    Dataset time range: {shifted_dataset.index[0]} to {shifted_dataset.index[-1]}")
        if len(val_events) > 0:
            print(f"    Events: {len(shifted_events)} events")

    print("\nConcatenating test sets from all voltages...")
    for target_voltage, test_df, test_events in all_test_datasets:
        if len(test_df) == 0:
            continue

        dataset_start = test_df.index[0]
        time_offset = current_time - dataset_start

        shifted_dataset = test_df.copy()
        shifted_dataset.index = test_df.index + time_offset

        if len(test_events) > 0:
            shifted_events = test_events.copy()
            shifted_events['start'] = test_events['start'] + time_offset
            shifted_events['end'] = test_events['end'] + time_offset
        else:
            shifted_events = test_events.copy()

        concatenated_datasets.append(shifted_dataset)
        concatenated_events.append(shifted_events)

        shifted_dataset_end = shifted_dataset.index[-1]
        current_time = shifted_dataset_end

        print(f"  Voltage {target_voltage}V test: shifted by {time_offset.total_seconds():.6f} seconds")
        print(f"    Dataset time range: {shifted_dataset.index[0]} to {shifted_dataset.index[-1]}")
        if len(test_events) > 0:
            print(f"    Events: {len(shifted_events)} events")

    dataset = pd.concat(concatenated_datasets, ignore_index=False)
    dataset = dataset.sort_index()

    events_df = pd.concat(concatenated_events, ignore_index=True)

    total_points = len(dataset)
    train_points = sum(len(train_df) for _, train_df, _ in all_train_datasets)
    val_points = sum(len(val_df) for _, val_df, _ in all_val_datasets)
    test_points = sum(len(test_df) for _, test_df, _ in all_test_datasets)

    total_events = len(events_df)
    train_events_count = sum(len(train_events) for _, _, train_events in all_train_datasets)
    val_events_count = sum(len(val_events) for _, _, val_events in all_val_datasets)
    test_events_count = sum(len(test_events) for _, _, test_events in all_test_datasets)

    print(f"\n{'='*60}")
    print("Final interleaved dataset:")
    print(f"{'='*60}")
    print(f"  Total data points: {total_points}")
    print(f"    Train portion: {train_points} ({100*train_points/total_points:.2f}%)")
    print(f"    Val portion: {val_points} ({100*val_points/total_points:.2f}%)")
    print(f"    Test portion: {test_points} ({100*test_points/total_points:.2f}%)")
    print(f"  Time range: {dataset.index[0]} to {dataset.index[-1]}")
    print(f"  Total duration: {(dataset.index[-1] - dataset.index[0]).total_seconds():.6f} seconds")
    print(f"  Total events: {total_events}")
    print(f"    Train events: {train_events_count}")
    print(f"    Val events: {val_events_count}")
    print(f"    Test events: {test_events_count}")
    print(f"\nExpected MetaModel split boundaries:")
    print(f"  DataSplitter at 95%: {dataset.index[int(total_points * 0.95)]}")
    print(f"  ModelTrainer at 85%: {dataset.index[int(total_points * 0.85)]}")
    print(f"{'='*60}")

    return dataset, events_df


def filter_events_by_mode(dataset, events_df, mode):
    """Filter events based on MODE variable."""
    if mode not in ["Neg", "Pos"]:
        raise ValueError(f"MODE must be 'Neg' or 'Pos', got '{mode}'")

    filtered_events = []

    for idx, event in events_df.iterrows():
        event_start = event['start']
        event_end = event['end']

        event_mask = (dataset.index >= event_start) & (dataset.index <= event_end)
        event_data = dataset.loc[event_mask, 'value'].values

        if len(event_data) == 0:
            continue

        max_abs_idx = np.argmax(np.abs(event_data))
        max_abs_value = event_data[max_abs_idx]

        if mode == "Neg":
            if max_abs_value < 0:
                filtered_events.append(event)
        elif mode == "Pos":
            if max_abs_value >= 0:
                filtered_events.append(event)

    if len(filtered_events) == 0:
        print(f"Warning: No events found matching MODE='{mode}'")
        return pd.DataFrame(columns=['start', 'end'])

    filtered_events_df = pd.DataFrame(filtered_events)
    print(f"Filtered events: {len(events_df)} -> {len(filtered_events_df)} (MODE='{mode}')")

    return filtered_events_df


def plot_training_inputs(dataset, output_dir="validation_plots"):
    """Plot model input features for the first PLOT_INPUTS_POINTS points in PLOT_INPUTS_BATCHES batches."""
    feature_columns = get_feature_columns()
    missing_columns = [col for col in feature_columns if col not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Cannot plot inputs. Missing feature columns: {missing_columns}")

    if PLOT_INPUTS_BATCHES < 1:
        raise ValueError(f"PLOT_INPUTS_BATCHES must be >= 1, got {PLOT_INPUTS_BATCHES}")

    points_to_plot = min(PLOT_INPUTS_POINTS, len(dataset))
    if points_to_plot == 0:
        print("PLOT_INPUTS is enabled, but dataset is empty. Skipping input plotting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    dataset_slice = dataset.iloc[:points_to_plot]

    feature_titles = {
        "value": "Current (denoised)",
        "voltage": "Voltage",
        "charge_rate": "Charge rate",
        "local_noise_std": "Local noise std",
        "derivative": "Derivative dI/dt",
        "hilbert_envelope": "Hilbert envelope",
        "value_original": "Current (original)",
    }

    batch_edges = np.linspace(0, points_to_plot, PLOT_INPUTS_BATCHES + 1, dtype=int)
    print("\n" + "=" * 60)
    print("Plotting training inputs...")
    print(f"Features: {feature_columns}")
    print(f"Points to plot: {points_to_plot}")
    print(f"Batches: {PLOT_INPUTS_BATCHES}")
    print("=" * 60)

    for batch_idx in range(PLOT_INPUTS_BATCHES):
        start_idx = batch_edges[batch_idx]
        end_idx = batch_edges[batch_idx + 1]
        if start_idx >= end_idx:
            print(f"Skipping empty input batch {batch_idx + 1}/{PLOT_INPUTS_BATCHES}.")
            continue

        batch_data = dataset_slice.iloc[start_idx:end_idx]
        x_values = np.arange(start_idx, end_idx)

        fig, axes = plt.subplots(
            len(feature_columns), 1, figsize=(15, 4 * len(feature_columns)), sharex=True
        )
        if len(feature_columns) == 1:
            axes = [axes]

        for ax, column_name in zip(axes, feature_columns):
            ax.plot(x_values, batch_data[column_name].values, linewidth=1.0)
            ax.set_ylabel(column_name, fontsize=10)
            ax.set_title(feature_titles.get(column_name, column_name), fontsize=11)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Point index")
        fig.tight_layout()

        plot_path = os.path.join(
            output_dir,
            f"input_features_batch_{batch_idx + 1}_of_{PLOT_INPUTS_BATCHES}.png",
        )
        fig.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {plot_path}")


def train_model_for_mode(dataset, events_df, mode, output_dir, width, models):
    """Train a model for a specific mode."""
    print("=" * 60)
    print(f"Training model for MODE='{mode}'")
    print("=" * 60)

    print(f"\nFiltering events based on MODE='{mode}'...")
    filtered_events_df = filter_events_by_mode(dataset, events_df, mode)

    feature_columns = get_feature_columns()
    dataset = dataset[feature_columns]

    print("\nDataset info:")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Feature columns: {feature_columns}")
    print(f"Time range: {dataset.index[0]} to {dataset.index[-1]}")
    print(f"Number of filtered events: {len(filtered_events_df)}")

    print("\nConfiguring MetaModel...")

    step = 4

    hyperparams_ffn = (1, 3, 32, 128, "sigmoid")
    hyperparams_cnn = (64, 64, 7, 7, 2, 2, "relu")
    hyperparams_rnn = (2, 2, 64, 64, "tanh")
    hyperparams_transformer = (256, 8, 4, True, "relu")

    epochs = 100
    batch_size = 128
    test_size = 0.1
    val_size = 20 / 90

    print(f"Output directory: {output_dir}")
    print(f"Window size: {width}")
    print(f"Step size: {step}")
    print(f"Models: {models}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    print("\nCreating MetaModel...")
    meta_model = MetaModel(
        output_dir=output_dir,
        dataset=dataset,
        events=filtered_events_df,
        width=width,
        step=step,
        models=models,
        hyperparams_rnn=hyperparams_rnn,
        hyperparams_cnn=hyperparams_cnn,
        hyperparams_transformer=hyperparams_transformer,
        type_training="average",
        epochs=epochs,
        batch_size=batch_size,
        test_size=test_size,
        val_size=val_size,
        scaler="RobustScaler",
        fill_nan="zeros",
        dropout=0.35,
        last_act_func="sigmoid",
        time_window=None,
        s_h=0.01
    )

    print(f"Training on {len(filtered_events_df)} events, {len(dataset)} samples")
    meta_model.fit()
    print(f"Saved model to {output_dir}")


def predict_with_model(dataset, model_path, mode):
    """Make predictions using a trained model."""
    print("=" * 60)
    print(f"Making predictions using {mode} model...")
    print("=" * 60)

    print(f"Model path: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    config_file = os.path.join(model_path, ".config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    print(f"Dataset shape: {dataset.shape}")
    print(f"Time range: {dataset.index[0]} to {dataset.index[-1]}")

    predicted_events, predicted_op, filtered_predicted_op, timestamps = predict(
        dataset=dataset,
        path=model_path
    )

    print(f"Predicted events: {len(predicted_events)}")
    print(f"OP values shape: {len(predicted_op)}")
    print(f"Timestamps shape: {len(timestamps)}")

    return predicted_events, predicted_op, filtered_predicted_op, timestamps


def combine_predicted_events(neg_events, pos_events):
    """Combine predicted events from negative and positive models."""
    print("\n" + "=" * 60)
    print("Combining predicted events from both models...")
    print("=" * 60)

    all_events = []

    for event in neg_events:
        all_events.append({
            'start': pd.to_datetime(event[0]),
            'end': pd.to_datetime(event[1]),
            'type': 'negative'
        })

    for event in pos_events:
        all_events.append({
            'start': pd.to_datetime(event[0]),
            'end': pd.to_datetime(event[1]),
            'type': 'positive'
        })

    combined_df = pd.DataFrame(all_events)

    combined_df = combined_df.sort_values('start').reset_index(drop=True)

    print(f"Total combined events: {len(combined_df)}")
    print(f"  - Negative events: {len(neg_events)}")
    print(f"  - Positive events: {len(pos_events)}")

    return combined_df


def save_border_refinement_data(dataset, op_timestamps, op_values, predicted_events_df,
                                 true_events_df, output_dir, chunk_index):
    """Save border refinement dataset for testing refinement algorithms."""
    os.makedirs(output_dir, exist_ok=True)

    if 'value_original' in dataset.columns:
        current_values = dataset['value_original'].values
    else:
        current_values = dataset['value'].values
    dataset_timestamps = dataset.index.values

    op_timestamps_save = op_timestamps if op_timestamps is not None else np.array([])
    op_values_save = op_values if op_values is not None else np.array([])

    predicted_borders = []
    if len(predicted_events_df) > 0:
        for _, event in predicted_events_df.iterrows():
            predicted_borders.append([event['start'], event['end']])

    true_borders = []
    if true_events_df is not None and len(true_events_df) > 0:
        for _, event in true_events_df.iterrows():
            true_borders.append([event['start'], event['end']])

    output_path = os.path.join(output_dir, f"border_refinement_data_file_{chunk_index:04d}.npz")
    np.savez(
        output_path,
        current_values=current_values,
        dataset_timestamps=dataset_timestamps,
        op_timestamps=op_timestamps_save,
        op_values=op_values_save,
        predicted_borders=np.array(predicted_borders, dtype=object) if len(predicted_borders) > 0 else np.array([], dtype=object).reshape(0, 2),
        true_borders=np.array(true_borders, dtype=object) if len(true_borders) > 0 else np.array([], dtype=object).reshape(0, 2)
    )
    print(f"  Saved border refinement data to: {output_path}")


def collect_border_refinement_data(dataset, events_df, results_base_dir):
    """Collect border refinement data by making predictions on full dataset and saving chunks."""
    print("\n" + "=" * 60)
    print("Collecting Border Refinement Data")
    print("=" * 60)

    neg_model_path = os.path.join(results_base_dir, "eventdetector_neg_training_results")
    pos_model_path = os.path.join(results_base_dir, "eventdetector_pos_training_results")

    if not os.path.exists(neg_model_path):
        raise FileNotFoundError(f"Negative model not found: {neg_model_path}")
    if not os.path.exists(pos_model_path):
        raise FileNotFoundError(f"Positive model not found: {pos_model_path}")

    feature_columns = get_feature_columns()
    print(f"Using feature columns: {feature_columns}")

    neg_events, neg_predicted_op, neg_filtered_predicted_op, neg_timestamps = predict_with_model(
        dataset=dataset[feature_columns],
        model_path=neg_model_path,
        mode="Neg"
    )

    pos_events, pos_predicted_op, pos_filtered_predicted_op, pos_timestamps = predict_with_model(
        dataset=dataset[feature_columns],
        model_path=pos_model_path,
        mode="Pos"
    )

    combined_events_df = combine_predicted_events(neg_events, pos_events)

    combined_op_timestamps = None
    combined_op_values = None
    if neg_timestamps is not None and neg_filtered_predicted_op is not None and len(neg_timestamps) > 0:
        combined_op_timestamps = list(neg_timestamps)
        combined_op_values = list(neg_filtered_predicted_op)
    if pos_timestamps is not None and pos_filtered_predicted_op is not None and len(pos_timestamps) > 0:
        if combined_op_timestamps is None:
            combined_op_timestamps = list(pos_timestamps)
            combined_op_values = list(pos_filtered_predicted_op)
        else:
            combined_op_timestamps.extend(pos_timestamps)
            combined_op_values.extend(pos_filtered_predicted_op)

    if combined_op_timestamps is not None and len(combined_op_timestamps) > 0:
        op_df = pd.DataFrame({
            'timestamp': combined_op_timestamps,
            'op_value': combined_op_values
        })

        op_df['timestamp'] = pd.to_datetime(op_df['timestamp'])

        op_df['timestamp_rounded'] = op_df['timestamp'].dt.round('1ms')

        op_df_grouped = op_df.groupby('timestamp_rounded')['op_value'].max().reset_index()

        combined_op_timestamps = op_df_grouped['timestamp_rounded'].values
        combined_op_values = op_df_grouped['op_value'].values

        sort_indices = np.argsort(combined_op_timestamps)
        combined_op_timestamps = combined_op_timestamps[sort_indices]
        combined_op_values = combined_op_values[sort_indices]

    border_refinement_dir = os.path.join(results_base_dir, "border_refinement_set")
    os.makedirs(border_refinement_dir, exist_ok=True)

    num_points = len(dataset)
    num_chunks = (num_points + BORDER_REFINEMENT_CHUNK_SIZE - 1) // BORDER_REFINEMENT_CHUNK_SIZE
    print(f"\nProcessing {num_points} data points in {num_chunks} chunks...")

    chunk_count = 0
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * BORDER_REFINEMENT_CHUNK_SIZE
        end_idx = min(start_idx + BORDER_REFINEMENT_CHUNK_SIZE, num_points)

        chunk_dataset = dataset.iloc[start_idx:end_idx]
        chunk_start_time = chunk_dataset.index[0]
        chunk_end_time = chunk_dataset.index[-1]

        chunk_predicted_events = combined_events_df[
            (combined_events_df['start'] <= chunk_end_time) & (combined_events_df['end'] >= chunk_start_time)
        ].copy()

        chunk_true_events = events_df[
            (events_df['start'] <= chunk_end_time) & (events_df['end'] >= chunk_start_time)
        ].copy()

        chunk_op_timestamps = None
        chunk_op_values = None
        if combined_op_timestamps is not None and len(combined_op_timestamps) > 0:
            op_timestamps_pd = pd.to_datetime(combined_op_timestamps)
            op_mask = (op_timestamps_pd >= chunk_start_time) & (op_timestamps_pd <= chunk_end_time)
            if op_mask.any():
                chunk_op_timestamps = combined_op_timestamps[op_mask]
                chunk_op_values = combined_op_values[op_mask]
            else:
                chunk_op_timestamps = np.array([])
                chunk_op_values = np.array([])

        save_border_refinement_data(
            dataset=chunk_dataset,
            op_timestamps=chunk_op_timestamps,
            op_values=chunk_op_values,
            predicted_events_df=chunk_predicted_events,
            true_events_df=chunk_true_events,
            output_dir=border_refinement_dir,
            chunk_index=chunk_idx + 1
        )

        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"Processed {chunk_count}/{num_chunks} chunks...")

    print(f"\nSaved {num_chunks} border-refinement files to {border_refinement_dir}")
    return border_refinement_dir


def main():
    print("=" * 60)
    print("EventDetector Training Script")
    print("=" * 60)

    if MODE not in ["Neg", "Pos", "Both"]:
        raise ValueError(f"MODE must be 'Neg', 'Pos', or 'Both', got '{MODE}'")

    if BR_MODEL not in ("CNN", "FFN", "RF"):
        raise ValueError(f"BR_MODEL must be 'CNN', 'FFN', or 'RF', got '{BR_MODEL}'")

    valid_features = set(FEATURE_NAME_MAP.keys())
    for f in FEATURES:
        if f not in valid_features:
            raise ValueError(f"Unknown feature '{f}' in FEATURES. Valid: {sorted(valid_features)}")
    current_like_features = {"i", "H"}
    if not any(f in FEATURES for f in current_like_features):
        raise ValueError(f"FEATURES must include at least one current-like feature: {sorted(current_like_features)}")
    print(f"Features: {FEATURES} -> columns {get_feature_columns()}")

    if USE_DENOISED_AS_FEATURE and not APPLY_WAVELET_FILTER:
        print("Warning: USE_DENOISED_AS_FEATURE is True but APPLY_WAVELET_FILTER is False.")
        print("  USE_DENOISED_AS_FEATURE requires APPLY_WAVELET_FILTER to be True.")
        print("  USE_DENOISED_AS_FEATURE will be ignored.")

    dataset, events_df = load_data()

    print("\nDataset info:")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Time range: {dataset.index[0]} to {dataset.index[-1]}")
    print(f"Total number of events (before filtering): {len(events_df)}")

    if PLOT_INPUTS:
        plot_training_inputs(dataset, output_dir="validation_plots")

    width = WINDOW_SIZE_WIDTH

    models = MODELS

    model_name_str = generate_model_name_string(models)

    now = datetime.now()
    date_str = f"{now.day}_{now.month}_{now.year}_{now.strftime('%H%M')}"

    if isinstance(TARGET_VOLTAGE, (int, float)):
        voltage_str = f"{int(TARGET_VOLTAGE)}V"
    elif isinstance(TARGET_VOLTAGE, (list, tuple)):
        voltage_str = "_".join([f"{int(v)}V" for v in TARGET_VOLTAGE])
    else:
        voltage_str = str(TARGET_VOLTAGE)

    results_base_dir = f"results_to_share/{date_str}_{voltage_str}_w{width}_{model_name_str}"
    os.makedirs(results_base_dir, exist_ok=True)
    print(f"\nResults will be saved to: {results_base_dir}")

    if MODE == "Both":
        print("\n" + "=" * 60)
        print("MODE='Both' detected: Training two models sequentially")
        print("=" * 60)

        print("\n" + "=" * 60)
        print("STEP 1: Training model for negative events")
        print("=" * 60)
        train_model_for_mode(
            dataset=dataset,
            events_df=events_df,
            mode="Neg",
            output_dir=f"{results_base_dir}/eventdetector_neg_training_results",
            width=width,
            models=models
        )

        print("\n" + "=" * 60)
        print("STEP 2: Training model for positive events")
        print("=" * 60)
        train_model_for_mode(
            dataset=dataset,
            events_df=events_df,
            mode="Pos",
            output_dir=f"{results_base_dir}/eventdetector_pos_training_results",
            width=width,
            models=models
        )

        print(f"Negative model: {results_base_dir}/eventdetector_neg_training_results")
        print(f"Positive model: {results_base_dir}/eventdetector_pos_training_results")

    else:
        if MODE == "Neg":
            output_dir = f"{results_base_dir}/eventdetector_neg_training_results"
        else:
            output_dir = f"{results_base_dir}/eventdetector_pos_training_results"

        train_model_for_mode(
            dataset=dataset,
            events_df=events_df,
            mode=MODE,
            output_dir=output_dir,
            width=width,
            models=models
        )

    if COLLECT_BORDER_REFINEMENT_DATA:
        if MODE != "Both":
            print("\n" + "=" * 60)
            print("Warning: Border refinement data collection requires both negative and positive models.")
            print("  Skipping data collection (MODE is not 'Both').")
            print("=" * 60)
        else:
            try:
                border_refinement_dir = collect_border_refinement_data(
                    dataset=dataset,
                    events_df=events_df,
                    results_base_dir=results_base_dir
                )

                if TRAIN_BORDER_REFINEMENT_ML:
                    print("\n" + "=" * 60)
                    print("Training Border Refinement ML Models")
                    print("=" * 60)
                    print(f"Border refinement model type: {BR_MODEL}")

                    print(f"\nLoading border refinement data from: {border_refinement_dir}")
                    data_list = load_border_refinement_data(border_refinement_dir)

                    if BR_MODEL == "CNN":
                        br_model_type = _MODEL_TYPE_CNN
                        X, y_left, y_right = prepare_cnn_training_data(data_list)
                        feature_names_for_save = None
                        train_kwargs = {
                            "test_size": TEST_SIZE,
                            "random_state": 42,
                            "model_type": br_model_type,
                        }
                    else:
                        br_model_type = _MODEL_TYPE_FFN if BR_MODEL == "FFN" else _MODEL_TYPE_RF
                        X, y_left, y_right, feature_names_for_save = prepare_training_data(data_list)
                        train_kwargs = {
                            "test_size": TEST_SIZE,
                            "random_state": 42,
                            "model_type": br_model_type,
                        }
                        if br_model_type == _MODEL_TYPE_RF:
                            train_kwargs["use_hyperparameter_tuning"] = USE_HYPERPARAMETER_TUNING
                            train_kwargs["hyperparameter_config"] = (
                                HYPERPARAMETER_CONFIG if USE_HYPERPARAMETER_TUNING else None
                            )
                            train_kwargs["n_iter"] = N_ITER
                            train_kwargs["cv"] = CV_FOLDS

                    training_results = train_models(X, y_left, y_right, **train_kwargs)

                    ml_models_dir = os.path.join(results_base_dir, "border_refinement_ml_models")
                    save_models(
                        model_left=training_results.get('model_left'),
                        model_right=training_results.get('model_right'),
                        feature_names=feature_names_for_save,
                        best_params_left=training_results.get('best_params_left'),
                        best_params_right=training_results.get('best_params_right'),
                        output_dir=ml_models_dir,
                        model_ffn=training_results.get('model_ffn'),
                        feature_scaler=training_results.get('feature_scaler'),
                        model_type=training_results.get('model_type', br_model_type)
                    )

                    print(f"Border refinement models: {ml_models_dir}")

            except Exception as e:
                print(f"\nError during border refinement data collection or ML training: {e}")
                print("Detection models are still available. Continuing...")
                import traceback
                traceback.print_exc()
    else:
        print("\nBorder refinement data collection skipped (COLLECT_BORDER_REFINEMENT_DATA=False)")

    print(f"\nResults: {results_base_dir}")


if __name__ == "__main__":
    main()
