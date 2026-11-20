import json
import numpy as np
import pandas as pd
import sys
import os
import gc
import pickle
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

_BR_METADATA_FILENAME = "model_metadata.json"
_BR_TARGET_TYPE       = "log_half_width"
_BR_MODEL_TYPE_RF     = "rf"
_BR_MODEL_TYPE_FFN    = "ffn"
_BR_MODEL_TYPE_CNN    = "cnn"
from _util_voltage_specific_vars import get_voltage_shift
import preprocessing

MODEL_DIR = "results_to_share/trained_model"

TEST_AGAINST_ANNOTATION_OUTPUT_DIR = "border_refinement_test"

TARGET_VOLTAGE = 2400

SUB_CAPACITOR = False

REFERENCE_VOLTAGE = 2000

SUBTRACT_MEAN = True

LOCAL_NOISE_WINDOW_SIZE = 500

NEG_FEATURES = ["i", "v", "H"]
NEG_APPLY_WAVELET_FILTER = False
NEG_USE_DENOISED_AS_FEATURE = False

POS_FEATURES = ["i", "v", "H"]
POS_APPLY_WAVELET_FILTER = False
POS_USE_DENOISED_AS_FEATURE = False

WAVELET_TYPE = 'bior2.2'
WAVELET_MODE = 'symmetric'
WAVELET_THRESHOLD_MODE = 'soft'
WAVELET_THRESHOLD_MULTIPLIER = 0.8
WAVELET_LEVEL = 8

USE_ANNOTATION = False

ANNOTATION_DATA_DIR = None

ANNOTATION_SELECTIONS_DIR = None

SAVE_PLOTS = True

SAVE_PLOTS_ONLY_WITH_EVENTS = True

APPLY_REFINEMENT = False

BORDER_REFINEMENT_ML_MODELS_DIR = ""

sys.path.append(os.path.join(os.path.dirname(__file__), 'eventdetector'))

from eventdetector_ts.prediction.prediction import predict
from read_classes import NpzFileIterator
from train_border_refinement_ml import (
    match_events_efficiently,
    calculate_border_metrics,
    refine_event_boundaries_ml,
)

FEATURE_NAME_MAP = {
    "i": "value",
    "v": "voltage",
    "q": "charge_rate",
    "ln": "local_noise_std",
    "der": "derivative",
    "H": "hilbert_envelope",
}

_ALL_FEATURES = set(NEG_FEATURES) | set(POS_FEATURES)

_GLOBAL_APPLY_WAVELET = NEG_APPLY_WAVELET_FILTER or POS_APPLY_WAVELET_FILTER

SAMPLE_PERIOD_NS = 0.5


def sample_period_s():
    return SAMPLE_PERIOD_NS * 1e-9


def get_feature_columns(features, use_denoised_as_feature, apply_wavelet_filter):
    """Build the list of dataset column names for a specific model."""
    cols = [FEATURE_NAME_MAP[f] for f in features]
    if use_denoised_as_feature and apply_wavelet_filter:
        cols.append("value_original")
    return cols


def prepare_model_dataset(dataset, features, use_denoised, apply_wavelet):
    """Build a feature DataFrame with exactly the columns a specific model expects."""
    col_names = get_feature_columns(features, use_denoised, apply_wavelet)

    sub = pd.DataFrame(index=dataset.index)
    for col in col_names:
        if col == 'value' and not apply_wavelet and _GLOBAL_APPLY_WAVELET:
            sub[col] = dataset['value_original']
        else:
            sub[col] = dataset[col]

    return sub


def load_border_refinement_models(models_dir="border_refinement_ml_models"):
    """Load trained ML models for border refinement."""
    feature_names_path = os.path.join(models_dir, "feature_names.pkl")
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")

    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)

    metadata_path = os.path.join(models_dir, _BR_METADATA_FILENAME)
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        target_type = metadata.get("target_type", _BR_TARGET_TYPE)
        model_type  = metadata.get("model_type",  _BR_MODEL_TYPE_RF)
    else:
        target_type = _BR_TARGET_TYPE
        model_type  = _BR_MODEL_TYPE_RF

    if model_type == _BR_MODEL_TYPE_CNN:
        left_path = os.path.join(models_dir, "model_left.keras")
        right_path = os.path.join(models_dir, "model_right.keras")
        scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
        if not os.path.exists(left_path):
            raise FileNotFoundError(f"CNN left model not found: {left_path}")
        if not os.path.exists(right_path):
            raise FileNotFoundError(f"CNN right model not found: {right_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Channel scaler not found: {scaler_path}")

        model_left = tf.keras.models.load_model(left_path)
        model_right = tf.keras.models.load_model(right_path)
        with open(scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)

        print(f"  Model type: CNN (separate left/right 1D-Conv models)")
        print(f"  Border refinement target type: '{target_type}'")
        return model_left, model_right, feature_names, target_type, model_type, feature_scaler
    elif model_type == _BR_MODEL_TYPE_FFN:
        ffn_path    = os.path.join(models_dir, "model_ffn.keras")
        scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
        if not os.path.exists(ffn_path):
            raise FileNotFoundError(f"FFN model not found: {ffn_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Feature scaler not found: {scaler_path}")

        model_ffn = tf.keras.models.load_model(ffn_path)
        with open(scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)

        print(f"  Model type: FFN (joint two-output)")
        print(f"  Border refinement target type: '{target_type}'")
        return model_ffn, None, feature_names, target_type, model_type, feature_scaler
    else:
        model_left_path  = os.path.join(models_dir, "model_left.pkl")
        model_right_path = os.path.join(models_dir, "model_right.pkl")
        if not os.path.exists(model_left_path):
            raise FileNotFoundError(f"Left border model not found: {model_left_path}")
        if not os.path.exists(model_right_path):
            raise FileNotFoundError(f"Right border model not found: {model_right_path}")

        with open(model_left_path, 'rb') as f:
            model_left = pickle.load(f)
        with open(model_right_path, 'rb') as f:
            model_right = pickle.load(f)

        print(f"  Model type: Random Forest (separate left/right models)")
        print(f"  Border refinement target type: '{target_type}'")
        return model_left, model_right, feature_names, target_type, model_type, None


def load_data_from_single_file(voltage, file_data, global_time_reference, accumulated_time):
    """Load time series data from a single file."""
    voltage_val, frequency, data, current_filepath = file_data

    t = data['t']
    i = data['i']
    v = data['v']

    if len(t) == 0:
        return None, accumulated_time, 0.0

    result = preprocessing.process_raw_data_file(
        t, i, v, voltage, global_time_reference, accumulated_time,
        store_raw_data=True, sub_capacitor=SUB_CAPACITOR
    )

    if result is None:
        return None, accumulated_time, 0.0

    (all_timestamps, all_values, all_voltages, new_accumulated_time, file_duration,
     all_raw_i, all_raw_t, all_raw_v, capacitor_current) = result

    dataset = preprocessing.create_dataset_from_processed_data(
        all_timestamps, all_values, all_voltages,
        all_raw_i, all_raw_t, all_raw_v, capacitor_current
    )

    dataset = preprocessing.normalize_voltage_relative(dataset, voltage=TARGET_VOLTAGE, reference_voltage=REFERENCE_VOLTAGE)

    if SUBTRACT_MEAN:
        dataset = preprocessing.subtract_mean(dataset)

    print("  Calculating signed charge and charge rate...")
    TIME_PER_STEP = sample_period_s()

    signed_charge = preprocessing.calculate_signed_charge(dataset, time_per_step=TIME_PER_STEP)

    charge_rate = preprocessing.calculate_charge_rate(signed_charge, time_per_step=TIME_PER_STEP)

    if len(charge_rate) > 0:
        charge_rate_padded = np.concatenate([[charge_rate[0]], charge_rate, [charge_rate[-1]]])
        charge_rate_padded = charge_rate_padded[:len(dataset)]
    else:
        charge_rate_padded = np.zeros(len(dataset))

    dataset['charge_rate'] = charge_rate_padded
    print(f"    Signed charge range: [{signed_charge.min():.6e}, {signed_charge.max():.6e}] C")
    print(f"    Charge rate range: [{charge_rate.min():.6e}, {charge_rate.max():.6e}] C/s")

    dataset = preprocessing.apply_voltage_based_scaling(
        dataset,
        voltage=voltage,
        reference_voltage=REFERENCE_VOLTAGE
    )

    TIME_PER_STEP = sample_period_s()
    if "ln" in _ALL_FEATURES:
        dataset['local_noise_std'] = preprocessing.calculate_local_noise_std(
            dataset, window_size=LOCAL_NOISE_WINDOW_SIZE
        )
    if "der" in _ALL_FEATURES:
        dataset['derivative'] = preprocessing.calculate_derivative(dataset, time_per_step=TIME_PER_STEP)
    if "H" in _ALL_FEATURES:
        dataset['hilbert_envelope'] = preprocessing.calculate_hilbert_envelope(dataset)

    if _GLOBAL_APPLY_WAVELET:
        dataset['value_original'] = dataset['value'].copy()

        dataset = preprocessing.apply_wavelet_denoising(
            dataset,
            wavelet=WAVELET_TYPE,
            mode=WAVELET_MODE,
            threshold_mode=WAVELET_THRESHOLD_MODE,
            noise_std=None,
            threshold_multiplier=WAVELET_THRESHOLD_MULTIPLIER,
            level=WAVELET_LEVEL
        )

    return dataset, new_accumulated_time, file_duration




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


def load_true_events(voltage, selections_dir=None):
    """Load true events from {selections_dir}/{voltage}/time_events.txt."""
    if selections_dir is None:
        selections_dir = "selections"
    events_file = f'{selections_dir}/{voltage}/time_events.txt'
    if not os.path.exists(events_file):
        raise FileNotFoundError(f"Events file does not exist: {events_file}")

    print(f"\nLoading true events from {events_file}...")
    events_df = pd.read_csv(
        events_file,
        header=None,
        names=['start', 'end'],
        sep=' ',
        dtype=str
    )

    events_df['start'] = pd.to_datetime(events_df['start'])
    events_df['end'] = pd.to_datetime(events_df['end'])

    print(f"Found {len(events_df)} true events")
    return events_df


def filter_true_events_by_polarity(dataset, true_events_df, polarity):
    """Filter true events by their polarity (negative or positive)."""
    if polarity not in ["negative", "positive"]:
        raise ValueError(f"polarity must be 'negative' or 'positive', got '{polarity}'")

    filtered_events = []

    for idx, event in true_events_df.iterrows():
        event_start = event['start']
        event_end = event['end']

        event_mask = (dataset.index >= event_start) & (dataset.index <= event_end)
        if 'value_original' in dataset.columns:
            event_data = dataset.loc[event_mask, 'value_original'].values
        else:
            event_data = dataset.loc[event_mask, 'value'].values

        if len(event_data) == 0:
            continue

        max_abs_idx = np.argmax(np.abs(event_data))
        max_abs_value = event_data[max_abs_idx]

        if polarity == "negative":
            if max_abs_value < 0:
                filtered_events.append(event)
        elif polarity == "positive":
            if max_abs_value >= 0:
                filtered_events.append(event)

    if len(filtered_events) == 0:
        return pd.DataFrame(columns=['start', 'end'])

    filtered_events_df = pd.DataFrame(filtered_events)
    return filtered_events_df


def calculate_metrics(predicted_events_df, true_events_df, event_type_label="All events"):
    """Calculate accuracy and false triggers metrics from true events perspective."""
    print("\n" + "=" * 60)
    print(f"Calculating metrics for {event_type_label}...")
    print("=" * 60)

    total_predicted_events = len(predicted_events_df)
    total_true_events = len(true_events_df)

    if total_true_events == 0:
        print("No true events found.")
        if total_predicted_events == 0:
            return 0, 0, 0
        else:
            print(f"All {total_predicted_events} predicted events are false triggers.")
            return 0, 0, total_predicted_events

    if total_predicted_events == 0:
        print("No predicted events to evaluate.")
        print(f"All {total_true_events} true events are missed.")
        return 0, total_true_events, 0

    correctly_predicted = 0
    missed = 0

    for _, true_event in true_events_df.iterrows():
        true_start = true_event['start']
        true_end = true_event['end']

        overlaps = False
        for _, pred_event in predicted_events_df.iterrows():
            pred_start = pred_event['start']
            pred_end = pred_event['end']

            if pred_start <= true_end and true_start <= pred_end:
                overlaps = True
                break

        if overlaps:
            correctly_predicted += 1
        else:
            missed += 1

    num_false_triggers = 0

    for _, pred_event in predicted_events_df.iterrows():
        pred_start = pred_event['start']
        pred_end = pred_event['end']

        overlaps = False
        for _, true_event in true_events_df.iterrows():
            true_start = true_event['start']
            true_end = true_event['end']

            if pred_start <= true_end and true_start <= pred_end:
                overlaps = True
                break

        if not overlaps:
            num_false_triggers += 1

    accuracy_rate = correctly_predicted / (correctly_predicted + missed) * 100 if (correctly_predicted + missed) > 0 else 0.0

    print(f"\nMetrics for {event_type_label}:")
    print(f"  Total predicted events: {total_predicted_events}")
    print(f"  Total true events: {total_true_events}")
    print(f"  Correctly predicted: {correctly_predicted}")
    print(f"  Missed: {missed}")
    print(f"  False triggers: {num_false_triggers}")
    print(f"  Accuracy rate: {accuracy_rate:.2f}% ({correctly_predicted}/{total_true_events})")

    return correctly_predicted, missed, num_false_triggers


def tag_false_triggers(predicted_events_df, file_true_events_df):
    """Add an 'is_false_trigger' boolean column to predicted_events_df."""
    result = predicted_events_df.copy()
    if file_true_events_df is None or len(file_true_events_df) == 0:
        result['is_false_trigger'] = False
        return result

    true_starts = file_true_events_df['start'].values
    true_ends   = file_true_events_df['end'].values

    def overlaps_any(pred_start, pred_end):
        return any(pred_start <= te and ts <= pred_end
                   for ts, te in zip(true_starts, true_ends))

    result['is_false_trigger'] = [
        not overlaps_any(row['start'], row['end'])
        for _, row in result.iterrows()
    ]
    return result


def calculate_border_refinement_rmse(refined_events_df, true_events_df, dataset_timestamps):
    """Calculate RMSE for border refinement (left and right borders separately)."""
    if len(refined_events_df) == 0 or len(true_events_df) == 0:
        return np.nan, np.nan, 0

    refined_events = [(pd.to_datetime(row['start']), pd.to_datetime(row['end']))
                      for _, row in refined_events_df.iterrows()]
    true_events = [(pd.to_datetime(row['start']), pd.to_datetime(row['end']))
                   for _, row in true_events_df.iterrows()]

    matches = match_events_efficiently(refined_events, true_events)

    if len(matches) == 0:
        return np.nan, np.nan, 0

    metrics = calculate_border_metrics(refined_events, true_events, matches, dataset_timestamps)

    return metrics['left_rmse'], metrics['right_rmse'], metrics['num_matched_events']


def save_predicted_events(events_df, output_dir):
    """Save predicted events to CSV file."""
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "predicted_events.csv")
    events_df[['start', 'end']].to_csv(
        output_path,
        sep=' ',
        header=False,
        index=False,
        date_format='%Y-%m-%dT%H:%M:%S.%f'
    )
    print(f"\nSaved predicted events to: {output_path}")


def save_border_refinement_data(dataset, op_timestamps, op_values, predicted_events_df,
                                 true_events_df, output_dir, file_index):
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

    output_path = os.path.join(output_dir, f"border_refinement_data_file_{file_index:04d}.npz")
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


def save_speed_stats(output_dir, total_time, num_files, time_per_step):
    """Save speed statistics to file."""
    stats_path = os.path.join(output_dir, "speed_stats.txt")
    with open(stats_path, 'w') as f:
        f.write("Model Speed Statistics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total files processed: {num_files}\n")
        f.write(f"Total time steps: {num_files * 200000}\n")
        f.write(f"Total time: {total_time:.4f} seconds\n")
        f.write(f"Time per time step: {time_per_step:.6e} seconds\n")
        f.write(f"Time per time step: {time_per_step * 1e6:.4f} microseconds\n")
        if total_time > 0:
            f.write(f"Throughput: {(num_files * 200000) / total_time:.2f} time steps/second\n")
        else:
            f.write(f"Throughput: N/A (total time is zero)\n")

    print(f"\nSaved speed statistics to: {stats_path}")


def save_metrics_to_file(output_dir, neg_false_triggers, neg_accuracy, neg_true_events_count,
                         pos_false_triggers, pos_accuracy, pos_true_events_count,
                         total_false_triggers, total_accuracy, total_true_events_count):
    """Save metrics to model_results.txt file."""
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "model_results.txt")

    with open(output_path, 'w') as f:
        f.write("Negative\n")
        f.write(f"False triggers: {neg_false_triggers}\n")
        if neg_true_events_count > 0:
            neg_accuracy_rate = neg_accuracy / neg_true_events_count * 100
            f.write(f"Accuracy rate: {neg_accuracy_rate:.2f}% ({neg_accuracy}/{neg_true_events_count})\n")
        else:
            f.write(f"Accuracy rate: N/A (0/{neg_true_events_count})\n")

        f.write("\nPositive\n")
        f.write(f"False triggers: {pos_false_triggers}\n")
        if pos_true_events_count > 0:
            pos_accuracy_rate = pos_accuracy / pos_true_events_count * 100
            f.write(f"Accuracy rate: {pos_accuracy_rate:.2f}% ({pos_accuracy}/{pos_true_events_count})\n")
        else:
            f.write(f"Accuracy rate: N/A (0/{pos_true_events_count})\n")

        f.write("\nAll\n")
        f.write(f"False triggers: {total_false_triggers}\n")
        if total_true_events_count > 0:
            total_accuracy_rate = total_accuracy / total_true_events_count * 100
            f.write(f"Accuracy rate: {total_accuracy_rate:.2f}% ({total_accuracy}/{total_true_events_count})\n")
        else:
            f.write(f"Accuracy rate: N/A (0/{total_true_events_count})\n")

    print(f"\nSaved metrics to: {output_path}")


def create_events_summary(stats_all, stats_positive, stats_negative, stats_dir, noise_std=None):
    """Create a text file with event statistics summary."""
    summary_path = os.path.join(stats_dir, "events_summary.txt")
    os.makedirs(stats_dir, exist_ok=True)

    groups = (
        ("all", stats_all),
        ("positive", stats_positive),
        ("negative", stats_negative),
    )

    def mean(values):
        return np.mean(values) if len(values) > 0 else float('nan')

    def std(values):
        return np.std(values) if len(values) > 1 else float('nan')

    def iqr(values):
        if len(values) < 4:
            return float('nan')
        return np.percentile(values, 75) - np.percentile(values, 25)

    def span(values):
        if len(values) == 0:
            return float('nan'), float('nan'), float('nan')
        lo = np.min(values)
        hi = np.max(values)
        return hi - lo, lo, hi

    def total(values):
        return np.sum(values) if len(values) > 0 else 0.0

    with open(summary_path, 'w') as f:
        f.write("amplitude, A\n")
        for name, stats in groups:
            f.write(f"  {name}: mean {mean(stats['amplitude']):.8f}, std {std(stats['amplitude']):.8f}\n")

        f.write("\nwidth, points\n")
        for name, stats in groups:
            width = stats['width_points']
            width_span, lo, hi = span(width)
            f.write(f"  {name}: mean {mean(width):.2f}, std {std(width):.2f}, range {width_span:.2f} ({lo:.2f} .. {hi:.2f})\n")

        f.write("\nsigned integral, pC\n")
        for name, stats in groups:
            f.write(f"  {name}: {total(stats['integral']):.8f}\n")

        f.write("\nabsolute integral, pC\n")
        for name, stats in groups:
            f.write(f"  {name}: {total([abs(x) for x in stats['integral']]):.8f}\n")

        f.write("\nintegral, pC\n")
        for name, stats in (("positive", stats_positive), ("negative", stats_negative)):
            values = stats['integral']
            if len(values) == 0:
                f.write(f"  {name}: none\n")
                continue
            charge_span, lo, hi = span(values)
            f.write(
                f"  {name}: mean |q| {mean([abs(x) for x in values]):.8f}, "
                f"iqr {iqr(values):.8f}, range {charge_span:.8f} ({lo:.8f} .. {hi:.8f})\n"
            )

        f.write("\nnoise, A\n")
        if noise_std is not None and not np.isnan(noise_std):
            f.write(f"  std {noise_std:.8f}\n")
        else:
            f.write("  std n/a\n")

        f.write("\nevents per file\n")
        for name, stats in groups:
            files = stats['file_index']
            n_events = len(files)
            n_files = len(set(files)) if n_events else 0
            per_file = n_events / n_files if n_files else float('nan')
            f.write(f"  {name}: {per_file:.2f} ({n_events} events, {n_files} files)\n")

        halfwave_flags = [hw for hw in stats_all['is_positive_halfwave'] if hw is not None]
        n_plus = sum(1 for hw in halfwave_flags if hw)
        n_minus = sum(1 for hw in halfwave_flags if not hw)
        signed = total(stats_all['integral'])
        absolute = total([abs(x) for x in stats_all['integral']])
        imbalance = signed / absolute if absolute != 0 else float('nan')

        f.write(f"\nN+ {n_plus}\n")
        f.write(f"N- {n_minus}\n")
        if np.isnan(imbalance):
            f.write("D n/a\n")
        else:
            f.write(f"D {imbalance:.8f}\n")

    print(f"Saved {summary_path}")


def visualize_predictions(dataset, predicted_events_df, batch_size=20000, output_dir="predictions_untouched", true_events_df=None, save_plots=True, file_index=None, op_timestamps=None, op_values=None, current_ylim=None, only_batches_with_events=False):
    """Visualize data with predicted events marked in colored vertical bands."""
    print("\n" + "=" * 60)
    print("Starting prediction visualization...")
    print("=" * 60)

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        print("Plot generation disabled (save_plots=False).")

    FIGURE_SIZE = (13, 9)
    CURRENT_LINE_COLOR = 'k'
    CURRENT_LINE_WIDTH = 3
    CURRENT_LINE_STYLE = '-'
    AXIS_LABEL_FONTSIZE = 20
    TICK_LABEL_FONTSIZE = 20
    TITLE_FONTSIZE = 18
    LEGEND_FONTSIZE = 20
    X_LABEL = "Время, нс"
    CURRENT_Y_LABEL = "Ток, А"
    CURRENT_LABEL = "Ток"
    OP_Y_LABEL = "Θ"
    OP_LABEL = "Θ"
    OP_LINE_COLOR = 'k'
    OP_LINE_WIDTH = 2
    OP_LINE_STYLE = '--'
    OP_LINE_ALPHA = 1.0
    LEGEND_LOCATION = 'upper left'
    GRID_LINESTYLE = '-'
    GRID_ALPHA = 0.7
    PULSE_COLOR = '0.2'
    FALSE_TRIGGER_COLOR = '0.75'
    EVENT_ALPHA = 0.55
    FALSE_TRIGGER_ALPHA = 0.85
    TRUE_EVENT_COLOR = '0.2'
    TRUE_EVENT_ALPHA = 0.55
    DPI = 100

    op_ylim = None
    if op_timestamps is not None and op_values is not None and len(op_values) > 0:
        op_min = np.min(op_values)
        op_max = np.max(op_values)
        op_range = op_max - op_min
        op_padding = op_range * 0.05 if op_range > 0 else 0.1
        op_ylim = (op_min - op_padding, op_max + op_padding)
        print(f"OP value range: [{op_min:.4f}, {op_max:.4f}], plot limits: [{op_ylim[0]:.4f}, {op_ylim[1]:.4f}]")

    show_true_events = true_events_df is not None and len(true_events_df) > 0

    global_time_reference = dataset.index[0]

    num_points = len(dataset)
    num_batches = (num_points + batch_size - 1) // batch_size
    print(f"Processing {num_points} data points in {num_batches} batches...")

    batch_count = 0
    saved_plot_count = 0

    TIME_PER_STEP = sample_period_s()

    if current_ylim is None:
        current_ylim = (-0.1 / 50, 0.3 / 50)

    file_event_stats = {
        'all': [],
        'positive': [],
        'negative': []
    }

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_points)

        batch_data = dataset.iloc[start_idx:end_idx]
        if 'value_original' in batch_data.columns:
            batch_values = batch_data['value_original'].values
        else:
            batch_values = batch_data['value'].values
        batch_timestamps = batch_data.index

        NS_PER_POINT = SAMPLE_PERIOD_NS
        batch_t_ns = np.arange(start_idx, end_idx) * NS_PER_POINT

        batch_start_time = batch_timestamps[0]
        batch_end_time = batch_timestamps[-1]

        has_predicted_events = False
        has_true_events = False
        if save_plots and (show_true_events or only_batches_with_events):
            if predicted_events_df is not None and len(predicted_events_df) > 0 and 'start' in predicted_events_df.columns:
                for _, event in predicted_events_df.iterrows():
                    event_start = event['start']
                    event_end = event['end']
                    if batch_start_time <= event_end and batch_end_time >= event_start:
                        has_predicted_events = True
                        break

            if show_true_events:
                for _, event in true_events_df.iterrows():
                    event_start = event['start']
                    event_end = event['end']
                    if batch_start_time <= event_end and batch_end_time >= event_start:
                        has_true_events = True
                        break

            if show_true_events and not (has_predicted_events or has_true_events):
                continue
            if only_batches_with_events and not has_predicted_events and not has_true_events:
                continue

        if save_plots:
            if show_true_events:
                fig, (ax_pred, ax_true) = plt.subplots(2, 1, figsize=FIGURE_SIZE, sharex=True)
            else:
                fig, ax_pred = plt.subplots(1, 1, figsize=FIGURE_SIZE)
                ax_true = None

            ax_pred.plot(batch_t_ns, batch_values, color=CURRENT_LINE_COLOR, linestyle=CURRENT_LINE_STYLE,
                    linewidth=CURRENT_LINE_WIDTH, label=CURRENT_LABEL)
            if ax_true is not None:
                ax_true.plot(batch_t_ns, batch_values, color=CURRENT_LINE_COLOR, linestyle=CURRENT_LINE_STYLE,
                        linewidth=CURRENT_LINE_WIDTH, label=CURRENT_LABEL)

            ax_op = None
            if op_timestamps is not None and op_values is not None and len(op_values) > 0:
                op_timestamps_pd = pd.to_datetime(op_timestamps)

                op_mask = (op_timestamps_pd >= batch_start_time) & (op_timestamps_pd <= batch_end_time)
                batch_op_values = op_values[op_mask]
                batch_op_timestamps = op_timestamps_pd[op_mask]

                if len(batch_op_values) > 0:
                    batch_op_t_ns = dataset.index.searchsorted(batch_op_timestamps) * NS_PER_POINT

                    ax_op = ax_pred.twinx()
                    ax_op.plot(batch_op_t_ns, batch_op_values, color=OP_LINE_COLOR, linestyle=OP_LINE_STYLE,
                              linewidth=OP_LINE_WIDTH, alpha=OP_LINE_ALPHA, label=OP_LABEL)
                    ax_op.set_ylabel(OP_Y_LABEL, fontsize=AXIS_LABEL_FONTSIZE, color=OP_LINE_COLOR,
                                     rotation=0, va='center', labelpad=15, fontstyle='italic')
                    ax_op.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE, labelcolor=OP_LINE_COLOR)

                    if op_ylim is not None:
                        ax_op.set_ylim(op_ylim)

            pulse_legend_added = False
            false_trigger_legend_added = False

            for _, event in predicted_events_df.iterrows():
                event_start = event['start']
                event_end = event['end']

                if batch_start_time <= event_end and batch_end_time >= event_start:
                    event_start_ns = dataset.index.searchsorted(event_start) * NS_PER_POINT
                    event_end_ns = dataset.index.searchsorted(event_end) * NS_PER_POINT

                    event_start_ns = max(event_start_ns, batch_t_ns[0])
                    event_end_ns = min(event_end_ns, batch_t_ns[-1])

                    if event.get('is_false_trigger', False):
                        color = FALSE_TRIGGER_COLOR
                        alpha = FALSE_TRIGGER_ALPHA
                        label = 'Ложные срабатывания' if not false_trigger_legend_added else ''
                        if label:
                            false_trigger_legend_added = True
                    else:
                        color = PULSE_COLOR
                        alpha = EVENT_ALPHA
                        label = 'Импульсы' if not pulse_legend_added else ''
                        if label:
                            pulse_legend_added = True

                    ax_pred.axvspan(event_start_ns, event_end_ns,
                              alpha=alpha,
                              color=color,
                              label=label)

            if show_true_events and ax_true is not None:
                true_legend_added = False
                for _, event in true_events_df.iterrows():
                    event_start = event['start']
                    event_end = event['end']

                    if batch_start_time <= event_end and batch_end_time >= event_start:
                        event_start_ns = dataset.index.searchsorted(event_start) * NS_PER_POINT
                        event_end_ns = dataset.index.searchsorted(event_end) * NS_PER_POINT

                        event_start_ns = max(event_start_ns, batch_t_ns[0])
                        event_end_ns = min(event_end_ns, batch_t_ns[-1])

                        label = 'Истинные импульсы' if not true_legend_added else ''
                        ax_true.axvspan(event_start_ns, event_end_ns,
                                  alpha=TRUE_EVENT_ALPHA,
                                  color=TRUE_EVENT_COLOR,
                                  label=label)
                        if label:
                            true_legend_added = True

            if ax_true is None:
                ax_pred.set_xlabel(X_LABEL, fontsize=AXIS_LABEL_FONTSIZE)
            ax_pred.set_ylabel(CURRENT_Y_LABEL, fontsize=AXIS_LABEL_FONTSIZE)
            ax_pred.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)
            ax_pred.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE)

            legend_handles, legend_labels = ax_pred.get_legend_handles_labels()
            if ax_op is not None:
                op_handles, op_labels = ax_op.get_legend_handles_labels()
                legend_handles += op_handles
                legend_labels += op_labels
            legend_ax = ax_op if ax_op is not None else ax_pred
            pred_legend = legend_ax.legend(legend_handles, legend_labels, loc=LEGEND_LOCATION,
                                           fontsize=LEGEND_FONTSIZE)
            for legend_text in pred_legend.get_texts():
                if legend_text.get_text() == OP_LABEL:
                    legend_text.set_fontstyle('italic')

            ax_pred.set_ylim(current_ylim)
            ax_pred.grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA, which="both")
            ax_pred.set_title(f'Участок {batch_idx + 1}/{num_batches} - Предсказанные импульсы', fontsize=TITLE_FONTSIZE)

            if ax_true is not None:
                ax_true.set_xlabel(X_LABEL, fontsize=AXIS_LABEL_FONTSIZE)
                ax_true.set_ylabel(CURRENT_Y_LABEL, fontsize=AXIS_LABEL_FONTSIZE)
                ax_true.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)
                ax_true.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE)
                ax_true.legend(loc=LEGEND_LOCATION, fontsize=LEGEND_FONTSIZE)
                ax_true.set_ylim(current_ylim)
                ax_true.grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA, which="both")
                ax_true.set_title('Истинные импульсы', fontsize=TITLE_FONTSIZE)

            plt.tight_layout()

            if file_index is not None:
                if show_true_events:
                    saved_plot_count += 1
                    plot_filename = f"{output_dir}/file_{file_index}_batch_{saved_plot_count}.png"
                else:
                    plot_filename = f"{output_dir}/file_{file_index}_batch_{batch_idx + 1}.png"
            else:
                if show_true_events:
                    saved_plot_count += 1
                    plot_filename = f"{output_dir}/batch_{saved_plot_count}.png"
                else:
                    plot_filename = f"{output_dir}/batch_{batch_idx + 1}.png"
            plt.savefig(plot_filename, dpi=DPI, bbox_inches='tight')
            plt.close(fig)

        batch_start_time = batch_timestamps[0]
        batch_end_time = batch_timestamps[-1]

        has_raw_data = 'raw_i' in batch_data.columns and 'raw_t' in batch_data.columns and 'capacitor_current' in batch_data.columns
        if has_raw_data:
            batch_raw_i = batch_data['raw_i'].values
            batch_raw_t = batch_data['raw_t'].values
            batch_capacitor_current = batch_data['capacitor_current'].values
            batch_raw_v = batch_data['raw_v'].values if 'raw_v' in batch_data.columns else None
            voltage_shift = get_voltage_shift(TARGET_VOLTAGE)
            batch_ii = batch_raw_i - batch_capacitor_current - voltage_shift
        else:
            batch_ii = batch_values
            batch_raw_v = None

        has_events = False
        for _, event in predicted_events_df.iterrows():
            event_start = event['start']
            event_end = event['end']
            if batch_start_time <= event_end and batch_end_time >= event_start:
                has_events = True

                event_mask = (batch_timestamps >= event_start) & (batch_timestamps <= event_end)
                event_indices = np.where(event_mask)[0]

                if len(event_indices) > 0:
                    event_start_idx = event_indices[0]
                    event_end_idx = event_indices[-1]

                    if has_raw_data:
                        event_region_i = batch_raw_i[event_start_idx:event_end_idx+1]
                        event_region_t = batch_raw_t[event_start_idx:event_end_idx+1]
                        event_region_ci = batch_capacitor_current[event_start_idx:event_end_idx+1]
                        event_region_ii = event_region_i - event_region_ci - voltage_shift
                        event_region_v = batch_raw_v[event_start_idx:event_end_idx+1] if batch_raw_v is not None else None
                    else:
                        event_region_ii = batch_values[event_start_idx:event_end_idx+1]
                        event_region_t = batch_t_ns[event_start_idx:event_end_idx+1]
                        event_region_v = None

                    if len(event_region_ii) > 0:
                        max_val = np.max(event_region_ii)
                        min_val = np.min(event_region_ii)

                        if abs(max_val) >= abs(min_val):
                            event_amplitude = max_val
                        else:
                            event_amplitude = min_val

                        event_integral = np.trapz(event_region_ii, event_region_t) * TIME_PER_STEP * 1e12

                        is_positive_halfwave = None
                        if event_region_v is not None and len(event_region_v) > 0:
                            positive_voltage_count = np.sum(event_region_v > 0)
                            total_voltage_count = len(event_region_v)
                            is_positive_halfwave = bool((positive_voltage_count / total_voltage_count) > 0.5)
                        if event_amplitude >= 0:
                            max_current = np.max(event_region_ii)
                            peak_idx = np.argmax(event_region_ii)
                        else:
                            max_current = np.min(event_region_ii)
                            peak_idx = np.argmin(event_region_ii)

                        peak_position = peak_idx / len(event_region_ii) if len(event_region_ii) > 0 else 0.5
                        width_points = len(event_region_ii)
                        start_current = event_region_ii[0]
                        end_current = event_region_ii[-1]
                        event_type = event.get('type', 'unknown')

                        event_stat = {
                            'width_points': width_points,
                            'amplitude': event_amplitude,
                            'integral': event_integral,
                            'start_current': start_current,
                            'end_current': end_current,
                            'max_current': max_current,
                            'peak_position': peak_position,
                            'file_index': file_index if file_index is not None else 0,
                            'is_positive': (event_amplitude >= 0),
                            'is_positive_halfwave': is_positive_halfwave
                        }

                        file_event_stats['all'].append(event_stat)
                        if event_type == 'positive' or event_amplitude >= 0:
                            file_event_stats['positive'].append(event_stat)
                        if event_type == 'negative' or event_amplitude < 0:
                            file_event_stats['negative'].append(event_stat)

        batch_count += 1
        if batch_count % 10 == 0:
            if show_true_events:
                print(f"Processed {batch_count} batches with events...")
            else:
                print(f"Processed {batch_count}/{num_batches} batches...")

    if save_plots:
        if show_true_events:
            print(f"\nSaved {saved_plot_count} plots to {output_dir}")
        else:
            print(f"\nSaved {batch_count} plots to {output_dir}")

    events_all = file_event_stats['all']
    events_positive = file_event_stats['positive']
    events_negative = file_event_stats['negative']

    file_stats_summary = {
        'all': {
            'width_points': [e['width_points'] for e in events_all],
            'amplitude': [e['amplitude'] for e in events_all],
            'integral': [e['integral'] for e in events_all],
            'start_current': [e['start_current'] for e in events_all],
            'end_current': [e['end_current'] for e in events_all],
            'max_current': [e['max_current'] for e in events_all],
            'peak_position': [e['peak_position'] for e in events_all],
            'current_diff': [e['end_current'] - e['start_current'] for e in events_all],
            'file_index': [e['file_index'] for e in events_all],
            'is_positive_halfwave': [e.get('is_positive_halfwave', None) for e in events_all]
        },
        'positive': {
            'width_points': [e['width_points'] for e in events_positive],
            'amplitude': [e['amplitude'] for e in events_positive],
            'integral': [e['integral'] for e in events_positive],
            'start_current': [e['start_current'] for e in events_positive],
            'end_current': [e['end_current'] for e in events_positive],
            'max_current': [e['max_current'] for e in events_positive],
            'peak_position': [e['peak_position'] for e in events_positive],
            'current_diff': [e['end_current'] - e['start_current'] for e in events_positive],
            'file_index': [e['file_index'] for e in events_positive],
            'is_positive_halfwave': [e.get('is_positive_halfwave', None) for e in events_positive]
        },
        'negative': {
            'width_points': [e['width_points'] for e in events_negative],
            'amplitude': [e['amplitude'] for e in events_negative],
            'integral': [e['integral'] for e in events_negative],
            'start_current': [e['start_current'] for e in events_negative],
            'end_current': [e['end_current'] for e in events_negative],
            'max_current': [e['max_current'] for e in events_negative],
            'peak_position': [e['peak_position'] for e in events_negative],
            'current_diff': [e['end_current'] - e['start_current'] for e in events_negative],
            'file_index': [e['file_index'] for e in events_negative],
            'is_positive_halfwave': [e.get('is_positive_halfwave', None) for e in events_negative]
        }
    }

    return file_stats_summary


def main():
    print("=" * 60)
    print("EventDetector Prediction Script")
    print("=" * 60)

    if not os.path.exists(MODEL_DIR):
        raise ValueError(f"MODEL_DIR does not exist: {MODEL_DIR}")

    def _resolve_model_path(subdir_name, label):
        candidate = os.path.join(MODEL_DIR, subdir_name)
        if not os.path.exists(candidate):
            raise ValueError(f"{label} not found: {candidate}")
        print(f"  {label}: loaded from {candidate}")
        return candidate

    neg_model_path = _resolve_model_path(
        "eventdetector_neg_training_results", "Negative model"
    )
    pos_model_path = _resolve_model_path(
        "eventdetector_pos_training_results", "Positive model"
    )

    if BORDER_REFINEMENT_ML_MODELS_DIR:
        refinement_models_dir = BORDER_REFINEMENT_ML_MODELS_DIR
        print(f"  Refinement model: using explicit override ({refinement_models_dir})")
    else:
        refinement_models_dir = _resolve_model_path(
            "border_refinement_ml_models", "Refinement model"
        )

    print(f"\nModel directory: {MODEL_DIR}")
    print(f"Target voltage: {TARGET_VOLTAGE}")
    print(f"Reference voltage: {REFERENCE_VOLTAGE}")
    print(f"Use annotation: {USE_ANNOTATION}")
    print(f"Save plots: {SAVE_PLOTS}")
    print(f"Apply refinement: {APPLY_REFINEMENT}")
    print(f"\nNeg model features: {NEG_FEATURES}, wavelet={NEG_APPLY_WAVELET_FILTER}, denoised_feat={NEG_USE_DENOISED_AS_FEATURE}")
    print(f"Pos model features: {POS_FEATURES}, wavelet={POS_APPLY_WAVELET_FILTER}, denoised_feat={POS_USE_DENOISED_AS_FEATURE}")
    print(f"Global wavelet applied: {_GLOBAL_APPLY_WAVELET}")

    for label, use_den, apply_wv in [("NEG", NEG_USE_DENOISED_AS_FEATURE, NEG_APPLY_WAVELET_FILTER),
                                      ("POS", POS_USE_DENOISED_AS_FEATURE, POS_APPLY_WAVELET_FILTER)]:
        if use_den and not apply_wv:
            print(f"Warning: {label}_USE_DENOISED_AS_FEATURE is True but {label}_APPLY_WAVELET_FILTER is False.")
            print(f"  {label}_USE_DENOISED_AS_FEATURE will be ignored.")

    ml_model_left    = None
    ml_model_right   = None
    ml_feature_names = None
    ml_model_type    = _BR_MODEL_TYPE_RF
    ml_feature_scaler = None

    if APPLY_REFINEMENT:
        print("\n" + "=" * 60)
        print("Loading ML models for border refinement...")
        print("=" * 60)
        try:
            (ml_model_left, ml_model_right, ml_feature_names,
             _ml_target_type, ml_model_type, ml_feature_scaler) = load_border_refinement_models(refinement_models_dir)
            print(f"Loaded border refinement models from {refinement_models_dir}")
        except FileNotFoundError as e:
            print(f"Error: Could not load ML models: {e}")
            print("Border refinement will be disabled.")
            ml_model_left     = None
            ml_model_right    = None
            ml_feature_names  = None
            ml_model_type     = _BR_MODEL_TYPE_RF
            ml_feature_scaler = None

    if USE_ANNOTATION:
        if ANNOTATION_DATA_DIR and ANNOTATION_DATA_DIR.strip():
            data_dir = ANNOTATION_DATA_DIR
        else:
            data_dir = "example_labeled_data"
        directory = f"{data_dir}/{TARGET_VOLTAGE}"
    else:
        directory = f"example_unlabeled_data/{TARGET_VOLTAGE}"
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")

    model_dir_name = os.path.basename(MODEL_DIR.rstrip('/'))
    output_dir = f"predictions_untouched/{model_dir_name}"
    os.makedirs(output_dir, exist_ok=True)

    global_time_reference = datetime(2024, 1, 1, 0, 0, 0)
    accumulated_time = 0.0

    aggregated_stats = {
        'all': {
            'width_points': [],
            'amplitude': [],
            'integral': [],
            'start_current': [],
            'end_current': [],
            'max_current': [],
            'peak_position': [],
            'current_diff': [],
            'file_index': [],
            'is_positive_halfwave': []
        },
        'positive': {
            'width_points': [],
            'amplitude': [],
            'integral': [],
            'start_current': [],
            'end_current': [],
            'max_current': [],
            'peak_position': [],
            'current_diff': [],
            'file_index': [],
            'is_positive_halfwave': []
        },
        'negative': {
            'width_points': [],
            'amplitude': [],
            'integral': [],
            'start_current': [],
            'end_current': [],
            'max_current': [],
            'peak_position': [],
            'current_diff': [],
            'file_index': [],
            'is_positive_halfwave': []
        }
    }

    all_combined_events = []

    total_neg_correctly_predicted = 0
    total_neg_missed = 0
    total_pos_correctly_predicted = 0
    total_pos_missed = 0
    total_neg_false_triggers = 0
    total_pos_false_triggers = 0
    total_neg_true_events = 0
    total_pos_true_events = 0

    true_events_polarity = {}

    first_file_dataset_timestamps = None

    true_events_df = None
    if USE_ANNOTATION:
        print("\n" + "=" * 60)
        print("Loading true events for validation")
        print("=" * 60)
        if ANNOTATION_SELECTIONS_DIR and ANNOTATION_SELECTIONS_DIR.strip():
            selections_dir = ANNOTATION_SELECTIONS_DIR
        else:
            selections_dir = "selections"
        true_events_df = load_true_events(TARGET_VOLTAGE, selections_dir=selections_dir)

    reader = NpzFileIterator(directory)
    file_count = 0

    speed_timer_start = time.perf_counter()

    print("\n" + "=" * 60)
    print("Processing files sequentially...")
    print("=" * 60)

    try:
        while True:
            file_count += 1
            print(f"\nProcessing file {file_count}...")

            file_data = next(reader)

            dataset, accumulated_time, file_duration = load_data_from_single_file(
                TARGET_VOLTAGE, file_data, global_time_reference, accumulated_time
            )

            if dataset is None or len(dataset) == 0:
                continue

            print(f"  Loaded {len(dataset)} data points from file {file_count}")

            if first_file_dataset_timestamps is None:
                first_file_dataset_timestamps = dataset.index

            neg_dataset = prepare_model_dataset(dataset, NEG_FEATURES, NEG_USE_DENOISED_AS_FEATURE, NEG_APPLY_WAVELET_FILTER)
            pos_dataset = prepare_model_dataset(dataset, POS_FEATURES, POS_USE_DENOISED_AS_FEATURE, POS_APPLY_WAVELET_FILTER)
            print(f"  Neg model columns: {list(neg_dataset.columns)}")
            print(f"  Pos model columns: {list(pos_dataset.columns)}")

            neg_events, neg_predicted_op, neg_filtered_predicted_op, neg_timestamps = predict_with_model(
                dataset=neg_dataset,
                model_path=neg_model_path,
                mode="Neg"
            )

            if APPLY_REFINEMENT and ml_model_left is not None:
                print("\nRefining negative event boundaries using ML models...")
                br_current = dataset['value_original'].values if 'value_original' in dataset.columns else dataset['value'].values
                neg_events_refined = refine_event_boundaries_ml(
                    predicted_events=neg_events,
                    op_timestamps=neg_timestamps,
                    op_values=neg_filtered_predicted_op,
                    dataset_timestamps=dataset.index,
                    current_values=br_current,
                    model_left=ml_model_left,
                    model_right=ml_model_right,
                    feature_names=ml_feature_names,
                    model_type=ml_model_type,
                    feature_scaler=ml_feature_scaler
                )
            else:
                print("\nSkipping border refinement for negative events (APPLY_REFINEMENT=False or models not loaded)")
                neg_events_refined = neg_events

            pos_events, pos_predicted_op, pos_filtered_predicted_op, pos_timestamps = predict_with_model(
                dataset=pos_dataset,
                model_path=pos_model_path,
                mode="Pos"
            )

            if APPLY_REFINEMENT and ml_model_left is not None:
                print("\nRefining positive event boundaries using ML models...")
                br_current = dataset['value_original'].values if 'value_original' in dataset.columns else dataset['value'].values
                pos_events_refined = refine_event_boundaries_ml(
                    predicted_events=pos_events,
                    op_timestamps=pos_timestamps,
                    op_values=pos_filtered_predicted_op,
                    dataset_timestamps=dataset.index,
                    current_values=br_current,
                    model_left=ml_model_left,
                    model_right=ml_model_right,
                    feature_names=ml_feature_names,
                    model_type=ml_model_type,
                    feature_scaler=ml_feature_scaler
                )
            else:
                print("\nSkipping border refinement for positive events (APPLY_REFINEMENT=False or models not loaded)")
                pos_events_refined = pos_events

            combined_events_df = combine_predicted_events(neg_events_refined, pos_events_refined)

            all_combined_events.append(combined_events_df)

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

                op_df['timestamp_rounded'] = op_df['timestamp'].dt.round('1ms')

                op_df_grouped = op_df.groupby('timestamp_rounded')['op_value'].max().reset_index()

                combined_op_timestamps = op_df_grouped['timestamp_rounded'].values
                combined_op_values = op_df_grouped['op_value'].values

                sort_indices = np.argsort(combined_op_timestamps)
                combined_op_timestamps = combined_op_timestamps[sort_indices]
                combined_op_values = combined_op_values[sort_indices]

            if USE_ANNOTATION and true_events_df is not None:
                file_start_time = dataset.index[0]
                file_end_time   = dataset.index[-1]
                file_true_events_window = true_events_df[
                    (true_events_df['start'] <= file_end_time) &
                    (true_events_df['end']   >= file_start_time)
                ]
                combined_events_df = tag_false_triggers(combined_events_df, file_true_events_window)

            plot_current_ylim = None

            file_stats_summary = visualize_predictions(
                dataset=dataset,
                predicted_events_df=combined_events_df,
                batch_size=1500,
                output_dir=output_dir,
                true_events_df=true_events_df,
                save_plots=SAVE_PLOTS,
                file_index=file_count,
                op_timestamps=combined_op_timestamps,
                op_values=combined_op_values,
                current_ylim=plot_current_ylim,
                only_batches_with_events=SAVE_PLOTS_ONLY_WITH_EVENTS,
            )

            for stat_type in ['all', 'positive', 'negative']:
                for key in aggregated_stats[stat_type].keys():
                    aggregated_stats[stat_type][key].extend(file_stats_summary[stat_type][key])

            stats_all = aggregated_stats['all']
            stats_positive = aggregated_stats['positive']
            stats_negative = aggregated_stats['negative']
            create_events_summary(stats_all, stats_positive, stats_negative, output_dir, noise_std=None)
            print(f"  Updated events_summary.txt after file {file_count}")

            if USE_ANNOTATION and len(combined_events_df) > 0:
                file_start_time = dataset.index[0]
                file_end_time = dataset.index[-1]
                file_true_events = true_events_df[
                    (true_events_df['start'] <= file_end_time) & (true_events_df['end'] >= file_start_time)
                ].copy()

                for _, event in file_true_events.iterrows():
                    event_key = (event['start'], event['end'])
                    if event_key not in true_events_polarity:
                        event_mask = (dataset.index >= event['start']) & (dataset.index <= event['end'])
                        if 'value_original' in dataset.columns:
                            event_data = dataset.loc[event_mask, 'value_original'].values
                        else:
                            event_data = dataset.loc[event_mask, 'value'].values
                        if len(event_data) > 0:
                            max_abs_idx = np.argmax(np.abs(event_data))
                            max_abs_value = event_data[max_abs_idx]
                            true_events_polarity[event_key] = 'negative' if max_abs_value < 0 else 'positive'

                if len(file_true_events) > 0:
                    neg_true_events_df = filter_true_events_by_polarity(
                        dataset, file_true_events, "negative"
                    )
                    pos_true_events_df = filter_true_events_by_polarity(
                        dataset, file_true_events, "positive"
                    )
                else:
                    neg_true_events_df = pd.DataFrame(columns=['start', 'end'])
                    pos_true_events_df = pd.DataFrame(columns=['start', 'end'])

                neg_events_df = combined_events_df[combined_events_df['type'] == 'negative']
                pos_events_df = combined_events_df[combined_events_df['type'] == 'positive']

                neg_correctly_predicted, neg_missed, neg_false_triggers = calculate_metrics(
                    neg_events_df, neg_true_events_df, f"Negative events (file {file_count})"
                )
                pos_correctly_predicted, pos_missed, pos_false_triggers = calculate_metrics(
                    pos_events_df, pos_true_events_df, f"Positive events (file {file_count})"
                )

                total_neg_correctly_predicted += neg_correctly_predicted
                total_neg_missed += neg_missed
                total_pos_correctly_predicted += pos_correctly_predicted
                total_pos_missed += pos_missed
                total_neg_false_triggers += neg_false_triggers
                total_pos_false_triggers += pos_false_triggers
                total_neg_true_events += len(neg_true_events_df)
                total_pos_true_events += len(pos_true_events_df)

            if USE_ANNOTATION:
                border_refinement_dir = "border_refinement_set"
                os.makedirs(border_refinement_dir, exist_ok=True)

                file_true_events_for_save = None
                if true_events_df is not None:
                    file_start_time = dataset.index[0]
                    file_end_time = dataset.index[-1]
                    file_true_events_for_save = true_events_df[
                        (true_events_df['start'] <= file_end_time) & (true_events_df['end'] >= file_start_time)
                    ].copy()

                op_ts = combined_op_timestamps if 'combined_op_timestamps' in locals() and combined_op_timestamps is not None else None
                op_vals = combined_op_values if 'combined_op_values' in locals() and combined_op_values is not None else None

                save_border_refinement_data(
                    dataset=dataset,
                    op_timestamps=op_ts,
                    op_values=op_vals,
                    predicted_events_df=combined_events_df,
                    true_events_df=file_true_events_for_save,
                    output_dir=border_refinement_dir,
                    file_index=file_count
                )

            del neg_dataset
            del pos_dataset
            del dataset
            del neg_events
            del pos_events
            del neg_events_refined
            del pos_events_refined
            del neg_predicted_op
            del pos_predicted_op
            del neg_filtered_predicted_op
            del pos_filtered_predicted_op
            del neg_timestamps
            del pos_timestamps
            del combined_events_df
            del file_stats_summary
            gc.collect()

            events_count = len(all_combined_events[-1]) if len(all_combined_events) > 0 else 0
            print(f"  File {file_count} processed. Found {events_count} events.")

    except StopIteration:
        pass

    speed_timer_end = time.perf_counter()

    print(f"\nProcessed {file_count} files total.")

    if file_count > 0:
        total_time = speed_timer_end - speed_timer_start
        total_time_steps = file_count * 200000
        time_per_step = total_time / total_time_steps

        print("\n" + "=" * 60)
        print("Model Speed Statistics")
        print("=" * 60)
        print(f"Total files processed: {file_count}")
        print(f"Total time steps: {total_time_steps}")
        print(f"Total time: {total_time:.4f} seconds")
        print(f"Time per time step: {time_per_step:.6e} seconds")
        print(f"Time per time step: {time_per_step * 1e6:.4f} microseconds")
        if total_time > 0:
            throughput = (total_time_steps / total_time)
            print(f"Throughput: {throughput:.2f} time steps/second")
        else:
            print("Throughput: N/A (total time is zero)")
        print("=" * 60)

        save_speed_stats(output_dir, total_time, file_count, time_per_step)

    print("\n" + "=" * 60)
    print("Combining predicted events from all files")
    print("=" * 60)

    if len(all_combined_events) > 0:
        final_combined_events_df = pd.concat(all_combined_events, ignore_index=True)
        final_combined_events_df = final_combined_events_df.sort_values('start').reset_index(drop=True)
    else:
        final_combined_events_df = pd.DataFrame(columns=['start', 'end', 'type'])

    total_correctly_predicted = None
    total_missed = None
    num_false_triggers = None
    total_true_events = None
    if USE_ANNOTATION:
        print("\n" + "=" * 60)
        print("Final metrics summary (aggregated from all files)")
        print("=" * 60)

        total_correctly_predicted = total_neg_correctly_predicted + total_pos_correctly_predicted
        total_missed = total_neg_missed + total_pos_missed
        num_false_triggers = total_neg_false_triggers + total_pos_false_triggers
        total_true_events = len(true_events_df)

        border_refinement_left_rmse = None
        border_refinement_right_rmse = None
        border_refinement_matched_events = None
        if APPLY_REFINEMENT and ml_model_left is not None:
            print("\n" + "=" * 60)
            print("Calculating Border Refinement RMSE")
            print("=" * 60)

            if first_file_dataset_timestamps is not None:
                dataset_timestamps_for_rmse = first_file_dataset_timestamps
            else:
                TIME_PER_STEP = sample_period_s()
                synthetic_start = pd.to_datetime('2024-01-01 00:00:00')
                dataset_timestamps_for_rmse = pd.date_range(
                    start=synthetic_start,
                    periods=1000,
                    freq=pd.Timedelta(seconds=TIME_PER_STEP)
                )

            if len(final_combined_events_df) > 0 and len(true_events_df) > 0:
                border_refinement_left_rmse, border_refinement_right_rmse, border_refinement_matched_events = \
                    calculate_border_refinement_rmse(
                        refined_events_df=final_combined_events_df,
                        true_events_df=true_events_df,
                        dataset_timestamps=dataset_timestamps_for_rmse
                    )

                print(f"Matched events for border refinement: {border_refinement_matched_events}")
                if not np.isnan(border_refinement_left_rmse):
                    print(f"Left border RMSE: {border_refinement_left_rmse:.2f} time steps")
                    print(f"Right border RMSE: {border_refinement_right_rmse:.2f} time steps")
                else:
                    print("Border refinement RMSE: N/A (no matched events)")
            else:
                print("Border refinement RMSE: N/A (no refined events or true events)")

        if TEST_AGAINST_ANNOTATION_OUTPUT_DIR and TEST_AGAINST_ANNOTATION_OUTPUT_DIR.strip():
            save_metrics_to_file(
                output_dir=TEST_AGAINST_ANNOTATION_OUTPUT_DIR,
                neg_false_triggers=total_neg_false_triggers,
                neg_accuracy=total_neg_correctly_predicted,
                neg_true_events_count=total_neg_true_events,
                pos_false_triggers=total_pos_false_triggers,
                pos_accuracy=total_pos_correctly_predicted,
                pos_true_events_count=total_pos_true_events,
                total_false_triggers=num_false_triggers,
                total_accuracy=total_correctly_predicted,
                total_true_events_count=len(true_events_df)
            )
    else:
        print(f"Total predicted events across all files: {len(final_combined_events_df)}")

    print("\n" + "=" * 60)
    print("Saving predicted events")
    print("=" * 60)
    save_predicted_events(final_combined_events_df, output_dir)

    print("\n" + "=" * 60)
    print("Processing event statistics")
    print("=" * 60)

    stats_all = aggregated_stats['all']
    stats_positive = aggregated_stats['positive']
    stats_negative = aggregated_stats['negative']

    create_events_summary(stats_all, stats_positive, stats_negative, output_dir, noise_std=None)

    print(f"\n{output_dir}/predicted_events.csv")
    print(f"{output_dir}/events_summary.txt")
    if USE_ANNOTATION:
        print("\nMetrics Summary:")
        print("\nNegative Events:")
        if total_neg_true_events > 0:
            neg_accuracy_rate = (total_neg_correctly_predicted / total_neg_true_events) * 100
            print(f"  - Accuracy: {neg_accuracy_rate:.2f}% ({total_neg_correctly_predicted}/{total_neg_true_events})")
            print(f"  - Missed: {total_neg_missed}")
        else:
            print(f"  - Accuracy: N/A (0/{total_neg_true_events})")
            print(f"  - Missed: {total_neg_missed}")
        print(f"  - False triggers: {total_neg_false_triggers}")
        neg_recall = neg_accuracy_rate if total_neg_true_events > 0 else float('nan')
        neg_precision_denom = total_neg_correctly_predicted + total_neg_false_triggers
        neg_precision = (total_neg_correctly_predicted / neg_precision_denom * 100) if neg_precision_denom > 0 else float('nan')
        print(f"  - Recall: {neg_recall:.2f}%") if total_neg_true_events > 0 else print(f"  - Recall: N/A")
        print(f"  - Precision: {neg_precision:.2f}% ({total_neg_correctly_predicted}/{neg_precision_denom})") if neg_precision_denom > 0 else print(f"  - Precision: N/A")

        print("\nPositive Events:")
        if total_pos_true_events > 0:
            pos_accuracy_rate = (total_pos_correctly_predicted / total_pos_true_events) * 100
            print(f"  - Accuracy: {pos_accuracy_rate:.2f}% ({total_pos_correctly_predicted}/{total_pos_true_events})")
            print(f"  - Missed: {total_pos_missed}")
        else:
            print(f"  - Accuracy: N/A (0/{total_pos_true_events})")
            print(f"  - Missed: {total_pos_missed}")
        print(f"  - False triggers: {total_pos_false_triggers}")
        pos_recall = pos_accuracy_rate if total_pos_true_events > 0 else float('nan')
        pos_precision_denom = total_pos_correctly_predicted + total_pos_false_triggers
        pos_precision = (total_pos_correctly_predicted / pos_precision_denom * 100) if pos_precision_denom > 0 else float('nan')
        print(f"  - Recall: {pos_recall:.2f}%") if total_pos_true_events > 0 else print(f"  - Recall: N/A")
        print(f"  - Precision: {pos_precision:.2f}% ({total_pos_correctly_predicted}/{pos_precision_denom})") if pos_precision_denom > 0 else print(f"  - Precision: N/A")

        if total_true_events > 0:
            overall_accuracy_rate = (total_correctly_predicted / total_true_events) * 100
            print(f"\nAll Events:")
            print(f"  - Accuracy: {overall_accuracy_rate:.2f}% ({total_correctly_predicted}/{total_true_events})")
            print(f"  - Missed: {total_missed}")
            print(f"  - False triggers: {num_false_triggers}")
            overall_precision_denom = total_correctly_predicted + num_false_triggers
            overall_precision = (total_correctly_predicted / overall_precision_denom * 100) if overall_precision_denom > 0 else float('nan')
            print(f"  - Recall: {overall_accuracy_rate:.2f}%")
            print(f"  - Precision: {overall_precision:.2f}% ({total_correctly_predicted}/{overall_precision_denom})") if overall_precision_denom > 0 else print(f"  - Precision: N/A")

        if APPLY_REFINEMENT and ml_model_left is not None:
            print("\nBorder Refinement RMSE:")
            if border_refinement_left_rmse is not None and not np.isnan(border_refinement_left_rmse):
                print(f"  - Left border RMSE: {border_refinement_left_rmse:.2f} time steps")
                print(f"  - Right border RMSE: {border_refinement_right_rmse:.2f} time steps")
                print(f"  - Matched events: {border_refinement_matched_events}")
            else:
                print("  - Border refinement RMSE: N/A (no matched events or refinement not applied)")
    print("=" * 60)


if __name__ == "__main__":
    main()
