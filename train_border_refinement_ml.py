import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import pickle
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
from datetime import timedelta
from border_refinement_utils import determine_event_polarity, extract_features, extract_cnn_window

_METADATA_FILENAME = "model_metadata.json"
_TARGET_TYPE       = "log_half_width"
_MODEL_TYPE_RF     = "rf"
_MODEL_TYPE_FFN    = "ffn"
_MODEL_TYPE_CNN    = "cnn"


def load_border_refinement_data(data_dir="border_refinement_set"):
    """Load all border refinement data files from the directory."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist. Run use_model.py with USE_ANNOTATION=True first.")

    pattern = os.path.join(data_dir, "border_refinement_data_file_*.npz")
    data_files = sorted(glob.glob(pattern))

    if len(data_files) == 0:
        raise FileNotFoundError(f"No border refinement data files found in {data_dir}")

    print(f"Found {len(data_files)} border refinement data files")

    all_data = []
    for file_path in data_files:
        data = np.load(file_path, allow_pickle=True)
        all_data.append({
            'file_path': file_path,
            'current_values': data['current_values'],
            'dataset_timestamps': data['dataset_timestamps'],
            'op_timestamps': data['op_timestamps'],
            'op_values': data['op_values'],
            'predicted_borders': data['predicted_borders'],
            'true_borders': data['true_borders']
        })
        print(f"  Loaded {file_path}: {len(data['current_values'])} points, "
              f"{len(data['op_timestamps'])} OP values, "
              f"{len(data['predicted_borders'])} predicted events, "
              f"{len(data['true_borders'])} true events")

    return all_data


def match_events_efficiently(predicted_events, true_events):
    """Match predicted events to true events based on overlap."""
    pred_starts = [pd.to_datetime(e[0]) for e in predicted_events]
    pred_ends = [pd.to_datetime(e[1]) for e in predicted_events]
    true_starts = [pd.to_datetime(e[0]) for e in true_events]
    true_ends = [pd.to_datetime(e[1]) for e in true_events]

    matches = []
    true_idx = 0

    for pred_idx, (pred_start, pred_end) in enumerate(zip(pred_starts, pred_ends)):
        best_match = None
        best_overlap = 0.0
        best_true_idx = None

        search_start = true_idx

        for t_idx in range(search_start, len(true_events)):
            true_start = true_starts[t_idx]
            true_end = true_ends[t_idx]

            if true_start > pred_end:
                break

            if pred_start <= true_end and true_start <= pred_end:
                overlap_start = max(pred_start, true_start)
                overlap_end = min(pred_end, true_end)
                overlap_seconds = (overlap_end - overlap_start).total_seconds()

                if overlap_seconds > best_overlap:
                    best_overlap = overlap_seconds
                    best_match = t_idx
                    best_true_idx = t_idx

            if true_end < pred_start and best_match is None:
                true_idx = max(true_idx, t_idx + 1)

        if best_match is not None:
            matches.append((pred_idx, best_true_idx, best_overlap))

    return matches


def match_events_with_merge(predicted_events, true_events):
    """Match predicted events to true events, merging multiple overlapping true events."""
    pred_starts = [pd.to_datetime(e[0]) for e in predicted_events]
    pred_ends = [pd.to_datetime(e[1]) for e in predicted_events]
    true_starts = [pd.to_datetime(e[0]) for e in true_events]
    true_ends = [pd.to_datetime(e[1]) for e in true_events]

    matches = []
    true_idx = 0

    for pred_idx, (pred_start, pred_end) in enumerate(zip(pred_starts, pred_ends)):
        overlapping_indices = []

        search_start = true_idx

        for t_idx in range(search_start, len(true_events)):
            true_start = true_starts[t_idx]
            true_end = true_ends[t_idx]

            if true_start > pred_end:
                break

            if pred_start <= true_end and true_start <= pred_end:
                overlapping_indices.append(t_idx)

            if true_end < pred_start and len(overlapping_indices) == 0:
                true_idx = max(true_idx, t_idx + 1)

        if len(overlapping_indices) > 0:
            combined_start = min(true_starts[i] for i in overlapping_indices)
            combined_end = max(true_ends[i] for i in overlapping_indices)
            matches.append((pred_idx, overlapping_indices, combined_start, combined_end))

    return matches


def prepare_training_data(data_list):
    """Prepare training data from border refinement dataset."""
    print("\n" + "=" * 60)
    print("Preparing Training Data")
    print("=" * 60)

    all_features = []
    all_left_adjustments = []
    all_right_adjustments = []

    for file_idx, data in enumerate(data_list):
        predicted_borders = data['predicted_borders']
        true_borders = data['true_borders']
        dataset_timestamps = pd.to_datetime(data['dataset_timestamps'])
        op_timestamps = pd.to_datetime(data['op_timestamps']) if len(data['op_timestamps']) > 0 else None
        op_values = data['op_values'] if len(data['op_values']) > 0 else None
        current_values = data['current_values']

        predicted_events = [(pd.to_datetime(b[0]), pd.to_datetime(b[1])) for b in predicted_borders]
        true_events = [(pd.to_datetime(b[0]), pd.to_datetime(b[1])) for b in true_borders]

        matches = match_events_with_merge(predicted_events, true_events)

        if len(matches) == 0:
            print(f"  File {file_idx + 1}: No matched events, skipping...")
            continue

        if len(dataset_timestamps) > 1:
            time_steps = (dataset_timestamps[1:] - dataset_timestamps[:-1]).total_seconds()
            avg_time_step = np.mean(time_steps) if len(time_steps) > 0 else 1.0
        else:
            avg_time_step = 1.0

        multi_match_count = 0

        for pred_idx, true_indices, combined_true_start, combined_true_end in matches:
            pred_start = pd.to_datetime(predicted_events[pred_idx][0])
            pred_end = pd.to_datetime(predicted_events[pred_idx][1])

            if len(true_indices) > 1:
                multi_match_count += 1

            if op_timestamps is not None and op_values is not None and len(op_timestamps) > 0:
                features = extract_features(
                    (pred_start, pred_end),
                    op_timestamps,
                    op_values,
                    dataset_timestamps,
                    current_values
                )

                if features is None:
                    continue

                peak_time = pred_start + (pred_end - pred_start) / 2

                left_hw_seconds  = (peak_time - combined_true_start).total_seconds()
                right_hw_seconds = (combined_true_end - peak_time).total_seconds()

                left_half_width  = max(left_hw_seconds  / avg_time_step, 0.5) if avg_time_step > 0 else max(left_hw_seconds, 0.5)
                right_half_width = max(right_hw_seconds / avg_time_step, 0.5) if avg_time_step > 0 else max(right_hw_seconds, 0.5)

                all_features.append(features)
                all_left_adjustments.append(np.log1p(left_half_width))
                all_right_adjustments.append(np.log1p(right_half_width))

        print(f"  File {file_idx + 1}: Extracted {len(matches)} matched events"
              f" ({multi_match_count} with merged true events)")

    if len(all_features) == 0:
        raise ValueError("No training samples extracted! Check data and matching.")

    features_df = pd.DataFrame(all_features)
    feature_names = list(features_df.columns)

    X = features_df.values
    y_left = np.array(all_left_adjustments)
    y_right = np.array(all_right_adjustments)

    print(f"\nTotal training samples: {len(X)}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Feature names: {feature_names}")

    return X, y_left, y_right, feature_names


def prepare_cnn_training_data(data_list, window_size=300):
    """Prepare training data for the CNN border refinement model."""
    print("\n" + "=" * 60)
    print("Preparing CNN Training Data")
    print("=" * 60)

    all_windows = []
    all_left = []
    all_right = []

    for file_idx, data in enumerate(data_list):
        predicted_borders = data['predicted_borders']
        true_borders = data['true_borders']
        dataset_timestamps = pd.to_datetime(data['dataset_timestamps'])
        op_timestamps = pd.to_datetime(data['op_timestamps']) if len(data['op_timestamps']) > 0 else None
        op_values = data['op_values'] if len(data['op_values']) > 0 else None
        current_values = data['current_values']

        predicted_events = [(pd.to_datetime(b[0]), pd.to_datetime(b[1])) for b in predicted_borders]
        true_events = [(pd.to_datetime(b[0]), pd.to_datetime(b[1])) for b in true_borders]

        matches = match_events_with_merge(predicted_events, true_events)
        if len(matches) == 0:
            print(f"  File {file_idx + 1}: No matched events, skipping...")
            continue

        if len(dataset_timestamps) > 1:
            time_steps = (dataset_timestamps[1:] - dataset_timestamps[:-1]).total_seconds()
            avg_time_step = np.mean(time_steps) if len(time_steps) > 0 else 1.0
        else:
            avg_time_step = 1.0

        multi_match_count = 0
        file_count = 0

        for pred_idx, true_indices, combined_true_start, combined_true_end in matches:
            pred_start = pd.to_datetime(predicted_events[pred_idx][0])
            pred_end = pd.to_datetime(predicted_events[pred_idx][1])

            if len(true_indices) > 1:
                multi_match_count += 1

            if op_timestamps is None or op_values is None:
                continue

            window = extract_cnn_window(
                (pred_start, pred_end),
                op_timestamps, op_values,
                dataset_timestamps, current_values,
                window_size=window_size
            )
            if window is None:
                continue

            peak_time = pred_start + (pred_end - pred_start) / 2
            left_hw_seconds = (peak_time - combined_true_start).total_seconds()
            right_hw_seconds = (combined_true_end - peak_time).total_seconds()

            left_half_width = max(left_hw_seconds / avg_time_step, 0.5) if avg_time_step > 0 else max(left_hw_seconds, 0.5)
            right_half_width = max(right_hw_seconds / avg_time_step, 0.5) if avg_time_step > 0 else max(right_hw_seconds, 0.5)

            all_windows.append(window)
            all_left.append(np.log1p(left_half_width))
            all_right.append(np.log1p(right_half_width))
            file_count += 1

        print(f"  File {file_idx + 1}: Extracted {file_count} windows from {len(matches)} matches"
              f" ({multi_match_count} with merged true events)")

    if len(all_windows) == 0:
        raise ValueError("No CNN training samples extracted! Check data and matching.")

    X_windows = np.array(all_windows)
    y_left = np.array(all_left)
    y_right = np.array(all_right)

    print(f"\nTotal CNN training samples: {len(X_windows)}")
    print(f"Window shape: {X_windows.shape}")

    return X_windows, y_left, y_right


def get_hyperparameter_distribution(hyperparameter_config):
    """Create hyperparameter search space for Random Forest based on configuration."""
    param_dist = {}

    if 'n_estimators' in hyperparameter_config:
        n_est_config = hyperparameter_config['n_estimators']
        if isinstance(n_est_config, dict) and 'min' in n_est_config and 'max' in n_est_config:
            param_dist['n_estimators'] = randint(n_est_config['min'], n_est_config['max'] + 1)
        elif isinstance(n_est_config, list):
            param_dist['n_estimators'] = n_est_config

    if 'max_depth' in hyperparameter_config:
        param_dist['max_depth'] = hyperparameter_config['max_depth']

    if 'min_samples_split' in hyperparameter_config:
        min_split_config = hyperparameter_config['min_samples_split']
        if isinstance(min_split_config, dict) and 'min' in min_split_config and 'max' in min_split_config:
            param_dist['min_samples_split'] = randint(min_split_config['min'], min_split_config['max'] + 1)
        elif isinstance(min_split_config, list):
            param_dist['min_samples_split'] = min_split_config

    if 'min_samples_leaf' in hyperparameter_config:
        min_leaf_config = hyperparameter_config['min_samples_leaf']
        if isinstance(min_leaf_config, dict) and 'min' in min_leaf_config and 'max' in min_leaf_config:
            param_dist['min_samples_leaf'] = randint(min_leaf_config['min'], min_leaf_config['max'] + 1)
        elif isinstance(min_leaf_config, list):
            param_dist['min_samples_leaf'] = min_leaf_config

    if 'max_features' in hyperparameter_config:
        param_dist['max_features'] = hyperparameter_config['max_features']

    if 'bootstrap' in hyperparameter_config:
        param_dist['bootstrap'] = hyperparameter_config['bootstrap']

    return param_dist


def tune_hyperparameters(X_train, y_train, param_dist, n_iter=50, cv=5,
                        scoring='neg_mean_squared_error', random_state=42, n_jobs=-1):
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    base_model = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)

    cv_fold = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_fold,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
        return_train_score=True
    )

    search.fit(X_train, y_train)

    return search


def build_ffn_model(n_features, hidden_units=None, dropout_rate=0.2, learning_rate=1e-3):
    """Build a joint FFN that predicts both left and right log half-widths simultaneously."""
    if hidden_units is None:
        hidden_units = [256, 128, 64]

    inputs = tf.keras.Input(shape=(n_features,), name='features')
    x = inputs
    for i, units in enumerate(hidden_units):
        x = tf.keras.layers.Dense(units, activation='relu', name=f'dense_{i}')(x)
        x = tf.keras.layers.BatchNormalization(name=f'bn_{i}')(x)
        rate = dropout_rate * (0.5 if i == len(hidden_units) - 1 else 1.0)
        if rate > 0:
            x = tf.keras.layers.Dropout(rate, name=f'dropout_{i}')(x)
    outputs = tf.keras.layers.Dense(2, activation='linear', name='half_widths')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model


def train_ffn_model(X, y_left, y_right, test_size=0.2, random_state=42,
                    epochs=300, batch_size=32, learning_rate=1e-3,
                    patience=30, hidden_units=None):
    """Train a joint FFN model predicting left and right log half-widths together."""
    print("\n" + "=" * 60)
    print("Training FFN Model (Joint Two-Output)")
    print("=" * 60)

    y = np.stack([y_left, y_right], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = build_ffn_model(X_train.shape[1], hidden_units=hidden_units, learning_rate=learning_rate)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience,
            restore_best_weights=True, verbose=1
        )
    ]

    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    y_pred_test  = model.predict(X_test_scaled,  verbose=0)
    y_pred_train = model.predict(X_train_scaled, verbose=0)

    y_left_train,      y_right_train      = y_train[:, 0], y_train[:, 1]
    y_left_test,       y_right_test       = y_test[:, 0],  y_test[:, 1]
    y_left_pred_train, y_right_pred_train = y_pred_train[:, 0], y_pred_train[:, 1]
    y_left_pred_test,  y_right_pred_test  = y_pred_test[:, 0],  y_pred_test[:, 1]

    metrics_left_train  = {
        'rmse': np.sqrt(mean_squared_error(y_left_train,  y_left_pred_train)),
        'mae':  mean_absolute_error(y_left_train,  y_left_pred_train),
        'r2':   r2_score(y_left_train,  y_left_pred_train)
    }
    metrics_left_test   = {
        'rmse': np.sqrt(mean_squared_error(y_left_test,   y_left_pred_test)),
        'mae':  mean_absolute_error(y_left_test,   y_left_pred_test),
        'r2':   r2_score(y_left_test,   y_left_pred_test)
    }
    metrics_right_train = {
        'rmse': np.sqrt(mean_squared_error(y_right_train, y_right_pred_train)),
        'mae':  mean_absolute_error(y_right_train, y_right_pred_train),
        'r2':   r2_score(y_right_train, y_right_pred_train)
    }
    metrics_right_test  = {
        'rmse': np.sqrt(mean_squared_error(y_right_test,  y_right_pred_test)),
        'mae':  mean_absolute_error(y_right_test,  y_right_pred_test),
        'r2':   r2_score(y_right_test,  y_right_pred_test)
    }

    print("\nLeft Border:")
    print(f"  Train - RMSE: {metrics_left_train['rmse']:.4f}, MAE: {metrics_left_train['mae']:.4f}, R²: {metrics_left_train['r2']:.4f}")
    print(f"  Test  - RMSE: {metrics_left_test['rmse']:.4f},  MAE: {metrics_left_test['mae']:.4f},  R²: {metrics_left_test['r2']:.4f}")
    print("Right Border:")
    print(f"  Train - RMSE: {metrics_right_train['rmse']:.4f}, MAE: {metrics_right_train['mae']:.4f}, R²: {metrics_right_train['r2']:.4f}")
    print(f"  Test  - RMSE: {metrics_right_test['rmse']:.4f},  MAE: {metrics_right_test['mae']:.4f},  R²: {metrics_right_test['r2']:.4f}")

    return {
        'model_left':           None,
        'model_right':          None,
        'model_ffn':            model,
        'feature_scaler':       scaler,
        'model_type':           _MODEL_TYPE_FFN,
        'X_test':               X_test,
        'y_left_test':          y_left_test,
        'y_right_test':         y_right_test,
        'y_left_pred_test':     y_left_pred_test,
        'y_right_pred_test':    y_right_pred_test,
        'metrics_left_train':   metrics_left_train,
        'metrics_left_test':    metrics_left_test,
        'metrics_right_train':  metrics_right_train,
        'metrics_right_test':   metrics_right_test,
        'best_params_left':     None,
        'best_params_right':    None,
        'cv_score_left':        None,
        'cv_score_right':       None,
    }


def build_cnn_refinement_model(window_size=300, n_channels=2, learning_rate=1e-3):
    """Build a 1D CNN that predicts a single scalar (left or right log half-width)."""
    inputs = tf.keras.Input(shape=(window_size, n_channels), name='window')
    x = inputs

    for filters, kernel in [(32, 7), (64, 5), (128, 3)]:
        x = tf.keras.layers.Conv1D(filters, kernel, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='linear', name='half_width')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model


def train_cnn_model(X_windows, y_left, y_right, test_size=0.2, random_state=42,
                    epochs=300, batch_size=32, learning_rate=1e-3, patience=30):
    """Train two separate CNN models for left and right border refinement."""
    print("\n" + "=" * 60)
    print("Training CNN Models (Separate Left / Right)")
    print("=" * 60)

    n_samples, window_size, n_channels = X_windows.shape

    X_train, X_test, y_left_train, y_left_test, y_right_train, y_right_test = train_test_split(
        X_windows, y_left, y_right, test_size=test_size, random_state=random_state
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    channel_means = np.zeros(n_channels)
    channel_stds = np.ones(n_channels)
    for ch in range(n_channels):
        ch_train = X_train[:, :, ch].ravel()
        channel_means[ch] = ch_train.mean()
        channel_stds[ch] = ch_train.std() + 1e-8

    def scale(arr):
        out = arr.copy()
        for ch in range(n_channels):
            out[:, :, ch] = (out[:, :, ch] - channel_means[ch]) / channel_stds[ch]
        return out

    X_train_s = scale(X_train)
    X_test_s = scale(X_test)

    scaler_info = {'channel_means': channel_means, 'channel_stds': channel_stds}

    callbacks_tmpl = lambda: [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience,
            restore_best_weights=True, verbose=1
        )
    ]

    print("\n" + "-" * 60)
    print("Training LEFT border CNN...")
    print("-" * 60)
    model_left = build_cnn_refinement_model(window_size, n_channels, learning_rate)
    model_left.summary()
    model_left.fit(
        X_train_s, y_left_train,
        validation_data=(X_test_s, y_left_test),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks_tmpl(), verbose=1
    )

    print("\n" + "-" * 60)
    print("Training RIGHT border CNN...")
    print("-" * 60)
    model_right = build_cnn_refinement_model(window_size, n_channels, learning_rate)
    model_right.fit(
        X_train_s, y_right_train,
        validation_data=(X_test_s, y_right_test),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks_tmpl(), verbose=1
    )

    y_left_pred_train = model_left.predict(X_train_s, verbose=0).ravel()
    y_left_pred_test = model_left.predict(X_test_s, verbose=0).ravel()
    y_right_pred_train = model_right.predict(X_train_s, verbose=0).ravel()
    y_right_pred_test = model_right.predict(X_test_s, verbose=0).ravel()

    metrics_left_train = {
        'rmse': np.sqrt(mean_squared_error(y_left_train, y_left_pred_train)),
        'mae': mean_absolute_error(y_left_train, y_left_pred_train),
        'r2': r2_score(y_left_train, y_left_pred_train),
    }
    metrics_left_test = {
        'rmse': np.sqrt(mean_squared_error(y_left_test, y_left_pred_test)),
        'mae': mean_absolute_error(y_left_test, y_left_pred_test),
        'r2': r2_score(y_left_test, y_left_pred_test),
    }
    metrics_right_train = {
        'rmse': np.sqrt(mean_squared_error(y_right_train, y_right_pred_train)),
        'mae': mean_absolute_error(y_right_train, y_right_pred_train),
        'r2': r2_score(y_right_train, y_right_pred_train),
    }
    metrics_right_test = {
        'rmse': np.sqrt(mean_squared_error(y_right_test, y_right_pred_test)),
        'mae': mean_absolute_error(y_right_test, y_right_pred_test),
        'r2': r2_score(y_right_test, y_right_pred_test),
    }

    print("\nLeft Border CNN:")
    print(f"  Train - RMSE: {metrics_left_train['rmse']:.4f}, MAE: {metrics_left_train['mae']:.4f}, R²: {metrics_left_train['r2']:.4f}")
    print(f"  Test  - RMSE: {metrics_left_test['rmse']:.4f},  MAE: {metrics_left_test['mae']:.4f},  R²: {metrics_left_test['r2']:.4f}")
    print("Right Border CNN:")
    print(f"  Train - RMSE: {metrics_right_train['rmse']:.4f}, MAE: {metrics_right_train['mae']:.4f}, R²: {metrics_right_train['r2']:.4f}")
    print(f"  Test  - RMSE: {metrics_right_test['rmse']:.4f},  MAE: {metrics_right_test['mae']:.4f},  R²: {metrics_right_test['r2']:.4f}")

    return {
        'model_left':           model_left,
        'model_right':          model_right,
        'model_ffn':            None,
        'feature_scaler':       scaler_info,
        'model_type':           _MODEL_TYPE_CNN,
        'X_test':               X_test,
        'y_left_test':          y_left_test,
        'y_right_test':         y_right_test,
        'y_left_pred_test':     y_left_pred_test,
        'y_right_pred_test':    y_right_pred_test,
        'metrics_left_train':   metrics_left_train,
        'metrics_left_test':    metrics_left_test,
        'metrics_right_train':  metrics_right_train,
        'metrics_right_test':   metrics_right_test,
        'best_params_left':     None,
        'best_params_right':    None,
        'cv_score_left':        None,
        'cv_score_right':       None,
    }


def train_models(X, y_left, y_right, test_size=0.2, random_state=42,
                use_hyperparameter_tuning=False, hyperparameter_config=None,
                n_iter=50, cv=5,
                model_type=_MODEL_TYPE_FFN,
                ffn_epochs=300, ffn_batch_size=32, ffn_learning_rate=1e-3,
                ffn_patience=30, ffn_hidden_units=None,
                cnn_epochs=300, cnn_batch_size=32, cnn_learning_rate=1e-3,
                cnn_patience=30):
    """Train border refinement models."""
    if model_type == _MODEL_TYPE_CNN:
        return train_cnn_model(
            X, y_left, y_right,
            test_size=test_size, random_state=random_state,
            epochs=cnn_epochs, batch_size=cnn_batch_size,
            learning_rate=cnn_learning_rate, patience=cnn_patience
        )
    if model_type == _MODEL_TYPE_FFN:
        return train_ffn_model(
            X, y_left, y_right,
            test_size=test_size, random_state=random_state,
            epochs=ffn_epochs, batch_size=ffn_batch_size,
            learning_rate=ffn_learning_rate, patience=ffn_patience,
            hidden_units=ffn_hidden_units
        )
    print("\n" + "=" * 60)
    print("Training Random Forest Models")
    print("=" * 60)

    X_train, X_test, y_left_train, y_left_test, y_right_train, y_right_test = train_test_split(
        X, y_left, y_right, test_size=test_size, random_state=random_state
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    param_dist = None
    if use_hyperparameter_tuning and hyperparameter_config is not None:
        param_dist = get_hyperparameter_distribution(hyperparameter_config)

    print("\n" + "-" * 60)
    print("Training left border model...")
    print("-" * 60)

    if use_hyperparameter_tuning and param_dist is not None:
        print(f"Performing hyperparameter tuning (n_iter={n_iter}, cv={cv})...")
        search_left = tune_hyperparameters(
            X_train, y_left_train, param_dist,
            n_iter=n_iter, cv=cv, random_state=random_state
        )
        model_left = search_left.best_estimator_
        best_params_left = search_left.best_params_
        cv_score_left = -search_left.best_score_

        print(f"\nBest CV RMSE: {np.sqrt(cv_score_left):.4f}")
        print(f"Best parameters:")
        for param, value in best_params_left.items():
            print(f"  {param}: {value}")
    else:
        model_left = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        model_left.fit(X_train, y_left_train)
        best_params_left = None
        cv_score_left = None

    print("\n" + "-" * 60)
    print("Training right border model...")
    print("-" * 60)

    if use_hyperparameter_tuning and param_dist is not None:
        print(f"Performing hyperparameter tuning (n_iter={n_iter}, cv={cv})...")
        search_right = tune_hyperparameters(
            X_train, y_right_train, param_dist,
            n_iter=n_iter, cv=cv, random_state=random_state
        )
        model_right = search_right.best_estimator_
        best_params_right = search_right.best_params_
        cv_score_right = -search_right.best_score_

        print(f"\nBest CV RMSE: {np.sqrt(cv_score_right):.4f}")
        print(f"Best parameters:")
        for param, value in best_params_right.items():
            print(f"  {param}: {value}")
    else:
        model_right = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        model_right.fit(X_train, y_right_train)
        best_params_right = None
        cv_score_right = None

    print("\nEvaluating models...")

    y_left_pred_train = model_left.predict(X_train)
    y_left_pred_test = model_left.predict(X_test)

    y_right_pred_train = model_right.predict(X_train)
    y_right_pred_test = model_right.predict(X_test)

    metrics_left_train = {
        'rmse': np.sqrt(mean_squared_error(y_left_train, y_left_pred_train)),
        'mae': mean_absolute_error(y_left_train, y_left_pred_train),
        'r2': r2_score(y_left_train, y_left_pred_train)
    }
    metrics_left_test = {
        'rmse': np.sqrt(mean_squared_error(y_left_test, y_left_pred_test)),
        'mae': mean_absolute_error(y_left_test, y_left_pred_test),
        'r2': r2_score(y_left_test, y_left_pred_test)
    }

    metrics_right_train = {
        'rmse': np.sqrt(mean_squared_error(y_right_train, y_right_pred_train)),
        'mae': mean_absolute_error(y_right_train, y_right_pred_train),
        'r2': r2_score(y_right_train, y_right_pred_train)
    }
    metrics_right_test = {
        'rmse': np.sqrt(mean_squared_error(y_right_test, y_right_pred_test)),
        'mae': mean_absolute_error(y_right_test, y_right_pred_test),
        'r2': r2_score(y_right_test, y_right_pred_test)
    }

    print("\nLeft Border Model:")
    print(f"  Train - RMSE: {metrics_left_train['rmse']:.2f}, MAE: {metrics_left_train['mae']:.2f}, R²: {metrics_left_train['r2']:.4f}")
    print(f"  Test  - RMSE: {metrics_left_test['rmse']:.2f}, MAE: {metrics_left_test['mae']:.2f}, R²: {metrics_left_test['r2']:.4f}")

    print("\nRight Border Model:")
    print(f"  Train - RMSE: {metrics_right_train['rmse']:.2f}, MAE: {metrics_right_train['mae']:.2f}, R²: {metrics_right_train['r2']:.4f}")
    print(f"  Test  - RMSE: {metrics_right_test['rmse']:.2f}, MAE: {metrics_right_test['mae']:.2f}, R²: {metrics_right_test['r2']:.4f}")

    return {
        'model_left': model_left,
        'model_right': model_right,
        'model_ffn': None,
        'feature_scaler': None,
        'model_type': _MODEL_TYPE_RF,
        'X_test': X_test,
        'y_left_test': y_left_test,
        'y_right_test': y_right_test,
        'y_left_pred_test': y_left_pred_test,
        'y_right_pred_test': y_right_pred_test,
        'metrics_left_train': metrics_left_train,
        'metrics_left_test': metrics_left_test,
        'metrics_right_train': metrics_right_train,
        'metrics_right_test': metrics_right_test,
        'best_params_left': best_params_left,
        'best_params_right': best_params_right,
        'cv_score_left': cv_score_left,
        'cv_score_right': cv_score_right
    }


def refine_event_boundaries_ml(predicted_events, op_timestamps, op_values, dataset_timestamps,
                                current_values, model_left, model_right, feature_names,
                                model_type=_MODEL_TYPE_RF, feature_scaler=None):
    """Refine event boundaries using trained ML models."""
    refined_events = []

    if len(dataset_timestamps) > 1:
        time_steps_arr = (dataset_timestamps[1:] - dataset_timestamps[:-1]).total_seconds()
        avg_time_step = float(np.mean(time_steps_arr)) if len(time_steps_arr) > 0 else 1.0
    else:
        avg_time_step = 1.0

    for event in predicted_events:
        event_start = pd.to_datetime(event[0])
        event_end   = pd.to_datetime(event[1])

        if model_type == _MODEL_TYPE_CNN:
            window = extract_cnn_window(
                (event_start, event_end),
                op_timestamps, op_values,
                dataset_timestamps, current_values
            )
            if window is None:
                refined_events.append((event_start, event_end))
                continue

            window_s = window.copy()
            ch_means = feature_scaler['channel_means']
            ch_stds = feature_scaler['channel_stds']
            for ch in range(window_s.shape[1]):
                window_s[:, ch] = (window_s[:, ch] - ch_means[ch]) / ch_stds[ch]

            X_in = window_s[np.newaxis, ...]
            raw_left = float(model_left.predict(X_in, verbose=0).ravel()[0])
            raw_right = float(model_right.predict(X_in, verbose=0).ravel()[0])
        else:
            features = extract_features(
                (event_start, event_end),
                op_timestamps,
                op_values,
                dataset_timestamps,
                current_values
            )

            if features is None:
                refined_events.append((event_start, event_end))
                continue

            feature_array = np.array([[features[name] for name in feature_names]])

            if model_type == _MODEL_TYPE_FFN:
                X_scaled  = feature_scaler.transform(feature_array)
                pred      = model_left.predict(X_scaled, verbose=0)
                raw_left  = float(pred[0, 0])
                raw_right = float(pred[0, 1])
            else:
                raw_left  = model_left.predict(feature_array)[0]
                raw_right = model_right.predict(feature_array)[0]

        left_half_width_steps  = np.expm1(raw_left)
        right_half_width_steps = np.expm1(raw_right)
        peak_time     = event_start + (event_end - event_start) / 2
        refined_start = peak_time - timedelta(seconds=left_half_width_steps  * avg_time_step)
        refined_end   = peak_time + timedelta(seconds=right_half_width_steps * avg_time_step)

        if refined_start < dataset_timestamps[0]:
            refined_start = dataset_timestamps[0]
        if refined_end > dataset_timestamps[-1]:
            refined_end = dataset_timestamps[-1]

        if refined_start >= refined_end:
            refined_start = event_start
            refined_end   = event_end

        refined_events.append((refined_start, refined_end))

    return refined_events


def calculate_border_metrics(predicted_events, true_events, matches, dataset_timestamps):
    """Calculate RMSE metrics for border refinement."""
    if len(matches) == 0:
        return {
            'left_rmse': np.nan,
            'right_rmse': np.nan,
            'width_rmse': np.nan,
            'num_matched_events': 0
        }

    left_errors = []
    right_errors = []
    width_errors = []

    if len(dataset_timestamps) > 1:
        time_steps = (dataset_timestamps[1:] - dataset_timestamps[:-1]).total_seconds()
        avg_time_step = np.mean(time_steps) if len(time_steps) > 0 else 1.0
    else:
        avg_time_step = 1.0

    for pred_idx, true_idx, overlap in matches:
        pred_start = pd.to_datetime(predicted_events[pred_idx][0])
        pred_end = pd.to_datetime(predicted_events[pred_idx][1])
        true_start = pd.to_datetime(true_events[true_idx][0])
        true_end = pd.to_datetime(true_events[true_idx][1])

        left_error_seconds = (pred_start - true_start).total_seconds()
        right_error_seconds = (pred_end - true_end).total_seconds()

        pred_width_seconds = (pred_end - pred_start).total_seconds()
        true_width_seconds = (true_end - true_start).total_seconds()
        width_error_seconds = pred_width_seconds - true_width_seconds

        left_error = left_error_seconds / avg_time_step if avg_time_step > 0 else left_error_seconds
        right_error = right_error_seconds / avg_time_step if avg_time_step > 0 else right_error_seconds
        width_error = width_error_seconds / avg_time_step if avg_time_step > 0 else width_error_seconds

        left_errors.append(left_error)
        right_errors.append(right_error)
        width_errors.append(width_error)

    left_rmse = np.sqrt(np.mean(np.array(left_errors) ** 2))
    right_rmse = np.sqrt(np.mean(np.array(right_errors) ** 2))
    width_rmse = np.sqrt(np.mean(np.array(width_errors) ** 2))

    return {
        'left_rmse': left_rmse,
        'right_rmse': right_rmse,
        'width_rmse': width_rmse,
        'num_matched_events': len(matches),
        'left_errors': left_errors,
        'right_errors': right_errors,
        'width_errors': width_errors
    }


def plot_refinement_results(data_list, model_left, model_right, feature_names,
                            output_dir="border_refinement_ml_plots", max_files=3,
                            model_type=_MODEL_TYPE_RF, feature_scaler=None):
    """Plot border refinement results comparing original and ML-refined borders."""
    os.makedirs(output_dir, exist_ok=True)

    FIGURE_SIZE = (15, 16)
    CURRENT_LINE_COLOR = 'k'
    CURRENT_LINE_WIDTH = 2
    OP_LINE_COLOR = 'blue'
    OP_LINE_WIDTH = 1.5
    OP_LINE_ALPHA = 0.7
    PREDICTED_EVENT_COLOR = 'red'
    REFINED_EVENT_COLOR = 'green'
    TRUE_EVENT_COLOR = 'black'
    EVENT_ALPHA = 0.3
    AXIS_LABEL_FONTSIZE = 14
    TICK_LABEL_FONTSIZE = 12
    TITLE_FONTSIZE = 12
    LEGEND_FONTSIZE = 10
    DPI = 100
    CURRENT_Y_LIMITS = (-0.1 / 50, 0.2 / 50)
    OP_Y_LIMITS = (0, 1)

    plot_count = 0
    batch_size = 2000

    print("\n" + "=" * 60)
    print("Plotting Refinement Results")
    print("=" * 60)

    for file_idx, data in enumerate(data_list[:max_files]):
        current_values = data['current_values']
        dataset_timestamps = pd.to_datetime(data['dataset_timestamps'])
        op_timestamps = pd.to_datetime(data['op_timestamps']) if len(data['op_timestamps']) > 0 else None
        op_values = data['op_values'] if len(data['op_values']) > 0 else None
        predicted_borders = data['predicted_borders']
        true_borders = data['true_borders']

        num_points = len(current_values)
        num_batches = (num_points + batch_size - 1) // batch_size
        global_time_reference = dataset_timestamps[0]

        predicted_events = [(pd.to_datetime(b[0]), pd.to_datetime(b[1])) for b in predicted_borders]
        true_events = [(pd.to_datetime(b[0]), pd.to_datetime(b[1])) for b in true_borders]

        if op_timestamps is not None and op_values is not None and len(op_timestamps) > 0:
            refined_predicted_events = refine_event_boundaries_ml(
                predicted_events,
                op_timestamps,
                op_values,
                dataset_timestamps,
                current_values,
                model_left,
                model_right,
                feature_names,
                model_type=model_type,
                feature_scaler=feature_scaler
            )
        else:
            refined_predicted_events = predicted_events

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_points)

            batch_current = current_values[start_idx:end_idx]
            batch_timestamps = dataset_timestamps[start_idx:end_idx]
            batch_start_time = batch_timestamps[0]
            batch_end_time = batch_timestamps[-1]

            time_deltas = (batch_timestamps - global_time_reference).total_seconds() * 1e9
            batch_t_ns = time_deltas.values

            batch_predicted = []
            batch_refined = []
            batch_true = []

            for event in predicted_events:
                if batch_start_time <= event[1] and batch_end_time >= event[0]:
                    batch_predicted.append(event)

            for event in refined_predicted_events:
                if batch_start_time <= event[1] and batch_end_time >= event[0]:
                    batch_refined.append(event)

            for event in true_events:
                if batch_start_time <= event[1] and batch_end_time >= event[0]:
                    batch_true.append(event)

            if len(batch_predicted) == 0 and len(batch_refined) == 0 and len(batch_true) == 0:
                continue

            batch_op_values = None
            batch_op_t_ns = None
            if op_timestamps is not None and op_values is not None and len(op_timestamps) > 0:
                op_mask = (op_timestamps >= batch_start_time) & (op_timestamps <= batch_end_time)
                batch_op_values = op_values[op_mask]
                batch_op_timestamps = op_timestamps[op_mask]
                if len(batch_op_values) > 0:
                    batch_op_t_ns = ((batch_op_timestamps - global_time_reference).total_seconds() * 1e9).values

            fig, axes = plt.subplots(4, 1, figsize=FIGURE_SIZE, sharex=True)
            ax_current_original, ax_current_refined, ax_op_original, ax_op_refined = axes

            ax_current_original.plot(batch_t_ns, batch_current, color=CURRENT_LINE_COLOR, linewidth=CURRENT_LINE_WIDTH, label='Current')
            for orig_start, orig_end in batch_predicted:
                event_start_ns = (orig_start - global_time_reference).total_seconds() * 1e9
                event_end_ns = (orig_end - global_time_reference).total_seconds() * 1e9
                event_start_ns = max(event_start_ns, batch_t_ns[0])
                event_end_ns = min(event_end_ns, batch_t_ns[-1])
                ax_current_original.axvspan(event_start_ns, event_end_ns, alpha=EVENT_ALPHA, color=PREDICTED_EVENT_COLOR)
            ax_current_original.set_ylabel('Current (A)', fontsize=AXIS_LABEL_FONTSIZE)
            ax_current_original.set_title('Current with Original Predicted Events', fontsize=TITLE_FONTSIZE)
            ax_current_original.set_ylim(CURRENT_Y_LIMITS)
            ax_current_original.grid(True, alpha=0.3)
            ax_current_original.legend(fontsize=LEGEND_FONTSIZE, loc='upper left')

            ax_current_refined.plot(batch_t_ns, batch_current, color=CURRENT_LINE_COLOR, linewidth=CURRENT_LINE_WIDTH, label='Current')
            refined_legend_added = False
            true_legend_added = False
            for ref_idx, (ref_start, ref_end) in enumerate(batch_refined):
                event_start_ns = (ref_start - global_time_reference).total_seconds() * 1e9
                event_end_ns = (ref_end - global_time_reference).total_seconds() * 1e9
                event_start_ns = max(event_start_ns, batch_t_ns[0])
                event_end_ns = min(event_end_ns, batch_t_ns[-1])
                label = 'Refined' if not refined_legend_added else ''
                if label:
                    refined_legend_added = True
                ax_current_refined.axvspan(event_start_ns, event_end_ns, alpha=EVENT_ALPHA, color=REFINED_EVENT_COLOR, label=label)
            for true_idx, (true_start, true_end) in enumerate(batch_true):
                event_start_ns = (true_start - global_time_reference).total_seconds() * 1e9
                event_end_ns = (true_end - global_time_reference).total_seconds() * 1e9
                event_start_ns = max(event_start_ns, batch_t_ns[0])
                event_end_ns = min(event_end_ns, batch_t_ns[-1])
                label = 'True' if not true_legend_added else ''
                if label:
                    true_legend_added = True
                ax_current_refined.axvspan(event_start_ns, event_end_ns, alpha=0.2, color=TRUE_EVENT_COLOR, label=label)
            ax_current_refined.set_ylabel('Current (A)', fontsize=AXIS_LABEL_FONTSIZE)
            ax_current_refined.set_title('Current with ML-Refined Predicted Events (Green) and True Events (Black)', fontsize=TITLE_FONTSIZE)
            ax_current_refined.set_ylim(CURRENT_Y_LIMITS)
            ax_current_refined.grid(True, alpha=0.3)
            ax_current_refined.legend(fontsize=LEGEND_FONTSIZE, loc='upper left')

            if batch_op_values is not None and batch_op_t_ns is not None and len(batch_op_values) > 0:
                ax_op_original.plot(batch_op_t_ns, batch_op_values, color=OP_LINE_COLOR, linewidth=OP_LINE_WIDTH, alpha=OP_LINE_ALPHA, label='OP')
            for orig_start, orig_end in batch_predicted:
                event_start_ns = (orig_start - global_time_reference).total_seconds() * 1e9
                event_end_ns = (orig_end - global_time_reference).total_seconds() * 1e9
                event_start_ns = max(event_start_ns, batch_t_ns[0])
                event_end_ns = min(event_end_ns, batch_t_ns[-1])
                ax_op_original.axvspan(event_start_ns, event_end_ns, alpha=EVENT_ALPHA, color=PREDICTED_EVENT_COLOR)
            ax_op_original.set_ylabel('OP Value', fontsize=AXIS_LABEL_FONTSIZE)
            ax_op_original.set_title('OP with Original Predicted Events', fontsize=TITLE_FONTSIZE)
            ax_op_original.set_ylim(OP_Y_LIMITS)
            ax_op_original.grid(True, alpha=0.3)
            ax_op_original.legend(fontsize=LEGEND_FONTSIZE, loc='upper left')

            if batch_op_values is not None and batch_op_t_ns is not None and len(batch_op_values) > 0:
                ax_op_refined.plot(batch_op_t_ns, batch_op_values, color=OP_LINE_COLOR, linewidth=OP_LINE_WIDTH, alpha=OP_LINE_ALPHA, label='OP')
            for ref_start, ref_end in batch_refined:
                event_start_ns = (ref_start - global_time_reference).total_seconds() * 1e9
                event_end_ns = (ref_end - global_time_reference).total_seconds() * 1e9
                event_start_ns = max(event_start_ns, batch_t_ns[0])
                event_end_ns = min(event_end_ns, batch_t_ns[-1])
                ax_op_refined.axvspan(event_start_ns, event_end_ns, alpha=EVENT_ALPHA, color=REFINED_EVENT_COLOR)
            ax_op_refined.set_xlabel('Time (ns)', fontsize=AXIS_LABEL_FONTSIZE)
            ax_op_refined.set_ylabel('OP Value', fontsize=AXIS_LABEL_FONTSIZE)
            ax_op_refined.set_title('OP with ML-Refined Predicted Events', fontsize=TITLE_FONTSIZE)
            ax_op_refined.set_ylim(OP_Y_LIMITS)
            ax_op_refined.grid(True, alpha=0.3)
            ax_op_refined.legend(fontsize=LEGEND_FONTSIZE, loc='upper left')

            fig.suptitle(f'File {file_idx + 1}, Batch {batch_idx + 1}/{num_batches}', fontsize=TITLE_FONTSIZE + 2, y=0.995)
            plt.tight_layout()

            plot_filename = os.path.join(output_dir, f"file_{file_idx + 1:04d}_batch_{batch_idx + 1:04d}.png")
            plt.savefig(plot_filename, dpi=DPI, bbox_inches='tight')
            plt.close(fig)
            plot_count += 1

            if plot_count % 10 == 0:
                print(f"  Generated {plot_count} plots...")

    print(f"\nSaved {plot_count} plots to {output_dir}")


def evaluate_on_test_data(data_list, model_left, model_right, feature_names,
                          model_type=_MODEL_TYPE_RF, feature_scaler=None):
    """Evaluate ML models on test data and calculate border metrics."""
    print("\n" + "=" * 60)
    print("Evaluating ML Models on Test Data")
    print("=" * 60)

    all_left_errors = []
    all_right_errors = []
    all_width_errors = []
    total_predicted = 0
    total_true = 0
    total_matched = 0

    for file_idx, data in enumerate(data_list):
        predicted_borders = data['predicted_borders']
        true_borders = data['true_borders']
        dataset_timestamps = pd.to_datetime(data['dataset_timestamps'])
        op_timestamps = pd.to_datetime(data['op_timestamps']) if len(data['op_timestamps']) > 0 else None
        op_values = data['op_values'] if len(data['op_values']) > 0 else None
        current_values = data['current_values']

        predicted_events = [(pd.to_datetime(b[0]), pd.to_datetime(b[1])) for b in predicted_borders]
        true_events = [(pd.to_datetime(b[0]), pd.to_datetime(b[1])) for b in true_borders]

        total_predicted += len(predicted_events)
        total_true += len(true_events)

        if op_timestamps is not None and op_values is not None and len(op_timestamps) > 0:
            refined_predicted_events = refine_event_boundaries_ml(
                predicted_events,
                op_timestamps,
                op_values,
                dataset_timestamps,
                current_values,
                model_left,
                model_right,
                feature_names,
                model_type=model_type,
                feature_scaler=feature_scaler
            )
        else:
            refined_predicted_events = predicted_events

        matches = match_events_efficiently(refined_predicted_events, true_events)
        total_matched += len(matches)

        metrics = calculate_border_metrics(refined_predicted_events, true_events, matches, dataset_timestamps)

        all_left_errors.extend(metrics.get('left_errors', []))
        all_right_errors.extend(metrics.get('right_errors', []))
        all_width_errors.extend(metrics.get('width_errors', []))

        print(f"File {file_idx + 1}: {len(matches)} matched events")

    if len(all_left_errors) > 0:
        overall_left_rmse = np.sqrt(np.mean(np.array(all_left_errors) ** 2))
        overall_right_rmse = np.sqrt(np.mean(np.array(all_right_errors) ** 2))
        overall_width_rmse = np.sqrt(np.mean(np.array(all_width_errors) ** 2))
    else:
        overall_left_rmse = np.nan
        overall_right_rmse = np.nan
        overall_width_rmse = np.nan

    print("\n" + "=" * 60)
    print("Overall Evaluation Results")
    print("=" * 60)
    print(f"Total predicted events: {total_predicted}")
    print(f"Total true events: {total_true}")
    print(f"Total matched events: {total_matched}")
    print(f"False triggers: {total_predicted - total_matched}")
    print(f"Missed events: {total_true - total_matched}")
    if len(all_left_errors) > 0:
        print(f"\nBorder Refinement Metrics:")
        print(f"  Left border RMSE: {overall_left_rmse:.2f} time steps")
        print(f"  Right border RMSE: {overall_right_rmse:.2f} time steps")
        print(f"  Width RMSE: {overall_width_rmse:.2f} time steps")

    return {
        'left_rmse': overall_left_rmse,
        'right_rmse': overall_right_rmse,
        'width_rmse': overall_width_rmse,
        'total_predicted': total_predicted,
        'total_true': total_true,
        'total_matched': total_matched
    }


def save_models(model_left=None, model_right=None, feature_names=None,
                best_params_left=None, best_params_right=None,
                output_dir="border_refinement_ml_models",
                model_ffn=None, feature_scaler=None,
                model_type=_MODEL_TYPE_RF):
    """Save trained models and metadata."""
    os.makedirs(output_dir, exist_ok=True)

    if model_type == _MODEL_TYPE_CNN:
        left_path = os.path.join(output_dir, "model_left.keras")
        right_path = os.path.join(output_dir, "model_right.keras")
        scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
        model_left.save(left_path)
        model_right.save(right_path)
        print(f"  CNN left model saved to {left_path}")
        print(f"  CNN right model saved to {right_path}")
        with open(scaler_path, 'wb') as f:
            pickle.dump(feature_scaler, f)
        print(f"  Channel scaler saved to {scaler_path}")
    elif model_type == _MODEL_TYPE_FFN:
        ffn_path    = os.path.join(output_dir, "model_ffn.keras")
        scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
        model_ffn.save(ffn_path)
        print(f"  FFN model saved to {ffn_path}")
        with open(scaler_path, 'wb') as f:
            pickle.dump(feature_scaler, f)
        print(f"  Feature scaler saved to {scaler_path}")
    else:
        with open(os.path.join(output_dir, "model_left.pkl"), 'wb') as f:
            pickle.dump(model_left, f)
        with open(os.path.join(output_dir, "model_right.pkl"), 'wb') as f:
            pickle.dump(model_right, f)
        print(f"  RF models saved to {output_dir}/model_left.pkl and model_right.pkl")

        if best_params_left is not None or best_params_right is not None:
            hyperparams = {
                'best_params_left':  best_params_left,
                'best_params_right': best_params_right
            }
            with open(os.path.join(output_dir, "best_hyperparameters.pkl"), 'wb') as f:
                pickle.dump(hyperparams, f)
            print(f"  Hyperparameters saved to {output_dir}/best_hyperparameters.pkl")

    with open(os.path.join(output_dir, "feature_names.pkl"), 'wb') as f:
        pickle.dump(feature_names if feature_names is not None else [], f)

    metadata = {"target_type": _TARGET_TYPE, "model_type": model_type}
    metadata_path = os.path.join(output_dir, _METADATA_FILENAME)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    print(f"  Metadata saved to {metadata_path}")

    print(f"\nModels saved to {output_dir}/")


def main():
    BORDER_REFINEMENT_DATA_DIR = "border_refinement_set"

    MODEL_TYPE = _MODEL_TYPE_FFN

    FFN_EPOCHS       = 300
    FFN_BATCH_SIZE   = 32
    FFN_LEARNING_RATE = 1e-3
    FFN_PATIENCE     = 30
    FFN_HIDDEN_UNITS = [256, 128, 64]

    USE_HYPERPARAMETER_TUNING = True
    HYPERPARAMETER_CONFIG = {
        'n_estimators':     {'min': 50, 'max': 300},
        'max_depth':        [None, 10, 15, 20, 25, 30, 35, 40],
        'min_samples_split': {'min': 2, 'max': 20},
        'min_samples_leaf': {'min': 1, 'max': 10},
        'max_features':     ['sqrt', 'log2', None],
        'bootstrap':        [True, False]
    }
    N_ITER   = 50
    CV_FOLDS = 5

    TEST_SIZE    = 0.2
    RANDOM_STATE = 42

    MODEL_OUTPUT_DIR = "border_refinement_ml_models"
    PLOTS_OUTPUT_DIR = "border_refinement_ml_plots"

    print("=" * 60)
    print("Border Refinement ML Training Script")
    print("=" * 60)
    print(f"\nModel type: {MODEL_TYPE.upper()}")
    if MODEL_TYPE == _MODEL_TYPE_FFN:
        print(f"  Epochs: {FFN_EPOCHS}  Batch: {FFN_BATCH_SIZE}  LR: {FFN_LEARNING_RATE}  Patience: {FFN_PATIENCE}")
        print(f"  Hidden units: {FFN_HIDDEN_UNITS}")
    else:
        if USE_HYPERPARAMETER_TUNING:
            print(f"  Hyperparameter tuning: ENABLED  (n_iter={N_ITER}, cv={CV_FOLDS})")
        else:
            print(f"  Hyperparameter tuning: DISABLED (using default parameters)")

    print(f"\nLoading border refinement data from: {BORDER_REFINEMENT_DATA_DIR}")
    data_list = load_border_refinement_data(BORDER_REFINEMENT_DATA_DIR)

    X, y_left, y_right, feature_names = prepare_training_data(data_list)

    training_results = train_models(
        X, y_left, y_right,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        model_type=MODEL_TYPE,
        ffn_epochs=FFN_EPOCHS,
        ffn_batch_size=FFN_BATCH_SIZE,
        ffn_learning_rate=FFN_LEARNING_RATE,
        ffn_patience=FFN_PATIENCE,
        ffn_hidden_units=FFN_HIDDEN_UNITS,
        use_hyperparameter_tuning=USE_HYPERPARAMETER_TUNING,
        hyperparameter_config=HYPERPARAMETER_CONFIG if USE_HYPERPARAMETER_TUNING else None,
        n_iter=N_ITER,
        cv=CV_FOLDS
    )

    actual_model_type = training_results.get('model_type', MODEL_TYPE)

    save_models(
        model_left=training_results.get('model_left'),
        model_right=training_results.get('model_right'),
        feature_names=feature_names,
        best_params_left=training_results.get('best_params_left'),
        best_params_right=training_results.get('best_params_right'),
        output_dir=MODEL_OUTPUT_DIR,
        model_ffn=training_results.get('model_ffn'),
        feature_scaler=training_results.get('feature_scaler'),
        model_type=actual_model_type
    )

    evaluation_results = evaluate_on_test_data(
        data_list,
        training_results.get('model_left'),
        training_results.get('model_right'),
        feature_names,
        model_type=actual_model_type,
        feature_scaler=training_results.get('feature_scaler')
    )

    plot_refinement_results(
        data_list,
        training_results.get('model_left'),
        training_results.get('model_right'),
        feature_names,
        output_dir=PLOTS_OUTPUT_DIR,
        model_type=actual_model_type,
        feature_scaler=training_results.get('feature_scaler')
    )

    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(PLOTS_OUTPUT_DIR, "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ML BORDER REFINEMENT EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write("Configuration:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Data directory: {BORDER_REFINEMENT_DATA_DIR}\n")
        f.write(f"Hyperparameter tuning: {USE_HYPERPARAMETER_TUNING}\n")
        if USE_HYPERPARAMETER_TUNING:
            f.write(f"Search iterations: {N_ITER}\n")
            f.write(f"CV folds: {CV_FOLDS}\n")
        f.write(f"Test size: {TEST_SIZE}\n")
        f.write(f"Random state: {RANDOM_STATE}\n")
        f.write("\n")

        f.write(f"Model Performance (Test Set):\n")
        f.write("-" * 60 + "\n")
        f.write(f"Left Border - RMSE: {training_results['metrics_left_test']['rmse']:.2f}, "
                f"MAE: {training_results['metrics_left_test']['mae']:.2f}, "
                f"R²: {training_results['metrics_left_test']['r2']:.4f}\n")
        f.write(f"Right Border - RMSE: {training_results['metrics_right_test']['rmse']:.2f}, "
                f"MAE: {training_results['metrics_right_test']['mae']:.2f}, "
                f"R²: {training_results['metrics_right_test']['r2']:.4f}\n")

        if training_results.get('best_params_left') is not None:
            f.write("\n")
            f.write("Best Hyperparameters:\n")
            f.write("-" * 60 + "\n")
            f.write("Left Border Model:\n")
            for param, value in training_results['best_params_left'].items():
                f.write(f"  {param}: {value}\n")
            if training_results.get('cv_score_left') is not None:
                f.write(f"  CV RMSE: {np.sqrt(training_results['cv_score_left']):.4f}\n")
            f.write("\nRight Border Model:\n")
            for param, value in training_results['best_params_right'].items():
                f.write(f"  {param}: {value}\n")
            if training_results.get('cv_score_right') is not None:
                f.write(f"  CV RMSE: {np.sqrt(training_results['cv_score_right']):.4f}\n")

        f.write("\n")
        f.write(f"Border Refinement Metrics (All Data):\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total predicted events: {evaluation_results['total_predicted']}\n")
        f.write(f"Total true events: {evaluation_results['total_true']}\n")
        f.write(f"Total matched events: {evaluation_results['total_matched']}\n")
        f.write(f"Left border RMSE: {evaluation_results['left_rmse']:.2f} time steps\n")
        f.write(f"Right border RMSE: {evaluation_results['right_rmse']:.2f} time steps\n")
        f.write(f"Width RMSE: {evaluation_results['width_rmse']:.2f} time steps\n")

    print(f"\nEvaluation: {results_path}")


if __name__ == "__main__":
    main()
