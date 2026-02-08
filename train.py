"""
Training script for the shift-equivariant U-Net on lithography data.

Task: predict the aerial image (continuous [0,1]) from a binary mask.
Loss: MSE (pixel-wise regression).

Usage:
    # Generate dataset and train (default 200 samples, 50 epochs):
    python train.py

    # Custom settings:
    python train.py --num-samples 500 --epochs 100 --batch-size 16

    # Train from existing dataset:
    python train.py --dataset litho_dataset.npz --epochs 50
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
import csv
import signal
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from shift_equivariant_unet import (
    shift_equivariant_unet,
    CircularPad2D,
    CircularConv2D,
    DilatedCircularConv2D,
)
from data_pipeline import (
    generate_dataset,
    save_dataset,
    load_dataset,
    make_tf_dataset,
)


class StopState:
    def __init__(self):
        self.requested = False
        self.reason = None


class StopOnSignalCallback(keras.callbacks.Callback):
    def __init__(self, stop_state):
        super().__init__()
        self.stop_state = stop_state

    def on_epoch_end(self, epoch, logs=None):
        if self.stop_state.requested:
            print(f"\nStop requested ({self.stop_state.reason}), stopping after epoch {epoch + 1}.")
            self.model.stop_training = True


class JsonlLoggerCallback(keras.callbacks.Callback):
    def __init__(self, jsonl_path):
        super().__init__()
        self.jsonl_path = jsonl_path

    def on_epoch_end(self, epoch, logs=None):
        payload = {"epoch": int(epoch + 1)}
        if logs:
            for k, v in logs.items():
                try:
                    payload[k] = float(v)
                except Exception:
                    pass
        with open(self.jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload) + '\n')


def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _resolve_path(base_dir, path):
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def build_model(input_shape=(64, 64, 1), num_filters_base=32):
    """Build a smaller shift-equivariant U-Net for 64x64 litho data.

    Uses reduced filter counts compared to the full model since
    64x64 inputs don't need as much capacity.
    """
    inputs = keras.Input(shape=input_shape)

    # Encoder
    x = CircularConv2D(num_filters_base, 3, activation='relu')(inputs)
    x = CircularConv2D(num_filters_base, 3, activation='relu')(x)
    feat1 = x

    x = DilatedCircularConv2D(num_filters_base * 2, dilation_rate=2)(feat1)
    x = DilatedCircularConv2D(num_filters_base * 2, dilation_rate=2)(x)
    feat2 = x

    x = DilatedCircularConv2D(num_filters_base * 4, dilation_rate=4)(feat2)
    x = DilatedCircularConv2D(num_filters_base * 4, dilation_rate=4)(x)
    feat3 = x

    # Bottleneck
    x = DilatedCircularConv2D(num_filters_base * 8, dilation_rate=8)(feat3)
    x = DilatedCircularConv2D(num_filters_base * 8, dilation_rate=8)(x)

    # Decoder
    x = keras.layers.Concatenate()([x, feat3])
    x = DilatedCircularConv2D(num_filters_base * 4, dilation_rate=4)(x)
    x = DilatedCircularConv2D(num_filters_base * 4, dilation_rate=4)(x)

    x = keras.layers.Concatenate()([x, feat2])
    x = DilatedCircularConv2D(num_filters_base * 2, dilation_rate=2)(x)
    x = DilatedCircularConv2D(num_filters_base * 2, dilation_rate=2)(x)

    x = keras.layers.Concatenate()([x, feat1])
    x = CircularConv2D(num_filters_base, 3, activation='relu')(x)
    x = CircularConv2D(num_filters_base, 3, activation='relu')(x)

    # Output: single channel, sigmoid for [0,1] range
    outputs = CircularConv2D(1, 1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs,
                        name='litho_unet')
    return model


def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training/validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history saved to {save_path}")


def plot_predictions(model, masks, aerials, num_samples=4,
                     save_path='predictions.png'):
    """Plot model predictions vs ground truth."""
    num_samples = min(num_samples, len(masks))
    indices = np.random.choice(len(masks), size=num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        mask_input = masks[idx:idx + 1]
        pred = model(mask_input, training=False).numpy()[0, :, :, 0]
        gt = aerials[idx, :, :, 0]
        mask_vis = masks[idx, :, :, 0]

        axes[row, 0].imshow(mask_vis, cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_title('Input Mask')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(gt, cmap='inferno', vmin=0, vmax=1)
        axes[row, 1].set_title('Ground Truth')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(pred, cmap='inferno', vmin=0, vmax=1)
        axes[row, 2].set_title(f'Prediction (MSE={np.mean((pred-gt)**2):.5f})')
        axes[row, 2].axis('off')

    plt.suptitle('Model Predictions vs Ground Truth', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Predictions saved to {save_path}")


def fit_with_numpy_batches(model, train_masks, train_aerials,
                           val_masks, val_aerials, epochs, batch_size,
                           save_every_epochs=0, checkpoint_dir='checkpoints',
                           checkpoint_prefix='litho_model',
                           best_model_path=None,
                           csv_log_path=None,
                           jsonl_log_path=None,
                           stop_state=None):
    """Fallback training loop for thread-constrained Docker environments.

    Uses train_on_batch/test_on_batch directly and avoids tf.data private
    threadpools that can fail under strict container limits.
    """
    history = {'loss': [], 'mae': [], 'val_loss': [], 'val_mae': []}
    best_val = float('inf')
    if csv_log_path:
        _ensure_parent(csv_log_path)
        with open(csv_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'mae', 'val_loss', 'val_mae'])
    if jsonl_log_path:
        _ensure_parent(jsonl_log_path)
        open(jsonl_log_path, 'w', encoding='utf-8').close()

    n_train = len(train_masks)
    n_val = len(val_masks)

    for epoch in range(epochs):
        if stop_state is not None and stop_state.requested:
            print(f"Stop requested ({stop_state.reason}), exiting at epoch boundary.")
            break

        order = np.random.permutation(n_train)
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        train_seen = 0

        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            xb = train_masks[idx]
            yb = train_aerials[idx]
            metrics = model.train_on_batch(xb, yb, return_dict=True)
            bs = len(idx)
            train_loss_sum += float(metrics['loss']) * bs
            train_mae_sum += float(metrics['mae']) * bs
            train_seen += bs

        train_loss = train_loss_sum / max(1, train_seen)
        train_mae = train_mae_sum / max(1, train_seen)

        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_seen = 0
        for start in range(0, n_val, batch_size):
            idx = slice(start, min(start + batch_size, n_val))
            xb = val_masks[idx]
            yb = val_aerials[idx]
            metrics = model.test_on_batch(xb, yb, return_dict=True)
            bs = len(xb)
            val_loss_sum += float(metrics['loss']) * bs
            val_mae_sum += float(metrics['mae']) * bs
            val_seen += bs

        val_loss = val_loss_sum / max(1, val_seen)
        val_mae = val_mae_sum / max(1, val_seen)

        history['loss'].append(train_loss)
        history['mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"loss: {train_loss:.6f} - mae: {train_mae:.6f} - "
              f"val_loss: {val_loss:.6f} - val_mae: {val_mae:.6f}")
        if save_every_epochs > 0 and (epoch + 1) % save_every_epochs == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(
                checkpoint_dir, f"{checkpoint_prefix}_epoch{epoch + 1:04d}.keras"
            )
            model.save(ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
        if best_model_path is not None and val_loss < best_val:
            best_val = val_loss
            _ensure_parent(best_model_path)
            model.save(best_model_path)
            print(f"Best model updated: {best_model_path} (val_loss={best_val:.6f})")

        if csv_log_path:
            with open(csv_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, train_loss, train_mae, val_loss, val_mae])
        if jsonl_log_path:
            with open(jsonl_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'epoch': int(epoch + 1),
                    'loss': float(train_loss),
                    'mae': float(train_mae),
                    'val_loss': float(val_loss),
                    'val_mae': float(val_mae),
                }) + '\n')

    class _History:
        def __init__(self, hist):
            self.history = hist
    return _History(history)


def evaluate_with_numpy_batches(model, val_masks, val_aerials, batch_size):
    """Evaluate model without tf.data to avoid private threadpool creation."""
    n_val = len(val_masks)
    val_loss_sum = 0.0
    val_mae_sum = 0.0
    val_seen = 0

    for start in range(0, n_val, batch_size):
        idx = slice(start, min(start + batch_size, n_val))
        xb = val_masks[idx]
        yb = val_aerials[idx]
        metrics = model.test_on_batch(xb, yb, return_dict=True)
        bs = len(xb)
        val_loss_sum += float(metrics['loss']) * bs
        val_mae_sum += float(metrics['mae']) * bs
        val_seen += bs

    return (
        val_loss_sum / max(1, val_seen),
        val_mae_sum / max(1, val_seen),
    )


def main():
    parser = argparse.ArgumentParser(
        description='Train shift-equivariant U-Net on lithography data'
    )
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to existing .npz dataset')
    parser.add_argument('--num-samples', type=int, default=200,
                        help='Number of training samples to generate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--filters', type=int, default=32,
                        help='Base filter count')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-model', type=str, default='litho_model.keras',
                        help='Path to save trained model')
    parser.add_argument('--docker-safe', action='store_true',
                        help='Use numpy batch training loop to avoid '
                             'tf.data threadpool issues in constrained '
                             'Docker environments')
    parser.add_argument('--intra-threads', type=int, default=None,
                        help='TensorFlow intra-op thread count')
    parser.add_argument('--inter-threads', type=int, default=None,
                        help='TensorFlow inter-op thread count')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Enable conservative runtime settings for '
                             'quick container checks (threads=1, XLA off, '
                             'and docker-safe training path)')
    parser.add_argument('--save-every-epochs', type=int, default=0,
                        help='Save checkpoint every N epochs (0 disables)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to write periodic checkpoints')
    parser.add_argument('--experiment-dir', type=str, default='experiments',
                        help='Root directory for experiment runs/logs')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Optional run folder name (default timestamp)')
    args = parser.parse_args()

    if args.smoke_test and not args.docker_safe:
        args.docker_safe = True
        print("Smoke-test mode enabled: forcing --docker-safe training path.")
    if args.save_every_epochs < 0:
        raise ValueError("--save-every-epochs must be >= 0")

    run_name = args.run_name or time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.experiment_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    stop_state = StopState()

    def _signal_handler(sig, _frame):
        stop_state.requested = True
        stop_state.reason = signal.Signals(sig).name
        print(f"\nReceived {stop_state.reason}. Will stop after current epoch and save latest artifacts.")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    resolved_save_model = _resolve_path(run_dir, args.save_model)
    resolved_checkpoint_dir = _resolve_path(run_dir, args.checkpoint_dir)
    model_basename = os.path.splitext(os.path.basename(resolved_save_model))[0]
    history_png_path = os.path.join(run_dir, 'training_history.png')
    pred_png_path = os.path.join(run_dir, 'predictions.png')
    csv_log_path = os.path.join(run_dir, 'training_log.csv')
    jsonl_log_path = os.path.join(run_dir, 'epoch_metrics.jsonl')
    best_model_path = os.path.join(run_dir, f'{model_basename}_best.keras')
    interrupted_model_path = os.path.join(run_dir, f'{model_basename}_interrupted.keras')
    summary_json_path = os.path.join(run_dir, 'run_summary.json')
    config_json_path = os.path.join(run_dir, 'run_config.json')

    with open(config_json_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Run directory: {run_dir}")
    print(f"Run config saved to {config_json_path}")

    # Configure TF thread usage before creating tensors/models.
    intra_threads = args.intra_threads
    inter_threads = args.inter_threads
    if args.smoke_test:
        if intra_threads is None:
            intra_threads = 1
        if inter_threads is None:
            inter_threads = 1

    if intra_threads is not None:
        tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
    if inter_threads is not None:
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
    if args.smoke_test:
        # Disable XLA JIT to reduce runtime thread pressure during smoke tests.
        tf.config.optimizer.set_jit(False)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # ----- Data -----
    dataset_path_for_report = None
    if args.dataset and os.path.exists(args.dataset):
        dataset_path_for_report = os.path.abspath(args.dataset)
        print(f"Loading dataset from {args.dataset}...")
        masks, aerials = load_dataset(args.dataset)
    else:
        print(f"Generating {args.num_samples} training samples...")
        masks, aerials = generate_dataset(
            args.num_samples, seed=args.seed
        )
        generated_dataset_path = os.path.join(run_dir, 'litho_dataset.npz')
        save_dataset(masks, aerials, generated_dataset_path)
        dataset_path_for_report = os.path.abspath(generated_dataset_path)

    print(f"Dataset: {masks.shape[0]} samples, "
          f"mask shape: {masks.shape[1:]}, "
          f"aerial shape: {aerials.shape[1:]}")

    # Train/val split
    n = len(masks)
    n_val = max(1, int(n * args.val_split))
    n_train = n - n_val

    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_masks, train_aerials = masks[train_idx], aerials[train_idx]
    val_masks, val_aerials = masks[val_idx], aerials[val_idx]

    print(f"Train: {n_train}, Val: {n_val}")

    if not args.docker_safe:
        train_ds = make_tf_dataset(train_masks, train_aerials,
                                   batch_size=args.batch_size, shuffle=True)
        val_ds = make_tf_dataset(val_masks, val_aerials,
                                 batch_size=args.batch_size, shuffle=False)

    # ----- Model -----
    input_shape = masks.shape[1:]  # (64, 64, 1)
    model = build_model(input_shape=input_shape,
                        num_filters_base=args.filters)
    print(f"\nModel: {model.count_params():,} parameters")
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss='mse',
        metrics=['mae'],
    )

    # ----- Train -----
    print(f"\nTraining for {args.epochs} epochs...")
    interrupted = False
    try:
        if args.docker_safe:
            history = fit_with_numpy_batches(
                model, train_masks, train_aerials,
                val_masks, val_aerials,
                epochs=args.epochs, batch_size=args.batch_size,
                save_every_epochs=args.save_every_epochs,
                checkpoint_dir=resolved_checkpoint_dir,
                checkpoint_prefix=model_basename,
                best_model_path=best_model_path,
                csv_log_path=csv_log_path,
                jsonl_log_path=jsonl_log_path,
                stop_state=stop_state,
            )
        else:
            callbacks = [
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6,
                    verbose=1
                ),
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=20, restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath=best_model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1,
                ),
                keras.callbacks.CSVLogger(csv_log_path, append=False),
                JsonlLoggerCallback(jsonl_log_path),
                StopOnSignalCallback(stop_state),
            ]
            if args.save_every_epochs > 0:
                os.makedirs(resolved_checkpoint_dir, exist_ok=True)
                def _save_periodic(epoch, logs):
                    epoch_n = epoch + 1
                    if epoch_n % args.save_every_epochs == 0:
                        ckpt_path = os.path.join(
                            resolved_checkpoint_dir,
                            f"{model_basename}_epoch{epoch_n:04d}.keras",
                        )
                        model.save(ckpt_path)
                        print(f"Checkpoint saved to {ckpt_path}")
                callbacks.append(keras.callbacks.LambdaCallback(on_epoch_end=_save_periodic))

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.epochs,
                callbacks=callbacks,
                verbose=1,
            )
    except KeyboardInterrupt:
        interrupted = True
        stop_state.requested = True
        stop_state.reason = 'KeyboardInterrupt'
        print("\nKeyboardInterrupt received. Saving latest artifacts...")
        class _History:
            def __init__(self):
                self.history = {}
        history = _History()

    # ----- Evaluate -----
    val_loss, val_mae = None, None
    try:
        if args.docker_safe:
            val_loss, val_mae = evaluate_with_numpy_batches(
                model, val_masks, val_aerials, args.batch_size
            )
        else:
            val_loss, val_mae = model.evaluate(val_ds, verbose=0)
        print(f"\nFinal val loss (MSE): {val_loss:.6f}")
        print(f"Final val MAE: {val_mae:.6f}")
    except Exception as e:
        print(f"Evaluation skipped due to error: {e}")

    # ----- Save -----
    _ensure_parent(resolved_save_model)
    model.save(resolved_save_model)
    print(f"Model saved to {resolved_save_model}")
    if stop_state.requested or interrupted:
        model.save(interrupted_model_path)
        print(f"Interrupted snapshot saved to {interrupted_model_path}")

    # ----- Plots -----
    if 'loss' in history.history and len(history.history.get('loss', [])) > 0:
        plot_training_history(history, save_path=history_png_path)
    plot_predictions(model, val_masks, val_aerials, num_samples=4, save_path=pred_png_path)

    run_summary = {
        'run_dir': os.path.abspath(run_dir),
        'dataset': dataset_path_for_report,
        'status': 'completed' if not stop_state.requested and not interrupted else 'interrupted',
        'stop_reason': stop_state.reason,
        'epochs_ran': int(len(history.history.get('loss', []))) if hasattr(history, 'history') else 0,
        'final_val_loss': None if val_loss is None else float(val_loss),
        'final_val_mae': None if val_mae is None else float(val_mae),
        'outputs': {
            'model': os.path.abspath(resolved_save_model),
            'best_model': os.path.abspath(best_model_path) if os.path.exists(best_model_path) else None,
            'interrupted_model': os.path.abspath(interrupted_model_path) if os.path.exists(interrupted_model_path) else None,
            'csv_log': os.path.abspath(csv_log_path) if os.path.exists(csv_log_path) else None,
            'jsonl_log': os.path.abspath(jsonl_log_path) if os.path.exists(jsonl_log_path) else None,
            'history_png': os.path.abspath(history_png_path) if os.path.exists(history_png_path) else None,
            'predictions_png': os.path.abspath(pred_png_path) if os.path.exists(pred_png_path) else None,
            'checkpoints_dir': os.path.abspath(resolved_checkpoint_dir) if os.path.exists(resolved_checkpoint_dir) else None,
        },
    }
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(run_summary, f, indent=2)
    print(f"Run summary saved to {summary_json_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
