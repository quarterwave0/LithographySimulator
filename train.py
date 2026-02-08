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

    # Train the TokenizedMLP variant:
    python train.py --model tokenized_mlp --save-model litho_model_mlp.keras
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model_baseline import build_model as build_baseline
from model_tokenized_mlp import build_model as build_tokenized_mlp
from data_pipeline import (
    generate_dataset,
    save_dataset,
    load_dataset,
    make_tf_dataset,
)


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
        pred = model.predict(mask_input, verbose=0)[0, :, :, 0]
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
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline', 'tokenized_mlp'],
                        help='Model variant: baseline or tokenized_mlp')
    parser.add_argument('--mlp-blocks', type=int, default=2,
                        help='Number of TokenizedMLP blocks (tokenized_mlp only)')
    parser.add_argument('--axis-kernel', type=int, default=31,
                        help='Axis conv kernel size (tokenized_mlp only)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # ----- Data -----
    if args.dataset and os.path.exists(args.dataset):
        print(f"Loading dataset from {args.dataset}...")
        masks, aerials = load_dataset(args.dataset)
    else:
        print(f"Generating {args.num_samples} training samples...")
        masks, aerials = generate_dataset(
            args.num_samples, seed=args.seed
        )
        save_dataset(masks, aerials, 'litho_dataset.npz')

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

    train_ds = make_tf_dataset(train_masks, train_aerials,
                               batch_size=args.batch_size, shuffle=True)
    val_ds = make_tf_dataset(val_masks, val_aerials,
                             batch_size=args.batch_size, shuffle=False)

    # ----- Model -----
    input_shape = masks.shape[1:]  # (64, 64, 1)
    if args.model == 'tokenized_mlp':
        model = build_tokenized_mlp(
            input_shape=input_shape,
            num_filters_base=args.filters,
            num_mlp_blocks=args.mlp_blocks,
            axis_kernel=args.axis_kernel,
        )
    else:
        model = build_baseline(input_shape=input_shape,
                               num_filters_base=args.filters)
    print(f"\nModel ({args.model}): {model.count_params():,} parameters")
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss='mse',
        metrics=['mae'],
    )

    # ----- Train -----
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True,
            verbose=1
        ),
    ]

    print(f"\nTraining for {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ----- Evaluate -----
    val_loss, val_mae = model.evaluate(val_ds, verbose=0)
    print(f"\nFinal val loss (MSE): {val_loss:.6f}")
    print(f"Final val MAE: {val_mae:.6f}")

    # ----- Save -----
    model.save(args.save_model)
    print(f"Model saved to {args.save_model}")

    # ----- Plots -----
    plot_training_history(history)
    plot_predictions(model, val_masks, val_aerials, num_samples=4)

    print("\nDone!")


if __name__ == '__main__':
    main()
