"""
Visualize per-layer feature maps and kernels for a specified input sample.

This helps researchers inspect what a chosen layer is extracting from a
specific mask input.

Example:
    python visualize_layer_features_kernels.py \
      --model litho_model_780_e200.keras \
      --dataset litho_dataset_780.npz \
      --sample-idx 0 \
      --layer circular_conv2d_3 \
      --output-dir layer_viz
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shift_equivariant_unet import (
    CircularPad2D,
    CircularConv2D,
    DilatedCircularConv2D,
)


def normalize01(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)


def load_sample(dataset_path, sample_idx):
    data = np.load(dataset_path)
    if "masks" not in data:
        raise KeyError(f"{dataset_path} must contain key 'masks'")
    masks = data["masks"].astype(np.float32)
    n = len(masks)
    if sample_idx < 0:
        sample_idx = int(np.random.randint(0, n))
    if sample_idx >= n:
        raise IndexError(f"sample_idx={sample_idx} out of range [0, {n-1}]")
    x = masks[sample_idx]
    y = data["aerials"][sample_idx].astype(np.float32) if "aerials" in data else None
    return x, y, n, sample_idx


def _find_kernel_from_layer(layer):
    """
    Try to extract a conv kernel for a given layer.
    Returns (kernel_np, source_name) or (None, None).
    """
    # Direct weights
    try:
        ws = layer.get_weights()
        if ws and ws[0].ndim == 4:
            return ws[0], layer.name
    except Exception:
        pass

    # Custom wrappers (CircularConv2D / DilatedCircularConv2D)
    if hasattr(layer, "conv"):
        try:
            ws = layer.conv.get_weights()
            if ws and ws[0].ndim == 4:
                return ws[0], f"{layer.name}.conv"
        except Exception:
            pass

    # Nested sublayers: first conv-like with 4D kernel
    if hasattr(layer, "submodules"):
        for sub in layer.submodules:
            if sub is layer:
                continue
            if hasattr(sub, "kernel"):
                try:
                    k = sub.kernel.numpy()
                    if k.ndim == 4:
                        return k, sub.name
                except Exception:
                    continue
    return None, None


def save_feature_maps_grid(output_dir, sample_idx, layer_name, fmap, max_maps=32):
    # fmap: [H, W, C]
    h, w, c = fmap.shape
    max_maps = min(max_maps, c)
    energy = np.mean(np.abs(fmap), axis=(0, 1))
    top_idx = np.argsort(energy)[::-1][:max_maps]

    cols = min(8, max_maps)
    rows = int(np.ceil(max_maps / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.1))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    k = 0
    for r in range(rows):
        for c2 in range(cols):
            ax = axes[r, c2]
            if k < len(top_idx):
                ch = int(top_idx[k])
                img = normalize01(fmap[:, :, ch])
                ax.imshow(img, cmap="viridis")
                ax.set_title(f"ch {ch}", fontsize=7)
            ax.axis("off")
            k += 1

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"sample_{sample_idx:04d}_{layer_name}_feature_maps.png")
    plt.savefig(out_path, dpi=180)
    plt.close()

    # Feature channel response bar chart
    bar_path = os.path.join(output_dir, f"sample_{sample_idx:04d}_{layer_name}_feature_energy.png")
    top_plot = min(64, len(energy))
    ord_idx = np.argsort(energy)[::-1][:top_plot]
    plt.figure(figsize=(10, 3.2))
    plt.bar(np.arange(top_plot), energy[ord_idx])
    plt.title(f"Feature channel energy (top {top_plot}) - {layer_name}")
    plt.xlabel("Ranked channel")
    plt.ylabel("Mean abs activation")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160)
    plt.close()

    return out_path, bar_path, top_idx.tolist(), energy


def save_kernels_grid(output_dir, sample_idx, layer_name, kernel, max_kernels=32):
    # kernel: [Kh, Kw, Cin, Cout]
    kh, kw, cin, cout = kernel.shape
    # summarize each output kernel as mean over input channels
    kernel_vis = np.mean(np.abs(kernel), axis=2)  # [Kh, Kw, Cout]
    norms = np.sqrt(np.sum(kernel ** 2, axis=(0, 1, 2)))
    order = np.argsort(norms)[::-1]
    top = order[: min(max_kernels, cout)]

    cols = min(8, len(top))
    rows = int(np.ceil(len(top) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    k = 0
    for r in range(rows):
        for c2 in range(cols):
            ax = axes[r, c2]
            if k < len(top):
                ch = int(top[k])
                img = normalize01(kernel_vis[:, :, ch])
                ax.imshow(img, cmap="gray")
                ax.set_title(f"k {ch}", fontsize=7)
            ax.axis("off")
            k += 1

    plt.tight_layout()
    kernel_path = os.path.join(output_dir, f"sample_{sample_idx:04d}_{layer_name}_kernels.png")
    plt.savefig(kernel_path, dpi=180)
    plt.close()

    norm_path = os.path.join(output_dir, f"sample_{sample_idx:04d}_{layer_name}_kernel_norms.png")
    top_plot = min(64, len(norms))
    ord_idx = np.argsort(norms)[::-1][:top_plot]
    plt.figure(figsize=(10, 3.2))
    plt.bar(np.arange(top_plot), norms[ord_idx])
    plt.title(f"Kernel L2 norm (top {top_plot}) - {layer_name}")
    plt.xlabel("Ranked output kernel")
    plt.ylabel("L2 norm")
    plt.tight_layout()
    plt.savefig(norm_path, dpi=160)
    plt.close()

    return kernel_path, norm_path, top.tolist(), norms


def save_input_prediction_panel(output_dir, sample_idx, x, y, pred, layer_name):
    x2d = x[:, :, 0]
    pred2d = pred[:, :, 0]
    gt2d = None if y is None else y[:, :, 0]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    axes[0].imshow(x2d, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input mask")
    axes[1].imshow(pred2d, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Prediction")
    if gt2d is None:
        axes[2].imshow(np.zeros_like(x2d), cmap="gray")
        axes[2].set_title("Ground truth (N/A)")
        axes[3].imshow(np.zeros_like(x2d), cmap="gray")
        axes[3].set_title("Abs error (N/A)")
    else:
        axes[2].imshow(gt2d, cmap="inferno", vmin=0, vmax=1)
        axes[2].set_title("Ground truth")
        axes[3].imshow(np.abs(pred2d - gt2d), cmap="magma")
        axes[3].set_title("Abs error")
    for ax in axes:
        ax.axis("off")
    plt.suptitle(f"Sample {sample_idx} - layer {layer_name}")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"sample_{sample_idx:04d}_{layer_name}_overview.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visualize feature maps and kernels at a specified layer.")
    parser.add_argument("--model", type=str, required=True, help="Path to .keras model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index; -1 for random")
    parser.add_argument("--layer", type=str, required=True, help="Layer name to inspect")
    parser.add_argument("--max-feature-maps", type=int, default=32, help="Max feature channels to visualize")
    parser.add_argument("--max-kernels", type=int, default=32, help="Max kernels(out channels) to visualize")
    parser.add_argument("--output-dir", type=str, default="layer_viz", help="Output directory")
    args = parser.parse_args()

    custom_objects = {
        "CircularPad2D": CircularPad2D,
        "CircularConv2D": CircularConv2D,
        "DilatedCircularConv2D": DilatedCircularConv2D,
    }
    model = keras.models.load_model(args.model, custom_objects=custom_objects)
    target_layer = model.get_layer(args.layer)

    x, y, n_total, idx = load_sample(args.dataset, args.sample_idx)
    x_batch = x[None, ...]
    pred = model(x_batch, training=False).numpy()[0]

    feat_model = keras.Model(inputs=model.inputs, outputs=target_layer.output)
    fmap = feat_model(x_batch, training=False).numpy()[0]
    if fmap.ndim != 3:
        raise RuntimeError(
            f"Layer '{args.layer}' output rank is {fmap.ndim + 1} with batch; expected 4D tensor."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    overview_path = save_input_prediction_panel(args.output_dir, idx, x, y, pred, args.layer)
    feature_path, feature_energy_path, top_feature_idx, feature_energy = save_feature_maps_grid(
        args.output_dir, idx, args.layer, fmap, max_maps=args.max_feature_maps
    )

    kernel, kernel_source = _find_kernel_from_layer(target_layer)
    kernel_path = None
    kernel_norm_path = None
    top_kernel_idx = []
    kernel_norms = None
    if kernel is not None and kernel.ndim == 4:
        kernel_path, kernel_norm_path, top_kernel_idx, kernel_norms = save_kernels_grid(
            args.output_dir, idx, args.layer, kernel, max_kernels=args.max_kernels
        )

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": os.path.abspath(args.model),
        "dataset": os.path.abspath(args.dataset),
        "dataset_size": int(n_total),
        "sample_index": int(idx),
        "layer": args.layer,
        "feature_shape_hwc": list(map(int, fmap.shape)),
        "feature_stats": {
            "min": float(np.min(fmap)),
            "max": float(np.max(fmap)),
            "mean": float(np.mean(fmap)),
            "std": float(np.std(fmap)),
        },
        "top_feature_channels_by_energy": [int(v) for v in top_feature_idx],
        "prediction_mse_to_gt": None if y is None else float(np.mean((pred - y) ** 2)),
        "kernel": {
            "found": kernel is not None,
            "source": kernel_source,
            "shape_kh_kw_cin_cout": None if kernel is None else list(map(int, kernel.shape)),
            "top_kernel_indices_by_norm": [int(v) for v in top_kernel_idx],
        },
        "outputs": {
            "overview_png": os.path.abspath(overview_path),
            "feature_maps_png": os.path.abspath(feature_path),
            "feature_energy_png": os.path.abspath(feature_energy_path),
            "kernels_png": None if kernel_path is None else os.path.abspath(kernel_path),
            "kernel_norms_png": None if kernel_norm_path is None else os.path.abspath(kernel_norm_path),
        },
    }
    if kernel_norms is not None:
        report["kernel"]["norm_stats"] = {
            "min": float(np.min(kernel_norms)),
            "max": float(np.max(kernel_norms)),
            "mean": float(np.mean(kernel_norms)),
            "std": float(np.std(kernel_norms)),
        }
    report_path = os.path.join(args.output_dir, f"sample_{idx:04d}_{args.layer}_layer_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Layer visualization complete.")
    print(f"Layer: {args.layer}")
    print(f"Overview: {overview_path}")
    print(f"Feature maps: {feature_path}")
    if kernel_path is not None:
        print(f"Kernels: {kernel_path}")
    else:
        print("Kernels: not found for this layer")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
