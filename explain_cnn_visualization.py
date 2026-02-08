"""
Lightweight CNN/U-Net visualization helper for lithography models.

Features:
1) Grad-CAM
2) Guided Grad-CAM (guided-gradient approximation x Grad-CAM)
3) Score-CAM

Outputs:
- PNG panel and per-method PNGs
- JSON report with basic statistics and perturbation checks

Example:
    python explain_cnn_visualization.py \
        --model litho_model_780_e200.keras \
        --dataset litho_dataset_780.npz \
        --sample-idx 0 \
        --output-dir explain_outputs
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


def choose_target_layer(model, user_layer=None):
    if user_layer:
        _ = model.get_layer(user_layer)
        return user_layer

    def _rank_and_channels(layer):
        try:
            shp = layer.output.shape
            rank = len(shp)
            ch = shp[-1]
            ch = int(ch) if ch is not None else None
            return rank, ch
        except Exception:
            pass
        try:
            shp = layer.output_shape
            rank = len(shp)
            ch = shp[-1]
            ch = int(ch) if ch is not None else None
            return rank, ch
        except Exception:
            return None, None

    chosen = None
    for layer in reversed(model.layers):
        rank, ch = _rank_and_channels(layer)
        if rank is None or rank < 4:
            continue
        name = layer.name.lower()
        cls = layer.__class__.__name__.lower()
        if ch is None or ch <= 1:
            continue
        if "conv" in name or "conv" in cls:
            chosen = layer.name
            break

    if chosen is None:
        for layer in reversed(model.layers):
            rank, _ = _rank_and_channels(layer)
            if rank is not None and rank >= 4:
                chosen = layer.name
                break

    if chosen is None:
        raise RuntimeError("Cannot find a suitable target layer for CAM.")
    return chosen


def target_score_from_output(pred_tensor, mode="mean", pixel_xy=None, gt_weight=None):
    # pred_tensor: [1, H, W, C]
    if mode == "pixel":
        if pixel_xy is None:
            raise ValueError("pixel mode requires --pixel-x and --pixel-y")
        x, y = pixel_xy
        return pred_tensor[0, y, x, 0]
    if mode == "gt_weighted":
        if gt_weight is None:
            raise ValueError("gt_weighted mode requires ground-truth map")
        w = tf.convert_to_tensor(gt_weight[None, :, :, None], dtype=pred_tensor.dtype)
        wsum = tf.reduce_sum(w) + 1e-8
        return tf.reduce_sum(pred_tensor * w) / wsum
    return tf.reduce_mean(pred_tensor)


def compute_gradcam(model, x, layer_name, target_mode, pixel_xy=None, gt_weight=None):
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(x_tf, training=False)
        score = target_score_from_output(
            pred, mode=target_mode, pixel_xy=pixel_xy, gt_weight=gt_weight
        )
    grads = tape.gradient(score, conv_out)  # [1,h,w,c]
    weights = tf.reduce_mean(grads, axis=[1, 2], keepdims=True)
    cam = tf.reduce_sum(weights * conv_out, axis=-1)[0]  # [h,w]
    cam = tf.nn.relu(cam)
    cam = tf.image.resize(cam[:, :, None], size=x.shape[1:3], method="bilinear")
    cam = cam[:, :, 0].numpy()
    return normalize01(cam), float(score.numpy())


def compute_guided_gradient_map(model, x, target_mode, pixel_xy=None, gt_weight=None):
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        pred = model(x_tf, training=False)
        score = target_score_from_output(
            pred, mode=target_mode, pixel_xy=pixel_xy, gt_weight=gt_weight
        )
    grads = tape.gradient(score, x_tf)[0].numpy()  # [H,W,C]
    # Guided-backprop approximation: keep positive gradients and positive inputs.
    guided = np.maximum(grads, 0.0) * np.maximum(x[0], 0.0)
    gray = np.mean(np.abs(guided), axis=-1)
    return normalize01(gray)


def compute_scorecam(model, x, layer_name, target_mode, pixel_xy=None, gt_weight=None, max_maps=32):
    feat_model = keras.Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)
    feats = feat_model(x, training=False).numpy()[0]  # [h,w,c]
    h, w, c = feats.shape
    if c == 0:
        return np.zeros(x.shape[1:3], dtype=np.float32)

    energy = np.mean(np.abs(feats), axis=(0, 1))
    idx = np.argsort(energy)[::-1][: min(max_maps, c)]

    act_maps = []
    scores = []
    x0 = x[0]
    for i in idx:
        a = feats[:, :, i]
        a = normalize01(a)
        if np.max(a) <= 1e-8:
            continue
        a_up = tf.image.resize(a[:, :, None], x.shape[1:3], method="bilinear").numpy()[:, :, 0]
        a_up = normalize01(a_up)
        x_masked = x0 * a_up[:, :, None]
        pred = model(x_masked[None, ...], training=False)
        s = target_score_from_output(
            pred, mode=target_mode, pixel_xy=pixel_xy, gt_weight=gt_weight
        )
        act_maps.append(a_up)
        scores.append(float(s.numpy()))

    if not act_maps:
        return np.zeros(x.shape[1:3], dtype=np.float32)

    scores = np.asarray(scores, dtype=np.float32)
    # positive soft weighting
    scores = scores - np.max(scores)
    w_soft = np.exp(scores)
    w_soft = w_soft / (np.sum(w_soft) + 1e-8)

    cam = np.zeros_like(act_maps[0], dtype=np.float32)
    for wv, am in zip(w_soft, act_maps):
        cam += wv * am
    return normalize01(np.maximum(cam, 0.0))


def score_on_input(model, x, target_mode, pixel_xy=None, gt_weight=None):
    pred = model(x[None, ...], training=False)
    s = target_score_from_output(pred, mode=target_mode, pixel_xy=pixel_xy, gt_weight=gt_weight)
    return float(s.numpy())


def perturbation_metrics(model, x, saliency, target_mode, pixel_xy=None, gt_weight=None, topk_ratio=0.2):
    h, w = saliency.shape
    n = h * w
    k = max(1, int(n * topk_ratio))
    flat = saliency.reshape(-1)
    idx = np.argpartition(flat, -k)[-k:]
    mask = np.zeros(n, dtype=np.float32)
    mask[idx] = 1.0
    mask = mask.reshape(h, w)

    base = score_on_input(model, x, target_mode, pixel_xy=pixel_xy, gt_weight=gt_weight)
    x_remove = x * (1.0 - mask[:, :, None])
    x_keep = x * mask[:, :, None]
    score_remove = score_on_input(model, x_remove, target_mode, pixel_xy=pixel_xy, gt_weight=gt_weight)
    score_keep = score_on_input(model, x_keep, target_mode, pixel_xy=pixel_xy, gt_weight=gt_weight)

    return {
        "base_score": base,
        "score_after_remove_topk": score_remove,
        "score_after_keep_topk": score_keep,
        "drop_remove_topk": base - score_remove,
        "ratio_keep_topk": score_keep / (abs(base) + 1e-8),
    }


def overlay_heatmap(base_gray, heatmap, alpha=0.45, cmap="jet"):
    base_gray = normalize01(base_gray)
    hm = normalize01(heatmap)
    cm = plt.get_cmap(cmap)(hm)[..., :3]
    base_rgb = np.stack([base_gray] * 3, axis=-1)
    out = (1.0 - alpha) * base_rgb + alpha * cm
    return np.clip(out, 0.0, 1.0)


def save_visuals(
    out_dir, idx, x, y_true, pred, gradcam, guided_map, guided_gradcam, scorecam
):
    os.makedirs(out_dir, exist_ok=True)
    x2d = x[:, :, 0]
    pred2d = pred[:, :, 0]
    gt2d = None if y_true is None else y_true[:, :, 0]
    err2d = None if gt2d is None else np.abs(pred2d - gt2d)

    ov_gradcam = overlay_heatmap(x2d, gradcam)
    ov_guided_gradcam = overlay_heatmap(x2d, guided_gradcam)
    ov_scorecam = overlay_heatmap(x2d, scorecam)

    panel_path = os.path.join(out_dir, f"sample_{idx:04d}_explain_panel.png")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(x2d, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Input")
    axes[0, 1].imshow(pred2d, cmap="inferno", vmin=0, vmax=1)
    axes[0, 1].set_title("Prediction")
    if gt2d is not None:
        axes[0, 2].imshow(gt2d, cmap="inferno", vmin=0, vmax=1)
        axes[0, 2].set_title("Ground Truth")
    else:
        axes[0, 2].imshow(np.zeros_like(x2d), cmap="gray")
        axes[0, 2].set_title("Ground Truth (N/A)")
    if err2d is not None:
        axes[0, 3].imshow(err2d, cmap="magma")
        axes[0, 3].set_title(f"|Pred-GT| (MSE={np.mean((pred2d-gt2d)**2):.5f})")
    else:
        axes[0, 3].imshow(np.zeros_like(x2d), cmap="gray")
        axes[0, 3].set_title("|Pred-GT| (N/A)")

    axes[1, 0].imshow(ov_gradcam)
    axes[1, 0].set_title("Grad-CAM")
    axes[1, 1].imshow(guided_map, cmap="viridis")
    axes[1, 1].set_title("Guided Gradient")
    axes[1, 2].imshow(ov_guided_gradcam)
    axes[1, 2].set_title("Guided Grad-CAM")
    axes[1, 3].imshow(ov_scorecam)
    axes[1, 3].set_title("Score-CAM")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(panel_path, dpi=160)
    plt.close()

    plt.imsave(os.path.join(out_dir, f"sample_{idx:04d}_gradcam.png"), ov_gradcam)
    plt.imsave(os.path.join(out_dir, f"sample_{idx:04d}_guided_gradcam.png"), ov_guided_gradcam)
    plt.imsave(os.path.join(out_dir, f"sample_{idx:04d}_scorecam.png"), ov_scorecam)
    plt.imsave(os.path.join(out_dir, f"sample_{idx:04d}_guided_gradient.png"), guided_map, cmap="viridis")

    return panel_path


def save_batch_grid(output_dir, indices, panel_paths, cols=3):
    if not panel_paths:
        return None
    cols = max(1, int(cols))
    n = len(panel_paths)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    k = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if k < n:
                img = plt.imread(panel_paths[k])
                ax.imshow(img)
                ax.set_title(f"sample_{indices[k]:04d}", fontsize=9)
            ax.axis("off")
            k += 1

    plt.tight_layout()
    out_path = os.path.join(output_dir, "batch_explain_grid.png")
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path


def load_npz_sample(dataset_path, sample_idx):
    data = np.load(dataset_path)
    if "masks" not in data:
        raise KeyError(f"{dataset_path} must contain key 'masks'")
    masks = data["masks"]
    n = len(masks)
    if sample_idx < 0:
        sample_idx = int(np.random.randint(0, n))
    if sample_idx >= n:
        raise IndexError(f"sample_idx={sample_idx} out of range [0, {n-1}]")
    x = masks[sample_idx].astype(np.float32)
    y = data["aerials"][sample_idx].astype(np.float32) if "aerials" in data else None
    return x, y, n, sample_idx


def load_npz_arrays(dataset_path):
    data = np.load(dataset_path)
    if "masks" not in data:
        raise KeyError(f"{dataset_path} must contain key 'masks'")
    masks = data["masks"].astype(np.float32)
    aerials = data["aerials"].astype(np.float32) if "aerials" in data else None
    return masks, aerials


def _safe_corr(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run_one_sample(
    model,
    masks,
    aerials,
    idx,
    layer_name,
    target_mode,
    pixel_xy,
    scorecam_max_maps,
    output_dir,
):
    x = masks[idx]
    y_true = aerials[idx] if aerials is not None else None
    x_batch = x[None, ...]
    pred = model(x_batch, training=False).numpy()[0]

    gt_weight = None
    if target_mode == "gt_weighted":
        if y_true is None:
            raise ValueError("target-mode=gt_weighted requires dataset with 'aerials'")
        gt_weight = y_true[:, :, 0]

    gradcam, base_score = compute_gradcam(
        model, x_batch, layer_name, target_mode, pixel_xy=pixel_xy, gt_weight=gt_weight
    )
    guided_map = compute_guided_gradient_map(
        model, x_batch, target_mode, pixel_xy=pixel_xy, gt_weight=gt_weight
    )
    guided_gradcam = normalize01(guided_map * gradcam)
    scorecam = compute_scorecam(
        model,
        x_batch,
        layer_name,
        target_mode,
        pixel_xy=pixel_xy,
        gt_weight=gt_weight,
        max_maps=scorecam_max_maps,
    )

    panel_path = save_visuals(
        output_dir, idx, x, y_true, pred, gradcam, guided_map, guided_gradcam, scorecam
    )

    m_grad = perturbation_metrics(model, x, gradcam, target_mode, pixel_xy=pixel_xy, gt_weight=gt_weight)
    m_guided = perturbation_metrics(
        model, x, guided_gradcam, target_mode, pixel_xy=pixel_xy, gt_weight=gt_weight
    )
    m_score = perturbation_metrics(model, x, scorecam, target_mode, pixel_xy=pixel_xy, gt_weight=gt_weight)

    report = {
        "sample_index": int(idx),
        "target_score": float(base_score),
        "prediction_stats": {
            "pred_min": float(np.min(pred)),
            "pred_max": float(np.max(pred)),
            "pred_mean": float(np.mean(pred)),
            "pred_std": float(np.std(pred)),
        },
        "mse_to_ground_truth": None if y_true is None else float(np.mean((pred - y_true) ** 2)),
        "saliency_stats": {
            "gradcam_mean": float(np.mean(gradcam)),
            "guided_gradcam_mean": float(np.mean(guided_gradcam)),
            "scorecam_mean": float(np.mean(scorecam)),
            "gradcam_vs_scorecam_corr": _safe_corr(gradcam, scorecam),
            "gradcam_vs_guided_corr": _safe_corr(gradcam, guided_gradcam),
        },
        "perturbation_top20": {
            "gradcam": m_grad,
            "guided_gradcam": m_guided,
            "scorecam": m_score,
        },
        "outputs": {
            "panel_png": os.path.abspath(panel_path),
            "gradcam_png": os.path.abspath(os.path.join(output_dir, f"sample_{idx:04d}_gradcam.png")),
            "guided_gradcam_png": os.path.abspath(
                os.path.join(output_dir, f"sample_{idx:04d}_guided_gradcam.png")
            ),
            "scorecam_png": os.path.abspath(os.path.join(output_dir, f"sample_{idx:04d}_scorecam.png")),
            "guided_gradient_png": os.path.abspath(
                os.path.join(output_dir, f"sample_{idx:04d}_guided_gradient.png")
            ),
        },
    }
    sample_report_path = os.path.join(output_dir, f"sample_{idx:04d}_explain_report.json")
    with open(sample_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report, sample_report_path


def main():
    parser = argparse.ArgumentParser(description="CNN/U-Net visualization helper")
    parser.add_argument("--model", type=str, required=True, help="Path to .keras model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to .npz dataset with masks/aerials")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index; -1 means random")
    parser.add_argument("--target-layer", type=str, default=None, help="Layer name for CAM; auto if omitted")
    parser.add_argument(
        "--target-mode",
        type=str,
        default="mean",
        choices=["mean", "pixel", "gt_weighted"],
        help="Target score definition for regression model",
    )
    parser.add_argument("--pixel-x", type=int, default=None, help="X for target-mode=pixel")
    parser.add_argument("--pixel-y", type=int, default=None, help="Y for target-mode=pixel")
    parser.add_argument("--scorecam-max-maps", type=int, default=32, help="Max feature maps for Score-CAM")
    parser.add_argument(
        "--batch-count",
        type=int,
        default=1,
        help="Number of samples to process. >1 enables batch mode.",
    )
    parser.add_argument(
        "--batch-random",
        action="store_true",
        help="In batch mode, draw random indices. Otherwise use consecutive indices from sample-idx.",
    )
    parser.add_argument(
        "--batch-grid-png",
        action="store_true",
        help="In batch mode, create one combined PNG grid from all sample panels.",
    )
    parser.add_argument(
        "--batch-grid-cols",
        type=int,
        default=3,
        help="Column count for batch grid PNG.",
    )
    parser.add_argument("--output-dir", type=str, default="explain_outputs", help="Directory for PNG/JSON outputs")
    args = parser.parse_args()

    custom_objects = {
        "CircularPad2D": CircularPad2D,
        "CircularConv2D": CircularConv2D,
        "DilatedCircularConv2D": DilatedCircularConv2D,
    }
    model = keras.models.load_model(args.model, custom_objects=custom_objects)

    masks, aerials = load_npz_arrays(args.dataset)
    n_total = len(masks)

    pixel_xy = None
    if args.target_mode == "pixel":
        if args.pixel_x is None or args.pixel_y is None:
            raise ValueError("--target-mode pixel requires --pixel-x and --pixel-y")
        pixel_xy = (int(args.pixel_x), int(args.pixel_y))

    layer_name = choose_target_layer(model, args.target_layer)

    batch_count = max(1, int(args.batch_count))
    if args.sample_idx < 0 and not args.batch_random:
        args.batch_random = True

    if args.batch_random:
        k = min(batch_count, n_total)
        indices = np.random.choice(n_total, size=k, replace=False).astype(int).tolist()
    else:
        start = max(0, int(args.sample_idx))
        end = min(n_total, start + batch_count)
        indices = list(range(start, end))
        if not indices:
            raise ValueError("No valid sample index selected.")

    os.makedirs(args.output_dir, exist_ok=True)
    all_reports = []
    all_paths = []
    panel_paths = []
    for idx in indices:
        rep, rep_path = run_one_sample(
            model=model,
            masks=masks,
            aerials=aerials,
            idx=idx,
            layer_name=layer_name,
            target_mode=args.target_mode,
            pixel_xy=pixel_xy,
            scorecam_max_maps=args.scorecam_max_maps,
            output_dir=args.output_dir,
        )
        all_reports.append(rep)
        all_paths.append(rep_path)
        panel_paths.append(rep["outputs"]["panel_png"])

    batch_grid_path = None
    if args.batch_grid_png and len(indices) > 1:
        batch_grid_path = save_batch_grid(
            output_dir=args.output_dir,
            indices=indices,
            panel_paths=panel_paths,
            cols=args.batch_grid_cols,
        )

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": os.path.abspath(args.model),
        "dataset": os.path.abspath(args.dataset),
        "dataset_size": int(n_total),
        "target_layer": layer_name,
        "target_mode": args.target_mode,
        "batch_count": len(indices),
        "indices": indices,
        "aggregate": {
            "mean_target_score": float(np.mean([r["target_score"] for r in all_reports])),
            "mean_mse_to_gt": None
            if aerials is None
            else float(np.mean([r["mse_to_ground_truth"] for r in all_reports])),
            "mean_drop_remove_topk_gradcam": float(
                np.mean([r["perturbation_top20"]["gradcam"]["drop_remove_topk"] for r in all_reports])
            ),
            "mean_drop_remove_topk_scorecam": float(
                np.mean([r["perturbation_top20"]["scorecam"]["drop_remove_topk"] for r in all_reports])
            ),
        },
        "sample_reports": [os.path.abspath(p) for p in all_paths],
        "batch_grid_png": None if batch_grid_path is None else os.path.abspath(batch_grid_path),
    }

    summary_path = os.path.join(args.output_dir, "batch_explain_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Visualization done.")
    print(f"Target layer: {layer_name}")
    print(f"Processed samples: {indices}")
    print(f"Batch summary: {summary_path}")


if __name__ == "__main__":
    main()
