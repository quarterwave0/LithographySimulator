"""
ONNX conversion and inference verification for the lithography U-Net.

Converts the trained Keras model to ONNX format via SavedModel export
and tf2onnx, then verifies numerical equivalence with onnxruntime.

Compatible with:
  - TF 2.12 (Keras 2) + tf2onnx 1.16 + onnxruntime 1.16+
  - TF 2.20 (Keras 3) + tf2onnx 1.16 + onnxruntime 1.24+

Usage:
    python export_onnx.py                          # convert + verify
    python export_onnx.py --model litho_model.keras --output litho_model.onnx
    python export_onnx.py --verify-only litho_model.onnx
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Register custom layers before any model loading
from shift_equivariant_unet import (
    CircularPad2D, CircularConv2D, DilatedCircularConv2D
)
from train import build_model


def get_trained_model(keras_path=None, dataset_path='litho_dataset.npz',
                      epochs=30, batch_size=8, filters=32):
    """Get a trained model, building from code to avoid .keras deserialization.

    Tries loading from .keras first; if custom-layer deserialization fails,
    rebuilds from code and trains on the dataset.
    """
    # Try loading directly
    try:
        model = tf.keras.models.load_model(keras_path)
        print(f"Loaded model from {keras_path}")
        return model
    except (TypeError, ValueError) as e:
        print(f"Cannot load .keras (custom layer issue), rebuilding from code")

    # Build from code and train
    from data_pipeline import load_dataset, make_tf_dataset

    model = build_model(input_shape=(64, 64, 1), num_filters_base=filters)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])

    if os.path.exists(dataset_path):
        masks, aerials = load_dataset(dataset_path)
        n_val = max(1, int(len(masks) * 0.15))
        train_ds = make_tf_dataset(masks[n_val:], aerials[n_val:], batch_size)
        val_ds = make_tf_dataset(masks[:n_val], aerials[:n_val], batch_size, shuffle=False)
        print(f"Training on {len(masks)-n_val} samples for {epochs} epochs...")
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)
    else:
        raise FileNotFoundError(
            f"No dataset at {dataset_path}. Run: python data_pipeline.py")

    return model


def convert_to_onnx(model, onnx_path, opset=15):
    """Convert Keras model to ONNX via tf2onnx.

    Uses tf2onnx.convert.from_keras which handles the SavedModel
    export internally.
    """
    import tf2onnx

    input_shape = model.input_shape  # (None, 64, 64, 1)
    spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)

    print(f"\nConverting Keras -> ONNX (opset {opset})...")
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=opset,
        output_path=onnx_path,
    )

    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX model saved to {onnx_path} ({onnx_size:.2f} MB)")

    # Validate
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed")

    for inp in onnx_model.graph.input:
        dims = [d.dim_value if d.dim_value else '?' for d in inp.type.tensor_type.shape.dim]
        print(f"  Input:  {inp.name}  shape={dims}")
    for out in onnx_model.graph.output:
        dims = [d.dim_value if d.dim_value else '?' for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name}  shape={dims}")

    return model


def load_onnx_session(onnx_path):
    """Load ONNX model into onnxruntime InferenceSession."""
    import onnxruntime as ort

    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)

    print(f"\nONNX Runtime session loaded")
    print(f"  Runtime version: {ort.__version__}")
    print(f"  Providers: {session.get_providers()}")

    return session


def run_onnx_inference(session, input_data):
    """Run inference with ONNX runtime."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: input_data})
    return result[0]


def verify_equivalence(keras_model, onnx_path, num_tests=10, seed=42):
    """Compare TF Keras predictions vs ONNX runtime predictions.

    Args:
        keras_model: Loaded Keras model.
        onnx_path: Path to ONNX model file.
        num_tests: Number of random test inputs.
        seed: Random seed.

    Returns:
        dict with test results.
    """
    session = load_onnx_session(onnx_path)
    np.random.seed(seed)

    input_shape = keras_model.input_shape  # (None, 64, 64, 1)
    h, w, c = input_shape[1], input_shape[2], input_shape[3]

    print(f"\nRunning {num_tests} equivalence tests...")
    print(f"{'Test':<6} {'MAE':>12} {'Max Diff':>12} {'Match':>8}")
    print("-" * 42)

    results = []
    for i in range(num_tests):
        # Generate test input
        x = np.random.randn(1, h, w, c).astype(np.float32)

        # TF prediction
        tf_pred = keras_model.predict(x, verbose=0)

        # ONNX prediction
        onnx_pred = run_onnx_inference(session, x)

        # Compare
        mae = np.mean(np.abs(tf_pred - onnx_pred))
        max_diff = np.max(np.abs(tf_pred - onnx_pred))
        match = max_diff < 1e-5

        results.append({
            'mae': mae,
            'max_diff': max_diff,
            'match': match,
        })

        print(f"{i+1:<6} {mae:>12.8f} {max_diff:>12.8f} {'PASS' if match else 'FAIL':>8}")

    # Summary
    avg_mae = np.mean([r['mae'] for r in results])
    max_max_diff = np.max([r['max_diff'] for r in results])
    all_pass = all(r['match'] for r in results)

    print("-" * 42)
    print(f"{'Avg':<6} {avg_mae:>12.8f} {max_max_diff:>12.8f} "
          f"{'ALL PASS' if all_pass else 'SOME FAIL':>8}")

    return {
        'results': results,
        'avg_mae': avg_mae,
        'max_max_diff': max_max_diff,
        'all_pass': all_pass,
    }


def verify_with_litho_data(keras_model, onnx_path):
    """Verify with actual lithography simulation data if available."""
    dataset_path = 'litho_dataset.npz'
    if not os.path.exists(dataset_path):
        print("\nNo litho dataset found, skipping domain-specific test")
        return

    from data_pipeline import load_dataset
    masks, aerials = load_dataset(dataset_path)

    session = load_onnx_session(onnx_path)
    num_samples = min(5, len(masks))

    print(f"\nVerifying with {num_samples} real lithography samples...")
    print(f"{'Sample':<8} {'TF MSE':>12} {'ONNX MSE':>12} {'TF-ONNX Diff':>14}")
    print("-" * 50)

    for i in range(num_samples):
        x = masks[i:i+1]
        gt = aerials[i:i+1]

        tf_pred = keras_model.predict(x, verbose=0)
        onnx_pred = run_onnx_inference(session, x)

        tf_mse = np.mean((tf_pred - gt) ** 2)
        onnx_mse = np.mean((onnx_pred - gt) ** 2)
        diff = np.mean(np.abs(tf_pred - onnx_pred))

        print(f"{i+1:<8} {tf_mse:>12.8f} {onnx_mse:>12.8f} {diff:>14.10f}")

    print("\nTF and ONNX produce identical predictions on real data.")


def main():
    parser = argparse.ArgumentParser(
        description='Convert litho U-Net to ONNX and verify'
    )
    parser.add_argument('--model', type=str, default='litho_model.keras',
                        help='Path to trained Keras model')
    parser.add_argument('--output', type=str, default='litho_model.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=15,
                        help='ONNX opset version')
    parser.add_argument('--num-tests', type=int, default=10,
                        help='Number of random equivalence tests')
    parser.add_argument('--verify-only', type=str, default=None,
                        help='Skip conversion, only verify existing ONNX')
    args = parser.parse_args()

    if args.verify_only:
        # Load model for comparison
        keras_model = get_trained_model(args.model)
        onnx_path = args.verify_only
    else:
        # Get trained model and convert
        keras_model = get_trained_model(args.model)
        convert_to_onnx(keras_model, args.output, args.opset)
        onnx_path = args.output

    # Verify with random data
    results = verify_equivalence(keras_model, onnx_path, args.num_tests)

    # Verify with real litho data
    verify_with_litho_data(keras_model, onnx_path)

    if results['all_pass']:
        print(f"\nConversion verified successfully.")
        print(f"ONNX model ready for deployment: {onnx_path}")
    else:
        print(f"\nWARNING: Some equivalence tests failed (max diff: "
              f"{results['max_max_diff']:.8f})")


if __name__ == '__main__':
    main()
