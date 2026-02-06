import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Compatibility shim: keras.ops (Keras 3 / TF>=2.16) vs tf ops (Keras 2 / TF<2.16)
try:
    import keras.ops as ops
    _concatenate = ops.concatenate
except ImportError:
    _concatenate = tf.concat


class CircularPad2D(layers.Layer):
    """Circular (wrap-around) padding layer for 2D spatial data."""
    def __init__(self, padding, **kwargs):
        super().__init__(**kwargs)
        self.pad = padding

    def call(self, x):
        if self.pad <= 0:
            return x
        p = self.pad
        # Pad height (top/bottom)
        x = _concatenate([x[:, -p:, :, :], x, x[:, :p, :, :]], axis=1)
        # Pad width (left/right)
        x = _concatenate([x[:, :, -p:, :], x, x[:, :, :p, :]], axis=2)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'padding': self.pad})
        return config


class CircularConv2D(layers.Layer):
    """Conv2D layer with circular padding for shift equivariance"""
    def __init__(self, filters, kernel_size, strides=1, activation=None, use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides
        self._activation_name = activation
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias

        # Calculate padding needed
        pad_h = (self.kernel_size[0] - 1) // 2
        pad_w = (self.kernel_size[1] - 1) // 2
        self._pad_amount = max(pad_h, pad_w)

        self.pad_layer = CircularPad2D(self._pad_amount)
        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            activation=None,
            use_bias=use_bias
        )

    def call(self, inputs):
        padded = self.pad_layer(inputs)
        output = self.conv(padded)
        if self.activation:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'activation': self._activation_name,
            'use_bias': self.use_bias,
        })
        return config


class DilatedCircularConv2D(layers.Layer):
    """Dilated Conv2D with circular padding, GroupNorm, and ReLU."""
    def __init__(self, filters, kernel_size=3, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation_rate - 1)
        pad = (effective_kernel_size - 1) // 2

        self.pad_layer = CircularPad2D(pad)
        self.conv = layers.Conv2D(
            filters, kernel_size, padding='valid',
            dilation_rate=dilation_rate, use_bias=False
        )
        self.norm = layers.GroupNormalization(groups=min(filters, 32))
        self.relu = layers.Activation('relu')

    def call(self, x):
        x = self.pad_layer(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
        })
        return config


def multi_scale_feature_block(x, filters):
    """Multi-scale feature extraction without spatial pooling"""
    conv1 = DilatedCircularConv2D(filters // 4, dilation_rate=1)(x)
    conv2 = DilatedCircularConv2D(filters // 4, dilation_rate=2)(x)
    conv4 = DilatedCircularConv2D(filters // 4, dilation_rate=4)(x)
    conv8 = DilatedCircularConv2D(filters // 4, dilation_rate=8)(x)

    multi_scale = layers.Concatenate()([conv1, conv2, conv4, conv8])
    output = CircularConv2D(filters, 1, activation='relu')(multi_scale)
    return output


def shift_equivariant_unet(input_shape=(512, 512, 3), num_classes=1):
    """
    Shift-equivariant U-Net using dilated convolutions instead of pooling.

    All operations preserve spatial dimensions and use circular (wrap-around)
    padding, ensuring that f(shift(x)) == shift(f(x)) for circular shifts.
    """
    inputs = keras.Input(shape=input_shape)

    # Encoder path with increasing dilation rates instead of downsampling
    # Stage 1
    conv1_1 = CircularConv2D(64, 3, activation='relu')(inputs)
    conv1_2 = CircularConv2D(64, 3, activation='relu')(conv1_1)
    feat1 = multi_scale_feature_block(conv1_2, 64)

    # Stage 2 - increase receptive field with dilation
    conv2_1 = DilatedCircularConv2D(128, dilation_rate=2)(feat1)
    conv2_2 = DilatedCircularConv2D(128, dilation_rate=2)(conv2_1)
    feat2 = multi_scale_feature_block(conv2_2, 128)

    # Stage 3 - further increase receptive field
    conv3_1 = DilatedCircularConv2D(256, dilation_rate=4)(feat2)
    conv3_2 = DilatedCircularConv2D(256, dilation_rate=4)(conv3_1)
    feat3 = multi_scale_feature_block(conv3_2, 256)

    # Stage 4 - largest receptive field
    conv4_1 = DilatedCircularConv2D(512, dilation_rate=8)(feat3)
    conv4_2 = DilatedCircularConv2D(512, dilation_rate=8)(conv4_1)
    feat4 = multi_scale_feature_block(conv4_2, 512)

    # Decoder path with skip connections
    # Decode 4->3
    up3 = layers.Concatenate()([feat4, feat3])
    conv_up3_1 = DilatedCircularConv2D(256, dilation_rate=4)(up3)
    conv_up3_2 = DilatedCircularConv2D(256, dilation_rate=4)(conv_up3_1)

    # Decode 3->2
    up2 = layers.Concatenate()([conv_up3_2, feat2])
    conv_up2_1 = DilatedCircularConv2D(128, dilation_rate=2)(up2)
    conv_up2_2 = DilatedCircularConv2D(128, dilation_rate=2)(conv_up2_1)

    # Decode 2->1
    up1 = layers.Concatenate()([conv_up2_2, feat1])
    conv_up1_1 = CircularConv2D(64, 3, activation='relu')(up1)
    conv_up1_2 = CircularConv2D(64, 3, activation='relu')(conv_up1_1)

    # Final output layer
    if num_classes == 1:
        outputs = CircularConv2D(1, 1, activation='sigmoid')(conv_up1_2)
    else:
        outputs = CircularConv2D(num_classes, 1, activation='softmax')(conv_up1_2)

    model = keras.Model(inputs=inputs, outputs=outputs, name='shift_equivariant_unet')
    return model


def shift_input_tensor(tensor, shift_h, shift_w):
    """
    Shift input tensor by given amounts using circular shifts.
    Args:
        tensor: Input tensor of shape (batch, height, width, channels)
        shift_h: Shift amount in height dimension (can be negative)
        shift_w: Shift amount in width dimension (can be negative)
    """
    shifted = tf.roll(tensor, shift=shift_h, axis=1)
    shifted = tf.roll(shifted, shift=shift_w, axis=2)
    return shifted


def test_shift_equivariance(model, input_shape=(1, 128, 128, 3), num_tests=5, max_shift=32):
    """
    Test shift equivariance of the model.
    Args:
        model: The model to test
        input_shape: Shape of input tensor
        num_tests: Number of random shift tests to perform
        max_shift: Maximum shift amount in pixels
    Returns:
        dict: Test results with metrics
    """
    print("Testing Shift Equivariance...")
    print("=" * 50)

    test_input = tf.random.normal(input_shape)
    original_output = model(test_input, training=False)

    results = []

    for i in range(num_tests):
        shift_h = np.random.randint(-max_shift, max_shift + 1)
        shift_w = np.random.randint(-max_shift, max_shift + 1)

        print(f"Test {i+1}: Shift = ({shift_h}, {shift_w})")

        shifted_input = shift_input_tensor(test_input, shift_h, shift_w)
        shifted_output = model(shifted_input, training=False)
        expected_output = shift_input_tensor(original_output, shift_h, shift_w)

        mae = tf.reduce_mean(tf.abs(shifted_output - expected_output))
        mse = tf.reduce_mean(tf.square(shifted_output - expected_output))
        max_diff = tf.reduce_max(tf.abs(shifted_output - expected_output))

        output_magnitude = tf.reduce_mean(tf.abs(original_output))
        relative_error = mae / (output_magnitude + 1e-8)

        result = {
            'shift': (shift_h, shift_w),
            'mae': float(mae),
            'mse': float(mse),
            'max_diff': float(max_diff),
            'relative_error': float(relative_error)
        }
        results.append(result)

        print(f"  MAE: {mae:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  Max Diff: {max_diff:.6f}")
        print(f"  Relative Error: {relative_error:.6f}")
        print()

    avg_mae = np.mean([r['mae'] for r in results])
    avg_mse = np.mean([r['mse'] for r in results])
    avg_relative_error = np.mean([r['relative_error'] for r in results])
    max_mae = np.max([r['mae'] for r in results])

    print("Summary:")
    print(f"Average MAE: {avg_mae:.6f}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average Relative Error: {avg_relative_error:.6f}")
    print(f"Maximum MAE: {max_mae:.6f}")

    # Threshold accounts for float32 numerical accumulation through many layers
    threshold = 1e-3
    is_equivariant = avg_mae < threshold

    print(f"\nShift Equivariance Test: {'PASS' if is_equivariant else 'FAIL'}")
    print(f"(Threshold: {threshold})")

    return {
        'results': results,
        'summary': {
            'avg_mae': avg_mae,
            'avg_mse': avg_mse,
            'avg_relative_error': avg_relative_error,
            'max_mae': max_mae,
            'is_equivariant': is_equivariant,
            'threshold': threshold
        }
    }


def visualize_shift_test(model, shift_h=32, shift_w=32, input_shape=(1, 128, 128, 3)):
    """Visualize shift equivariance test with sample images"""
    test_input = tf.random.normal(input_shape)
    original_output = model(test_input, training=False)

    shifted_input = shift_input_tensor(test_input, shift_h, shift_w)
    shifted_output = model(shifted_input, training=False)
    expected_output = shift_input_tensor(original_output, shift_h, shift_w)

    difference = tf.abs(shifted_output - expected_output)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    test_input_vis = test_input[0, :, :, 0].numpy()
    shifted_input_vis = shifted_input[0, :, :, 0].numpy()
    original_output_vis = original_output[0, :, :, 0].numpy()
    shifted_output_vis = shifted_output[0, :, :, 0].numpy()
    difference_vis = difference[0, :, :, 0].numpy()

    axes[0, 0].imshow(test_input_vis, cmap='gray')
    axes[0, 0].set_title('Original Input')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(shifted_input_vis, cmap='gray')
    axes[0, 1].set_title(f'Shifted Input ({shift_h}, {shift_w})')
    axes[0, 1].axis('off')

    axes[0, 2].axis('off')

    axes[1, 0].imshow(original_output_vis, cmap='viridis')
    axes[1, 0].set_title('Original Output')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(shifted_output_vis, cmap='viridis')
    axes[1, 1].set_title('Actual Shifted Output')
    axes[1, 1].axis('off')

    im = axes[1, 2].imshow(difference_vis, cmap='hot')
    axes[1, 2].set_title('Difference (Error)')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig('shift_equivariance_test.png', dpi=100)
    plt.close()

    mae = tf.reduce_mean(difference)
    max_diff = tf.reduce_max(difference)
    print(f"MAE: {mae:.6f}")
    print(f"Max Difference: {max_diff:.6f}")


if __name__ == "__main__":
    print("Creating shift-equivariant U-Net...")
    model = shift_equivariant_unet(input_shape=(128, 128, 3), num_classes=1)

    print(f"Model created with {model.count_params():,} parameters")
    print("\nModel Summary:")
    model.summary()

    print("\n" + "=" * 60)
    test_results = test_shift_equivariance(model, num_tests=3, max_shift=16)

    print("\nGenerating visualization...")
    visualize_shift_test(model, shift_h=8, shift_w=12, input_shape=(1, 128, 128, 3))
