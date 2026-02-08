"""Baseline shift-equivariant U-Net for lithography aerial image prediction.

Uses dilated circular convolutions throughout (no pooling) to preserve
shift equivariance. All spatial dimensions are maintained at input resolution.
"""

from tensorflow import keras

from shift_equivariant_unet import (
    CircularConv2D,
    DilatedCircularConv2D,
)


def build_model(input_shape=(64, 64, 1), num_filters_base=32):
    """Build shift-equivariant U-Net with dilated conv bottleneck.

    Args:
        input_shape: Spatial input shape (H, W, C).
        num_filters_base: Base filter count; stages use 1x, 2x, 4x, 8x.

    Returns:
        keras.Model with ~1.9M parameters (at base=32).
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
                        name='litho_unet_baseline')
    return model
