"""Shift-equivariant U-Net with TokenizedMLP bottleneck.

Replaces the dilated conv bottleneck with axis-decomposed global mixing
blocks adapted from UNeXt-ILT (JMM 2025). Uses large-kernel depthwise
circular convolutions along each axis for shift-equivariant global context.
"""

from tensorflow import keras

from shift_equivariant_unet import (
    CircularConv2D,
    DilatedCircularConv2D,
    TokenizedMLPBlock,
)


def build_model(input_shape=(64, 64, 1), num_filters_base=32,
                num_mlp_blocks=2, axis_kernel=31):
    """Build shift-equivariant U-Net with TokenizedMLP bottleneck.

    Args:
        input_shape: Spatial input shape (H, W, C).
        num_filters_base: Base filter count; stages use 1x, 2x, 4x, 8x.
        num_mlp_blocks: Number of TokenizedMLP blocks in the bottleneck.
        axis_kernel: Kernel size for the axis-decomposed depthwise convs.
            31 covers ~48% of a 64-wide grid per axis.

    Returns:
        keras.Model with ~1.4M parameters (at base=32, 2 blocks, kernel=31).
    """
    inputs = keras.Input(shape=input_shape)

    # Encoder (same as baseline)
    x = CircularConv2D(num_filters_base, 3, activation='relu')(inputs)
    x = CircularConv2D(num_filters_base, 3, activation='relu')(x)
    feat1 = x

    x = DilatedCircularConv2D(num_filters_base * 2, dilation_rate=2)(feat1)
    x = DilatedCircularConv2D(num_filters_base * 2, dilation_rate=2)(x)
    feat2 = x

    x = DilatedCircularConv2D(num_filters_base * 4, dilation_rate=4)(feat2)
    x = DilatedCircularConv2D(num_filters_base * 4, dilation_rate=4)(x)
    feat3 = x

    # Bottleneck: project to bottleneck dim, then TokenizedMLP blocks
    bottleneck_dim = num_filters_base * 8
    x = CircularConv2D(bottleneck_dim, 1, activation='relu')(feat3)
    for _ in range(num_mlp_blocks):
        x = TokenizedMLPBlock(
            dim=bottleneck_dim,
            axis_kernel=axis_kernel,
        )(x)

    # Decoder (same as baseline)
    x = keras.layers.Concatenate()([x, feat3])
    x = DilatedCircularConv2D(num_filters_base * 4, dilation_rate=4)(x)
    x = DilatedCircularConv2D(num_filters_base * 4, dilation_rate=4)(x)

    x = keras.layers.Concatenate()([x, feat2])
    x = DilatedCircularConv2D(num_filters_base * 2, dilation_rate=2)(x)
    x = DilatedCircularConv2D(num_filters_base * 2, dilation_rate=2)(x)

    x = keras.layers.Concatenate()([x, feat1])
    x = CircularConv2D(num_filters_base, 3, activation='relu')(x)
    x = CircularConv2D(num_filters_base, 3, activation='relu')(x)

    # Output
    outputs = CircularConv2D(1, 1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs,
                        name='litho_unet_tokenized_mlp')
    return model
