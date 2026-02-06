# Shift-Equivariant U-Net

A U-Net variant for lithography image segmentation that guarantees **shift equivariance**: shifting the input by any amount produces an identically shifted output, with no artifacts or boundary effects.

```
f(shift(x)) == shift(f(x))
```

## Motivation

Standard U-Nets break shift equivariance through two mechanisms:

1. **Max/average pooling** in the encoder creates aliasing when features straddle pool boundaries. A 1-pixel input shift can change which features survive downsampling.
2. **Zero padding** in convolutions treats image borders as special, so features near edges behave differently from features in the interior.

For lithography simulation, where mask patterns tile periodically and sub-pixel alignment matters, these artifacts are unacceptable. This implementation eliminates both sources of shift variance.

## Architecture

### Design Principle

Replace all spatially destructive operations (pooling, upsampling, zero-padded convolutions) with operations that preserve spatial dimensions and treat boundaries as periodic.

### Layer Components

| Layer | Purpose | File Reference |
|---|---|---|
| `CircularPad2D` | Wrap-around padding that copies opposite edges | `shift_equivariant_unet.py:15` |
| `CircularConv2D` | Conv2D with circular padding (replaces zero-padded Conv2D) | `shift_equivariant_unet.py:37` |
| `DilatedCircularConv2D` | Dilated conv + circular padding + GroupNorm + ReLU | `shift_equivariant_unet.py:82` |

### Circular Padding

Instead of zero-padding (which breaks translational symmetry at boundaries), each convolution wraps the input toroidally:

```
Original:    [A B C D E]
Zero-pad:    [0 A B C D E 0]    <-- edges see zeros, breaking equivariance
Circular:    [E A B C D E A]    <-- edges see wrapped neighbors, preserving it
```

For a kernel of size `k` with dilation rate `d`, the effective kernel size is `k + (k-1)(d-1)`, and padding is `(effective_size - 1) // 2` on each side.

### Dilated Convolutions Replace Pooling

The standard U-Net encoder uses pooling to grow the receptive field. This architecture instead uses **increasing dilation rates** across encoder stages:

| Stage | Dilation Rate | Effective 3x3 Receptive Field | Equivalent To |
|---|---|---|---|
| 1 | 1 | 3x3 | Standard conv |
| 2 | 2 | 5x5 | After 1 pool layer |
| 3 | 4 | 9x9 | After 2 pool layers |
| 4 | 8 | 17x17 | After 3 pool layers |

All stages operate at the **original spatial resolution**. No information is lost to downsampling.

### Multi-Scale Feature Blocks

Each encoder/decoder stage ends with a multi-scale feature block that applies four parallel dilated convolutions (rates 1, 2, 4, 8) and concatenates the results. This is similar to an ASPP (Atrous Spatial Pyramid Pooling) module, giving each stage access to multiple receptive field scales simultaneously.

```
Input --> DilatedConv(r=1) --\
      --> DilatedConv(r=2) ---+--> Concat --> 1x1 Conv --> Output
      --> DilatedConv(r=4) --/
      --> DilatedConv(r=8) -/
```

### Network Topology

```
Input (H x W x C)
  |
  v
[Stage 1] CircConv(64) x2 --> MultiScale(64) -------- feat1
  |                                                      |
  v                                                      |
[Stage 2] DilConv(128,d=2) x2 --> MultiScale(128) --- feat2
  |                                                      |  |
  v                                                      |  |
[Stage 3] DilConv(256,d=4) x2 --> MultiScale(256) --- feat3
  |                                                      |  |  |
  v                                                      |  |  |
[Stage 4] DilConv(512,d=8) x2 --> MultiScale(512) --- feat4
  |                                                   |  |  |
  v  [Decode 4->3] Concat(feat4, feat3) --> DilConv(256,d=4) x2
     |                                            |  |
     v  [Decode 3->2] Concat(prev, feat2) --> DilConv(128,d=2) x2
        |                                           |
        v  [Decode 2->1] Concat(prev, feat1) --> CircConv(64) x2
           |
           v
        1x1 Conv --> sigmoid/softmax --> Output (H x W x num_classes)
```

All spatial dimensions are preserved throughout: input and output share the same `H x W`.

### Normalization

GroupNormalization (32 groups or fewer) is used instead of BatchNormalization. GroupNorm is independent of batch size and doesn't introduce batch-dependent statistics that could affect equivariance during inference.

## Usage

### Building a Model

```python
from shift_equivariant_unet import shift_equivariant_unet

# Binary segmentation (default)
model = shift_equivariant_unet(input_shape=(256, 256, 3), num_classes=1)

# Multi-class segmentation
model = shift_equivariant_unet(input_shape=(256, 256, 1), num_classes=5)
```

### Testing Shift Equivariance

```python
from shift_equivariant_unet import test_shift_equivariance

results = test_shift_equivariance(
    model,
    input_shape=(1, 128, 128, 3),
    num_tests=10,
    max_shift=32
)

print(results['summary']['avg_mae'])        # Should be ~1e-6
print(results['summary']['is_equivariant']) # Should be True
```

### Visualizing the Equivariance Test

```python
from shift_equivariant_unet import visualize_shift_test

# Saves shift_equivariance_test.png showing:
# - Original vs shifted input
# - Original vs shifted output
# - Pixel-wise error map
visualize_shift_test(model, shift_h=16, shift_w=24)
```

### Running the Demo

```bash
python shift_equivariant_unet.py
```

This builds a 128x128 model (~11M parameters), prints the model summary, runs 3 shift equivariance tests, and saves a visualization.

## API Reference

### `shift_equivariant_unet(input_shape, num_classes)`

Build and return a Keras functional model.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_shape` | tuple | `(512, 512, 3)` | Spatial dimensions and channels of input |
| `num_classes` | int | `1` | Number of output classes. 1 = binary (sigmoid), >1 = multi-class (softmax) |

**Returns:** `keras.Model` with input shape `(batch, H, W, C)` and output shape `(batch, H, W, num_classes)`.

### `test_shift_equivariance(model, input_shape, num_tests, max_shift)`

Run randomized shift equivariance tests on a model.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `keras.Model` | required | Model to test |
| `input_shape` | tuple | `(1, 128, 128, 3)` | Shape including batch dimension |
| `num_tests` | int | `5` | Number of random shift tests |
| `max_shift` | int | `32` | Maximum shift in pixels (each direction) |

**Returns:** dict with `results` (per-test metrics) and `summary` (aggregated metrics including `is_equivariant` bool).

### `visualize_shift_test(model, shift_h, shift_w, input_shape)`

Generate and save a visualization comparing shifted predictions.

### `CircularConv2D(filters, kernel_size, strides, activation, use_bias)`

Drop-in replacement for `keras.layers.Conv2D` with circular padding. Supports all standard activations.

### `DilatedCircularConv2D(filters, kernel_size, dilation_rate)`

Composite layer: CircularPad2D + Conv2D(dilated) + GroupNorm + ReLU.

## Compatibility

Tested with:

| TensorFlow | Keras | Status |
|---|---|---|
| 2.12.x | 2.12 (Keras 2) | Working |
| 2.20.x | 3.13 (Keras 3) | Working |

A compatibility shim handles the API difference between Keras 2 (`tf.concat` on KerasTensors) and Keras 3 (`keras.ops.concatenate` required):

```python
try:
    import keras.ops as ops
    _concatenate = ops.concatenate
except ImportError:
    _concatenate = tf.concat
```

## Performance Considerations

Because all operations run at full spatial resolution (no pooling), memory usage scales as `O(H * W * max_filters)`. For a 512x512 input at stage 4 (512 filters), a single feature map is 512 * 512 * 512 * 4 bytes = 512 MB in float32.

Practical guidelines:
- **64x64 to 128x128**: Comfortable on most GPUs (< 4 GB)
- **256x256**: Requires a GPU with 8+ GB VRAM
- **512x512**: Requires 16+ GB VRAM or gradient checkpointing

For training on large images, consider reducing the filter counts or the number of stages.

## Theoretical Guarantee

Shift equivariance holds **exactly** (up to floating-point precision) because:

1. **Convolution** is inherently translation-equivariant.
2. **Circular padding** makes the boundary condition periodic, so there is no spatial position where the network behaves differently.
3. **No pooling or strided operations** means no aliasing or spatial quantization.
4. **GroupNormalization** normalizes per-sample, per-group, independent of spatial position.
5. **Element-wise activations** (ReLU, sigmoid, softmax-per-pixel) commute with translation.

The measured equivariance error is ~1e-6 MAE, attributable entirely to floating-point non-associativity in the accumulation order of convolution operations.
