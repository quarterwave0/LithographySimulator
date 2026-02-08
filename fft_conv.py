"""FFT-based circular convolution for accelerating large-kernel depthwise conv.

Provides a drop-in replacement for spatial depthwise convolution when the
kernel size is large.  For small kernels the overhead of FFT/IFFT dominates,
so callers should gate on kernel size (recommended threshold: >= 11).

Implementation notes – TensorFlow FFT conventions
--------------------------------------------------
* ``tf.signal.rfft`` computes the 1-D DFT of a real signal along the
  **innermost** (last) axis and returns the positive-frequency half
  (length ``N // 2 + 1``) as complex64/128.
* ``tf.signal.irfft`` inverts it back to a real signal.
* Unlike PyTorch's ``torch.fft.rfft`` which defaults to the last dim and
  returns the same number of frequency bins, TF requires an explicit
  ``fft_length`` argument when the desired transform length differs from
  the tensor's innermost dimension.

Frequency-domain cross-correlation
-----------------------------------
TensorFlow's ``Conv2D`` / ``DepthwiseConv2D`` compute **cross-correlation**
(not convolution):

    out[n] = Σ_k  kernel[k] · x[(n + k − half) mod N]

where ``half = K // 2`` and ``N`` is the signal length.

In frequency domain this equals:

    Out = IFFT( FFT(x) · conj(FFT(h_circ)) )

where ``h_circ`` is the kernel placed into a length-N circular buffer with
the kernel centre at index 0:

    h_circ[0 .. K-1-half]         = kernel[half .. K-1]     (causal part)
    h_circ[N-half .. N-1]         = kernel[0 .. half-1]     (anti-causal wrap)
    all other entries              = 0
"""

import tensorflow as tf


def _build_circ_kernel_1d(kernel_1d, signal_length):
    """Place a 1-D kernel into a circular buffer of length *signal_length*.

    Args:
        kernel_1d: Tensor of shape ``(K,)`` – one channel's 1-D kernel.
        signal_length: Python int or scalar tf.Tensor – the FFT length ``N``.

    Returns:
        Tensor of shape ``(signal_length,)`` with the kernel centred at index 0
        (ready for ``tf.signal.rfft``).
    """
    K = tf.shape(kernel_1d)[0]
    half = K // 2

    # Causal part: kernel[half:] → positions [0, K-1-half]
    causal = kernel_1d[half:]          # length K - half
    # Anti-causal part: kernel[:half] → positions [N-half, N-1]
    anti_causal = kernel_1d[:half]     # length half

    # Zero-pad in between
    n_zeros = signal_length - K
    zeros = tf.zeros([n_zeros], dtype=kernel_1d.dtype)

    # h_circ = [causal | zeros | anti_causal]
    h_circ = tf.concat([causal, zeros, anti_causal], axis=0)
    return h_circ


def fft_circular_depthwise_conv1d(x, kernel_1d_per_channel, axis):
    """FFT-accelerated circular depthwise 1-D cross-correlation.

    Computes, for each spatial position ``n`` and channel ``c``:

        out[n, c] = Σ_k  kernel[k, c] · x[(n + k − K//2) mod N, c]

    This matches the result of ``CircularPad1D`` + ``DepthwiseConv2D`` with
    ``padding='valid'`` for a ``(1, K)`` or ``(K, 1)`` kernel.

    Args:
        x: Input tensor of shape ``(B, H, W, C)``  (channels-last).
        kernel_1d_per_channel: Kernel tensor of shape ``(K, C)`` where ``K``
            is the kernel size and ``C`` matches the channel dim of *x*.
        axis: ``1`` for height-axis convolution, ``2`` for width-axis.

    Returns:
        Tensor of same shape as *x*.
    """
    # Move the target spatial axis to the innermost position for rfft.
    if axis == 2:  # width
        # (B, H, W, C) → (B, H, C, W)
        x_t = tf.transpose(x, [0, 1, 3, 2])
    else:  # height
        # (B, H, W, C) → (B, W, C, H)
        x_t = tf.transpose(x, [0, 2, 3, 1])

    # Use static shape for N (spatial dims are always known at build time).
    # This is required for Keras 3 symbolic tracing compatibility since
    # tf.signal.rfft needs a concrete fft_length for shape inference.
    N_static = x_t.shape[-1]
    if N_static is not None:
        N = int(N_static)
    else:
        N = tf.shape(x_t)[-1]

    # Kernel shape: use static K since it is always known.
    K = int(kernel_1d_per_channel.shape[0])
    C_k = kernel_1d_per_channel.shape[1]
    half = K // 2

    # --- Signal FFT ---
    X = tf.signal.rfft(x_t, fft_length=[N])  # (..., N//2+1) complex

    # --- Kernel FFT (per channel) ---
    # Build circular kernels for all channels: (C, N)
    # Transpose kernel to (C, K) for vectorised construction.
    kern_t = tf.transpose(kernel_1d_per_channel, [1, 0])  # (C, K)

    causal = kern_t[:, half:]          # (C, K - half)
    anti_causal = kern_t[:, :half]     # (C, half)
    n_zeros = N - K
    zeros = tf.zeros([C_k, n_zeros] if C_k is not None else
                     [tf.shape(kern_t)[0], n_zeros], dtype=kern_t.dtype)
    h_circ = tf.concat([causal, zeros, anti_causal], axis=1)  # (C, N)

    H = tf.signal.rfft(h_circ, fft_length=[N])  # (C, N//2+1)

    # --- Frequency-domain cross-correlation ---
    # X shape: (..., C, N//2+1);  H shape: (C, N//2+1)
    # Broadcast H over batch dims.
    Y = X * tf.math.conj(H)

    # --- Inverse FFT ---
    y_t = tf.signal.irfft(Y, fft_length=[N])  # (..., N)

    # Transpose back to (B, H, W, C)
    if axis == 2:
        y = tf.transpose(y_t, [0, 1, 3, 2])
    else:
        y = tf.transpose(y_t, [0, 3, 1, 2])

    return y
