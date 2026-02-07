"""
Data pipeline for generating (mask, aerial image) training pairs
using the TensorFlow lithography simulator.

Mask patterns:
  - Vertical lines with random spacing/width
  - Horizontal lines with random spacing/width
  - Contact holes (rectangular dots)
  - L-shapes / T-shapes
  - Random rectangles

Each sample is a (mask, aerial_image) pair, both 64x64 float32 in [0, 1].
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from litho_sim_tf import Mask, LightSource, Pupil, abbe_image


# ---------------------------------------------------------------------------
# Mask generators
# ---------------------------------------------------------------------------

def make_vertical_lines(n=64, num_lines=None, line_width=None, margin=6):
    """Random vertical lines."""
    g = np.zeros((n, n), dtype=np.float32)
    if num_lines is None:
        num_lines = np.random.randint(2, 8)
    if line_width is None:
        line_width = np.random.randint(2, 6)

    positions = np.sort(np.random.choice(
        range(margin, n - margin - line_width), size=num_lines, replace=False
    ))
    y_start = np.random.randint(4, 15)
    y_end = np.random.randint(n - 15, n - 4)
    for x in positions:
        g[y_start:y_end, x:x + line_width] = 1
    return g


def make_horizontal_lines(n=64, num_lines=None, line_width=None, margin=6):
    """Random horizontal lines."""
    return make_vertical_lines(n, num_lines, line_width, margin).T


def make_contact_holes(n=64, num_holes=None, hole_size=None):
    """Random contact holes (small squares)."""
    g = np.zeros((n, n), dtype=np.float32)
    if num_holes is None:
        num_holes = np.random.randint(3, 12)
    if hole_size is None:
        hole_size = np.random.randint(3, 7)

    margin = hole_size + 2
    for _ in range(num_holes):
        y = np.random.randint(margin, n - margin)
        x = np.random.randint(margin, n - margin)
        g[y:y + hole_size, x:x + hole_size] = 1
    return g


def make_l_shape(n=64):
    """Random L-shaped feature."""
    g = np.zeros((n, n), dtype=np.float32)
    w = np.random.randint(3, 6)
    max_arm = min(30, n - 30)
    arm_len = np.random.randint(12, max(13, max_arm))

    margin = 8
    cx = np.random.randint(margin, max(margin + 1, n - margin - arm_len))
    cy = np.random.randint(margin, max(margin + 1, n - margin - arm_len))

    # Vertical arm
    g[cy:cy + arm_len, cx:cx + w] = 1
    # Horizontal arm
    g[cy + arm_len - w:cy + arm_len, cx:cx + arm_len] = 1
    return g


def make_random_rectangles(n=64, num_rects=None):
    """Random assortment of rectangles."""
    g = np.zeros((n, n), dtype=np.float32)
    if num_rects is None:
        num_rects = np.random.randint(2, 8)

    for _ in range(num_rects):
        w = np.random.randint(3, 12)
        h = np.random.randint(3, 20)
        x = np.random.randint(3, n - w - 3)
        y = np.random.randint(3, n - h - 3)
        g[y:y + h, x:x + w] = 1
    return g


MASK_GENERATORS = [
    make_vertical_lines,
    make_horizontal_lines,
    make_contact_holes,
    make_l_shape,
    make_random_rectangles,
]


def generate_random_mask(n=64):
    """Pick a random generator and produce a mask."""
    gen = MASK_GENERATORS[np.random.randint(len(MASK_GENERATORS))]
    return gen(n)


# ---------------------------------------------------------------------------
# Simulation with pre-built optics (avoids redundant setup)
# ---------------------------------------------------------------------------

class SimulationContext:
    """Pre-computes light source and pupil for repeated simulation."""

    def __init__(self, pixel_number=64, pixel_size=25, wavelength=193.0,
                 sigma_in=0.4, sigma_out=0.8, na=0.7, aberrations=None,
                 source_type='annular'):
        self.pixel_number = pixel_number
        self.pixel_size = pixel_size
        self.wavelength = wavelength

        ls = LightSource(sigma_in=sigma_in, sigma_out=sigma_out,
                         pixel_number=pixel_number, na=na)
        if source_type == 'quasar':
            self.light_source = ls.generate_quasar()
        else:
            self.light_source = ls.generate_annular()

        pupil = Pupil(pixel_number, wavelength, na, aberrations)
        self.pupil_function = pupil.generate_pupil_function()

    def simulate(self, geometry):
        """Run simulation on a single mask geometry.

        Args:
            geometry: 2D numpy array (pixel_number x pixel_number), binary.

        Returns:
            aerial_image: 2D numpy array, normalized to [0, 1].
        """
        mask = Mask(geometry=geometry, pixel_size=self.pixel_size)
        aerial = abbe_image(
            mask, self.wavelength,
            self.light_source, self.pupil_function, fft=True
        )
        aerial = aerial.numpy()

        # Normalize to [0, 1]
        a_min, a_max = aerial.min(), aerial.max()
        if a_max - a_min > 0:
            aerial = (aerial - a_min) / (a_max - a_min)
        else:
            aerial = np.zeros_like(aerial)
        return aerial


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(num_samples, pixel_number=64, pixel_size=25,
                     wavelength=193.0, sigma_in=0.4, sigma_out=0.8,
                     na=0.7, seed=None, verbose=True):
    """Generate a dataset of (mask, aerial_image) pairs.

    Args:
        num_samples: Number of samples to generate.
        pixel_number: Grid size.
        pixel_size: Pixel size in nm.
        wavelength: Illumination wavelength in nm.
        sigma_in: Inner partial coherence.
        sigma_out: Outer partial coherence.
        na: Numerical aperture.
        seed: Random seed for reproducibility.
        verbose: Print progress.

    Returns:
        masks: np.ndarray of shape (num_samples, pixel_number, pixel_number, 1).
        aerials: np.ndarray of shape (num_samples, pixel_number, pixel_number, 1).
    """
    if seed is not None:
        np.random.seed(seed)

    ctx = SimulationContext(
        pixel_number=pixel_number, pixel_size=pixel_size,
        wavelength=wavelength, sigma_in=sigma_in, sigma_out=sigma_out, na=na
    )

    masks = np.zeros((num_samples, pixel_number, pixel_number, 1),
                     dtype=np.float32)
    aerials = np.zeros((num_samples, pixel_number, pixel_number, 1),
                       dtype=np.float32)

    for i in range(num_samples):
        geom = generate_random_mask(pixel_number)
        aerial = ctx.simulate(geom)

        masks[i, :, :, 0] = geom
        aerials[i, :, :, 0] = aerial

        if verbose and (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")

    if verbose:
        print(f"  Dataset complete: {num_samples} samples")

    return masks, aerials


def save_dataset(masks, aerials, path='litho_dataset.npz'):
    """Save dataset to compressed .npz file."""
    np.savez_compressed(path, masks=masks, aerials=aerials)
    print(f"Dataset saved to {path} "
          f"({os.path.getsize(path) / 1024:.1f} KB)")


def load_dataset(path='litho_dataset.npz'):
    """Load dataset from .npz file."""
    data = np.load(path)
    return data['masks'], data['aerials']


def make_tf_dataset(masks, aerials, batch_size=8, shuffle=True):
    """Create a tf.data.Dataset from numpy arrays.

    Args:
        masks: (N, H, W, 1) input masks.
        aerials: (N, H, W, 1) target aerial images.
        batch_size: Batch size.
        shuffle: Whether to shuffle.

    Returns:
        tf.data.Dataset yielding (mask_batch, aerial_batch).
    """
    ds = tf.data.Dataset.from_tensor_slices((masks, aerials))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(masks))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate lithography simulation dataset'
    )
    parser.add_argument('-n', '--num-samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('-o', '--output', type=str,
                        default='litho_dataset.npz',
                        help='Output file path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print(f"Generating {args.num_samples} samples...")
    masks, aerials = generate_dataset(
        args.num_samples, seed=args.seed
    )
    save_dataset(masks, aerials, args.output)
