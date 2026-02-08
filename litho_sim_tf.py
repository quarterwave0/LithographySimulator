"""
Lithography simulator ported to TensorFlow.

Consolidates mask.py, lightsource.py, pupil.py, and imageformation.py
into a single TensorFlow-native module. Uses float32 for numerical
stability (original used float16 in many places).

Original PyTorch implementation by the LithographySimulator project.
"""

import tensorflow as tf
import numpy as np
from math import factorial, sqrt, ceil


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------

class Mask:
    """Binary photomask with Fraunhofer diffraction computation."""

    def __init__(self, geometry=None, pixel_size=25, pixel_number=64):
        """
        Args:
            geometry: 2D numpy array or tf.Tensor (square, binary 0/1).
                      If None, uses a demo mask with vertical lines.
            pixel_size: Physical size of each pixel in nm.
            pixel_number: Grid size (only used when geometry is None).
        """
        if geometry is not None:
            geometry = np.asarray(geometry, dtype=np.float32)
            if geometry.ndim != 2 or geometry.shape[0] != geometry.shape[1]:
                raise ValueError("Mask geometry must be a square 2D array")
            self.geometry = tf.constant(geometry, dtype=tf.float32)
            self.pixel_number = geometry.shape[0]
        else:
            self.pixel_number = pixel_number
            g = np.zeros((pixel_number, pixel_number), dtype=np.float32)
            g[9:55, 16:20] = 1
            g[9:55, 25:29] = 1
            g[9:55, 34:38] = 1
            g[9:55, 43:47] = 1
            self.geometry = tf.constant(g, dtype=tf.float32)

        self.pixel_size = pixel_size
        self._pixel_bound = self.pixel_number / 2 * self.pixel_size
        self.delta_k = 4.0 / self.pixel_number
        self._k_bound = self.pixel_number / 2 * self.delta_k

    def fraunhofer(self, wavelength, fft=True):
        """Compute Fraunhofer diffraction pattern of the mask.

        Args:
            wavelength: Illumination wavelength in nm.
            fft: If True, use FFT-accelerated method (recommended).

        Returns:
            Complex tf.Tensor of shape (pixel_number, pixel_number).
        """
        if fft:
            epsilon, N = self._calculate_epsilon_n(wavelength)
            return self._ff_fraunhofer(epsilon, N)
        else:
            return self._classical_fraunhofer(wavelength)

    def _calculate_epsilon_n(self, wavelength):
        beta = ((self.delta_k * self.pixel_size) / wavelength) ** -1
        N = _nearest_pow2(beta)
        epsilon = N / beta
        return epsilon, N

    def _ff_fraunhofer(self, epsilon, N):
        """FFT-based Fraunhofer diffraction (from Gau et al. 2023)."""
        N = int(N)
        geom = self.geometry

        # Scale mask by epsilon using bilinear interpolation
        new_size = int(round(self.pixel_number * epsilon))
        scaled = tf.image.resize(
            geom[tf.newaxis, :, :, tf.newaxis],
            [new_size, new_size],
            method='bilinear'
        )[0, :, :, 0]

        # Pad to NxN
        pad_total = N - scaled.shape[0]
        pw = pad_total // 2
        corr = int(scaled.shape[0] % 2)
        padded = tf.pad(scaled, [[pw, pw + corr], [pw, pw + corr]])

        # FFT
        standard_form = tf.signal.fftshift(tf.cast(padded, tf.complex64))
        fraunhofer_fft = tf.signal.fft2d(standard_form)
        fft_result = tf.signal.ifftshift(fraunhofer_fft)

        # Trim back to pixel_number x pixel_number
        trim = (N - self.pixel_number) // 2
        fft_result = fft_result[trim:trim + self.pixel_number,
                                trim:trim + self.pixel_number]
        return fft_result

    def _classical_fraunhofer(self, wavelength):
        """Classical (slow) Fraunhofer via direct integration."""
        pn = self.pixel_number
        fraunhofer_const = tf.constant(
            (2j * np.pi) / wavelength, dtype=tf.complex64
        )

        kx = tf.cast(
            tf.linspace(-self._k_bound, self._k_bound - self.delta_k, pn),
            tf.float32
        )
        xs = tf.cast(
            tf.linspace(-self._pixel_bound,
                         self._pixel_bound - self.pixel_size, pn),
            tf.float32
        )

        # (pn, pn, 1, 1) x (1, 1, pn, pn) via broadcasting
        KX, KY = tf.meshgrid(kx, kx, indexing='xy')
        XS, YS = tf.meshgrid(xs, xs, indexing='xy')

        k_grid = tf.stack([KX, KY], axis=-1)         # (pn,pn,2)
        xy_grid = tf.stack([XS, YS], axis=-1)         # (pn,pn,2)

        k_grid = k_grid[:, :, tf.newaxis, tf.newaxis, :]   # (pn,pn,1,1,2)
        xy_grid = xy_grid[tf.newaxis, tf.newaxis, :, :, :]  # (1,1,pn,pn,2)

        dot = tf.reduce_sum(k_grid * xy_grid, axis=-1)  # (pn,pn,pn,pn)
        exponent = tf.cast(dot, tf.complex64) * fraunhofer_const

        geom_c = tf.cast(self.geometry, tf.complex64)
        intermediate = geom_c[tf.newaxis, tf.newaxis, :, :] * tf.exp(exponent)

        # Trapezoidal integration over last two dims
        solution = tf.reduce_sum(intermediate, axis=[2, 3])
        return solution


# ---------------------------------------------------------------------------
# Light Source
# ---------------------------------------------------------------------------

class LightSource:
    """Illumination source for partial coherence imaging."""

    def __init__(self, sigma_in=0.0, sigma_out=0.6, pixel_number=64,
                 na=0.7, shift_x=0, shift_y=0):
        """
        Args:
            sigma_in: Inner partial coherence factor.
            sigma_out: Outer partial coherence factor.
            pixel_number: Grid size.
            na: Projection numerical aperture.
            shift_x: Source shift in x (units of wavelength/NA).
            shift_y: Source shift in y.
        """
        self.pixel_number = pixel_number
        self.NA = na
        self.sigma_inner = sigma_in
        self.sigma_outer = sigma_out
        self.shift_x = shift_x
        self.shift_y = shift_y

    def generate_annular(self):
        """Generate annular (ring) light source.

        Returns:
            tf.Tensor of shape (pixel_number, pixel_number), values 0 or 1.
        """
        sigma_span = 2.0
        delta_sigma = sigma_span * 2 / self.pixel_number

        sx = tf.cast(tf.linspace(
            -sigma_span - self.shift_x,
            sigma_span - self.shift_x - delta_sigma,
            self.pixel_number
        ), tf.float32)
        sy = tf.cast(tf.linspace(
            -sigma_span - self.shift_y,
            sigma_span - self.shift_y - delta_sigma,
            self.pixel_number
        ), tf.float32)

        SX, SY = tf.meshgrid(sx, sy, indexing='xy')
        O = tf.sqrt(SX ** 2 + SY ** 2)

        ls = tf.where(
            (O >= self.sigma_inner) & (O <= self.sigma_outer),
            tf.ones_like(O), tf.zeros_like(O)
        )
        return ls

    def generate_quasar(self, count=4, rotation=None):
        """Generate quasar (multi-pole) light source.

        Args:
            count: Number of poles.
            rotation: Angular rotation in radians.
                      Default: -pi/(count*2).

        Returns:
            tf.Tensor of shape (pixel_number, pixel_number), values 0 or 1.
        """
        if rotation is None:
            rotation = -np.pi / (count * 2)

        sigma_span = 2.0
        delta_sigma = sigma_span * 2 / self.pixel_number

        sx = tf.cast(tf.linspace(
            -sigma_span - self.shift_x,
            sigma_span - self.shift_x - delta_sigma,
            self.pixel_number
        ), tf.float32)
        sy = tf.cast(tf.linspace(
            -sigma_span - self.shift_y,
            sigma_span - self.shift_y - delta_sigma,
            self.pixel_number
        ), tf.float32)

        SX, SY = tf.meshgrid(sx, sy, indexing='xy')
        O = tf.sqrt(SX ** 2 + SY ** 2)
        theta = tf.math.atan2(SY, SX) + rotation
        theta = theta % (2 * np.pi)

        annular = tf.where(
            (O >= self.sigma_inner) & (O <= self.sigma_outer),
            tf.ones_like(O), tf.zeros_like(O)
        )

        ls = annular
        angular_spacing = np.pi / count
        for gap in range(count):
            gap_mask = tf.where(
                (theta > (gap + gap) * angular_spacing) &
                (theta < (gap + gap + 1) * angular_spacing),
                tf.zeros_like(O), tf.ones_like(O)
            )
            ls = ls * gap_mask

        return ls


# ---------------------------------------------------------------------------
# Pupil Function (Zernike Polynomials)
# ---------------------------------------------------------------------------

class Pupil:
    """Optical pupil function with Zernike polynomial aberrations."""

    def __init__(self, pixel_number=64, wavelength=193.0, na=0.7,
                 aberrations=None):
        """
        Args:
            pixel_number: Grid size.
            wavelength: Illumination wavelength in nm.
            na: Projection numerical aperture.
            aberrations: 1D array of Zernike coefficients (OSA indexing).
                         If None, assumes a perfect system (all zeros).
        """
        self.pixel_number = pixel_number
        self.wavelength = wavelength
        self.NA = na

        if aberrations is None:
            self.aberrations = np.array([0.0], dtype=np.float32)
        else:
            self.aberrations = np.array(aberrations, dtype=np.float32).copy()

    def generate_pupil_function(self):
        """Generate the complex pupil function.

        Returns:
            Complex tf.Tensor of shape (pixel_number, pixel_number).
        """
        we = _generate_wavefront_error(
            self.aberrations, self.pixel_number, self.NA, self.wavelength
        )
        phi = _generate_phi(we, self.pixel_number)
        return phi

    def generate_wavefront_error(self):
        """Generate the wavefront error map (before exponentiation).

        Returns:
            Complex tf.Tensor of shape (pixel_number, pixel_number).
        """
        return _generate_wavefront_error(
            self.aberrations, self.pixel_number, self.NA, self.wavelength
        )


def _dirac_delta(v):
    return 1 if v == 0 else 0


def _osa_index_to_mn(ji):
    """Convert OSA single-index to (m, n) Zernike indices."""
    n = ceil(0.5 * (-3 + sqrt(9 + 8 * ji)))
    m = 2 * ji - n * (n + 2)
    return m, n


def _generate_z(m, n, pixel_number, coeff):
    """Generate a single Zernike mode on the unit disk."""
    sigma_span = 2.0
    delta_sigma = sigma_span * 2 / pixel_number

    x = np.linspace(-sigma_span, sigma_span - delta_sigma,
                     pixel_number).astype(np.float32)
    X, Y = np.meshgrid(x, x, indexing='xy')
    r = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)

    l_lim = int((n - abs(m)) / 2)
    il_lim = int((n + abs(m)) / 2)

    R = np.zeros((pixel_number, pixel_number), dtype=np.float32)
    for k in range(l_lim + 1):
        static_coeff = (
            ((-1) ** k * factorial(n - k)) /
            (factorial(k) * factorial(il_lim - k) * factorial(l_lim - k))
        )
        R += static_coeff * r ** (n - 2 * k)

    Nmn = sqrt((2 * n + 1) / (1 + _dirac_delta(m)))

    if m >= 0:
        Z = coeff * Nmn * R * np.cos(m * theta)
    else:
        Z = coeff * (-Nmn) * R * np.sin(m * theta)

    Z = np.where(r <= 1, Z, 0)
    return tf.constant(Z, dtype=tf.float32)


def _generate_wavefront_error(aberrations, pixel_number, na, wavelength):
    """Sum Zernike modes weighted by aberration coefficients."""
    aberrations = aberrations.copy()
    if len(aberrations) >= 5:
        aberrations[4] = aberrations[4] * na ** 2 / (4 * wavelength)

    we = tf.zeros((pixel_number, pixel_number), dtype=tf.float32)
    for i in range(len(aberrations)):
        m, n = _osa_index_to_mn(i)
        Z = _generate_z(m, n, pixel_number, aberrations[i])
        we = we + Z

    return tf.cast(we, tf.complex64)


def _generate_phi(we, pixel_number):
    """Convert wavefront error to complex pupil function."""
    phi = tf.exp(1j * 2 * np.pi * we)

    sigma_span = 2.0
    delta_sigma = sigma_span * 2 / pixel_number
    x = np.linspace(-sigma_span, sigma_span - delta_sigma,
                     pixel_number).astype(np.float32)
    X, Y = np.meshgrid(x, x, indexing='xy')
    r = np.sqrt(X ** 2 + Y ** 2)
    disk = tf.constant(r <= 1, dtype=tf.float32)

    return phi * tf.cast(disk, tf.complex64)


# ---------------------------------------------------------------------------
# Image Formation (Abbe Partial Coherence)
# ---------------------------------------------------------------------------

def calculate_fft_aerial(pupil_fn, mask_fft, pixel_number, N):
    """Compute aerial image contribution for a single source point (FFT).

    Args:
        pupil_fn: Shifted pupil function, complex (pixel_number, pixel_number).
        mask_fft: Mask Fraunhofer diffraction, complex (pixel_number, pixel_number).
        pixel_number: Grid size.
        N: FFT padding size.

    Returns:
        Complex tf.Tensor (pixel_number, pixel_number).
    """
    N = int(N)
    pf_amp = pupil_fn * mask_fft

    pw = (N - pixel_number) // 2
    padded = tf.pad(pf_amp, [[pw, pw], [pw, pw]])

    standard_form = tf.signal.fftshift(padded)
    abbe_fft = tf.signal.ifft2d(standard_form) * tf.cast(
        tf.constant(N * N, dtype=tf.float32), tf.complex64
    )
    unrolled = tf.signal.ifftshift(abbe_fft)

    trimmed = unrolled[pw:pw + pixel_number, pw:pw + pixel_number]
    return trimmed


def abbe_image(mask, wavelength=193.0, light_source_tensor=None,
               pupil_function=None, fft=True):
    """Compute aerial image using Abbe partial coherence formulation.

    This is the main entry point for simulation. It accepts pre-built
    components or uses defaults.

    Args:
        mask: Mask instance.
        wavelength: Illumination wavelength in nm.
        light_source_tensor: 2D tensor from LightSource.generate_*().
            If None, generates a default annular source.
        pupil_function: Complex 2D tensor from Pupil.generate_pupil_function().
            If None, generates a perfect pupil.
        fft: Use FFT-accelerated computation.

    Returns:
        Real tf.Tensor of shape (pixel_number, pixel_number), the aerial
        image intensity.
    """
    pixel_number = mask.pixel_number

    # Defaults
    if light_source_tensor is None:
        ls = LightSource(sigma_in=0.4, sigma_out=0.8,
                         pixel_number=pixel_number)
        light_source_tensor = ls.generate_annular()

    if pupil_function is None:
        pupil = Pupil(pixel_number, wavelength)
        pupil_function = pupil.generate_pupil_function()

    # Compute mask diffraction
    mask_ft = mask.fraunhofer(wavelength, fft=fft)

    if fft:
        epsilon, N = mask._calculate_epsilon_n(wavelength)
    else:
        fraunhofer_const = tf.constant(
            (-2j * np.pi) / wavelength, dtype=tf.complex64
        )

    # Find active source points
    ls_np = light_source_tensor.numpy()
    source_points = np.argwhere(ls_np)  # (num_points, 2)
    center = pixel_number // 2
    shifts = source_points - center  # relative shifts

    # Accumulate image
    image = tf.zeros((pixel_number, pixel_number), dtype=tf.float32)

    for i in range(len(shifts)):
        shift_y, shift_x = int(shifts[i, 0]), int(shifts[i, 1])
        pupil_shifted = tf.roll(pupil_function, shift=[shift_y, shift_x],
                                axis=[0, 1])

        if fft:
            aerial = calculate_fft_aerial(
                pupil_shifted, mask_ft, pixel_number, N
            )
        else:
            # Classical path (slow, for reference)
            aerial = _calculate_classical_aerial(
                pupil_shifted, mask_ft, fraunhofer_const,
                pixel_number, mask.pixel_size
            )

        image = image + tf.abs(aerial) ** 2

    # For FFT mode, rescale and trim
    if fft:
        image = tf.abs(image)
        new_size = int(round(pixel_number / epsilon))
        image_resized = tf.image.resize(
            image[tf.newaxis, :, :, tf.newaxis],
            [new_size, new_size],
            method='bilinear'
        )[0, :, :, 0]

        pw = (pixel_number - new_size) // 2
        if pw > 0:
            image = tf.pad(image_resized,
                           [[pw, pixel_number - new_size - pw],
                            [pw, pixel_number - new_size - pw]])
        elif pw < 0:
            pw = -pw
            image = image_resized[pw:pw + pixel_number,
                                  pw:pw + pixel_number]
        else:
            image = image_resized

        # Ensure output is exactly pixel_number x pixel_number
        if image.shape[0] != pixel_number or image.shape[1] != pixel_number:
            image = tf.image.resize(
                image[tf.newaxis, :, :, tf.newaxis],
                [pixel_number, pixel_number],
                method='bilinear'
            )[0, :, :, 0]

    return tf.math.real(tf.cast(image, tf.complex64))


def _calculate_classical_aerial(pupil_fn, mask_ft, fraunhofer_const,
                                pixel_number, pixel_size):
    """Classical (non-FFT) aerial image for a single source point."""
    pixel_bound = pixel_number / 2 * pixel_size
    delta_k = 4.0 / pixel_number
    k_bound = pixel_number / 2 * delta_k

    kx = tf.cast(tf.linspace(-k_bound, k_bound - delta_k, pixel_number),
                 tf.float32)
    xs = tf.cast(tf.linspace(-pixel_bound, pixel_bound - pixel_size,
                              pixel_number), tf.float32)

    KX, KY = tf.meshgrid(kx, kx, indexing='xy')
    XS, YS = tf.meshgrid(xs, xs, indexing='xy')

    k_grid = tf.stack([KX, KY], axis=-1)[:, :, tf.newaxis, tf.newaxis, :]
    xy_grid = tf.stack([XS, YS], axis=-1)[tf.newaxis, tf.newaxis, :, :, :]

    dot = tf.reduce_sum(k_grid * xy_grid, axis=-1)
    exponent = tf.cast(dot, tf.complex64) * fraunhofer_const

    intermediate = pupil_fn[:, :, tf.newaxis, tf.newaxis] * \
                   mask_ft[:, :, tf.newaxis, tf.newaxis] * tf.exp(exponent)

    solution = tf.reduce_sum(intermediate, axis=[2, 3])
    return solution


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _nearest_pow2(value):
    """Find the nearest power of 2 to the given value."""
    powers = [2 ** i for i in range(1, 15)]
    idx = np.argmin([abs(p - value) for p in powers])
    return powers[idx]


def simulate(geometry=None, pixel_size=25, pixel_number=64,
             wavelength=193.0, sigma_in=0.4, sigma_out=0.8, na=0.7,
             aberrations=None, source_type='annular',
             quasar_count=4, quasar_rotation=None):
    """High-level convenience function to run a full simulation.

    Args:
        geometry: 2D binary mask array. None for demo mask.
        pixel_size: Pixel size in nm.
        pixel_number: Grid size (used when geometry is None).
        wavelength: Illumination wavelength in nm.
        sigma_in: Inner partial coherence factor.
        sigma_out: Outer partial coherence factor.
        na: Projection numerical aperture.
        aberrations: Zernike coefficients (OSA). None for perfect lens.
        source_type: 'annular' or 'quasar'.
        quasar_count: Number of poles for quasar source.
        quasar_rotation: Rotation for quasar. None for default.

    Returns:
        tuple: (mask_geometry, aerial_image) as numpy arrays.
    """
    mask = Mask(geometry=geometry, pixel_size=pixel_size,
                pixel_number=pixel_number)

    ls = LightSource(sigma_in=sigma_in, sigma_out=sigma_out,
                     pixel_number=mask.pixel_number, na=na)
    if source_type == 'quasar':
        ls_tensor = ls.generate_quasar(quasar_count, quasar_rotation)
    else:
        ls_tensor = ls.generate_annular()

    pupil = Pupil(mask.pixel_number, wavelength, na, aberrations)
    pf = pupil.generate_pupil_function()

    aerial = abbe_image(mask, wavelength, ls_tensor, pf, fft=True)

    return mask.geometry.numpy(), aerial.numpy()
