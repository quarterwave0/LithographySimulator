# Roadmap: Incorporating Global Receptive Field into the Shift-Equivariant U-Net

## 1. Current Architecture Analysis

### 1.1 Architecture Summary

The current shift-equivariant U-Net replaces pooling/upsampling with **dilated convolutions** and uses **circular (wrap-around) padding** throughout. For the 64×64 lithography model (`train.py`):

| Stage | Filters | Dilation | Effective Kernel | Per-Layer RF Growth |
|-------|---------|----------|-----------------|-------------------|
| Encoder 1 | 32 | 1 | 3×3 | +2 px |
| Encoder 2 | 64 | 2 | 5×5 | +4 px |
| Encoder 3 | 128 | 4 | 9×9 | +8 px |
| Bottleneck | 256 | 8 | 17×17 | +16 px |
| Decoder 3 | 128 | 4 | 9×9 | +8 px |
| Decoder 2 | 64 | 2 | 5×5 | +4 px |
| Decoder 1 | 32 | 1 | 3×3 | +2 px |

- **Parameters**: ~488K
- **Theoretical receptive field** (stacking all layers): ~89×89 — exceeds the 64×64 grid, so *theoretically* every pixel can see every other pixel.
- **Effective receptive field** (ERF): Much smaller. Research shows ERF follows a Gaussian distribution concentrated in the center, typically covering only 30–50% of the theoretical RF. For this network, the ERF is estimated at **~30–40 pixels** diameter — roughly half the grid.

### 1.2 Why Global Context Matters for Lithography

The lithography aerial image formation is fundamentally a **frequency-domain process**:

1. **Diffraction is global**: The simulator computes `FFT(mask) × pupil_filter`, meaning every output pixel depends on *all* input pixels through the frequency domain.
2. **Optical Proximity Effects (OPE)**: Features separated by several wavelengths (10–20+ pixels at 25nm pitch) still interact through diffraction.
3. **Periodic boundary conditions**: The simulator assumes periodic masks, perfectly matching our circular padding — but the CNN still lacks true global coupling within a single forward pass.
4. **Frequency filtering**: The pupil function acts as a low-pass filter in frequency space. A frequency-aware network could directly learn this operation.

### 1.3 Current Limitations

| Limitation | Impact |
|-----------|--------|
| No true global mixing | Cannot capture long-range mask interactions in a single layer |
| Purely spatial operations | No explicit frequency-domain reasoning, despite the physics being FFT-based |
| ERF << grid size | Peripheral regions of the mask contribute weakly to center predictions |
| All context via stacking | Deep stacking is parameter-inefficient for global context |

---

## 2. Candidate Methods

We surveyed three families of approaches: **frequency-domain methods**, **attention mechanisms**, and **MLP/global-mixing layers**. Each is evaluated on:
- **Global reach**: Does it provide true global receptive field?
- **Shift equivariance**: Is it compatible with circular padding / toroidal topology?
- **Overhead**: Parameter count and FLOPs increase
- **Complexity**: Implementation difficulty and ONNX export compatibility

### 2.1 Frequency-Domain Methods

#### 2.1.1 Fast Fourier Convolution (FFC) / LaMa-style Dual Branch
- **Papers**: Chi et al. "Fast Fourier Convolution" (NeurIPS 2020); Suvorov et al. "LaMa" (WACV 2022)
- **Idea**: Split channels into a local branch (standard convolution) and a spectral branch (Real FFT → 1×1 conv in frequency domain → iFFT). The two branches exchange information.
- **Global reach**: ✅ Full — FFT gives instant global context
- **Shift equivariance**: ✅ Circular FFT is inherently shift-equivariant (circular convolution theorem)
- **Overhead**: Low — spectral branch uses 1×1 convolutions on frequency coefficients. For 64×64 with C channels: FFT costs O(N² log N) ≈ O(64² × 12) ≈ 50K multiply-adds, negligible vs convolution costs
- **ONNX**: ⚠️ Requires custom FFT op or tf.signal export. tf2onnx supports `tf.signal.rfft2d`/`irfft2d` from opset 17+
- **Physics alignment**: ★★★★★ — The lithography simulator itself is FFT-based. A spectral branch directly mirrors the physics.

#### 2.1.2 Adaptive Fourier Neural Operator (AFNO)
- **Papers**: Guibas et al. "AFNO" (NeurIPS 2021 workshop); Pathak et al. "FourCastNet" (NVIDIA, 2022)
- **Idea**: Token mixing via FFT → element-wise MLP in frequency domain → iFFT. Used in weather prediction at 720×1440 resolution.
- **Global reach**: ✅ Full
- **Shift equivariance**: ✅ Yes (pointwise ops in frequency domain commute with shifts)
- **Overhead**: Very low — only 2 dense layers per frequency coefficient. For 64×64×C: ~2×C² parameters per block.
- **ONNX**: Same FFT export considerations as FFC
- **Physics alignment**: ★★★★★ — Designed for PDE-governed systems, directly applicable to wave optics

#### 2.1.3 Global Filter Network (GFNet)
- **Paper**: Rao et al. (NeurIPS 2021)
- **Idea**: Learnable global frequency-domain filter: `iFFT(FFT(x) ⊙ W)` where W is a learnable complex-valued weight matrix.
- **Global reach**: ✅ Full
- **Shift equivariance**: ✅ Yes (pointwise multiplication in frequency domain = circular convolution in spatial domain)
- **Overhead**: Minimal — W has shape (H, W//2+1, C) for real FFT, so 64×33×C ≈ 2112C parameters per layer
- **ONNX**: Same FFT considerations
- **Physics alignment**: ★★★★★ — This is *exactly* how the pupil function works in lithography (frequency-domain filtering)

#### 2.1.4 Wavelet-based Approaches (WATNet)
- **Paper**: Phutke et al. "WATNet" (2022)
- **Idea**: Discrete wavelet transform (DWT) for multi-resolution decomposition; process low-frequency (global) and high-frequency (detail) sub-bands separately.
- **Global reach**: Partial — multi-scale but not truly global in one step
- **Shift equivariance**: ⚠️ Standard DWT is NOT shift-equivariant (decimation). Stationary Wavelet Transform (SWT) preserves equivariance but is more expensive.
- **Overhead**: Moderate — wavelet decomposition adds ~4× channels per level
- **Physics alignment**: ★★★ — Useful for multi-scale analysis but doesn't directly mirror the physics

### 2.2 Attention Mechanisms

#### 2.2.1 Squeeze-and-Excitation (SE) Blocks
- **Paper**: Hu et al. (CVPR 2018)
- **Idea**: Global average pooling → FC → ReLU → FC → Sigmoid → channel-wise rescaling
- **Global reach**: ✅ Full (via global average pooling)
- **Shift equivariance**: ✅ Yes — GAP is shift-invariant, channel rescaling is pointwise
- **Overhead**: Negligible — adds only 2C²/r parameters per block (r=16 typical). For C=128: 2×128²/16 = 2048 params
- **ONNX**: ✅ Trivial — standard ops only
- **Physics alignment**: ★★★ — Captures global channel statistics but no spatial structure

#### 2.2.2 CBAM (Convolutional Block Attention Module)
- **Paper**: Woo et al. (ECCV 2018)
- **Idea**: SE-style channel attention + spatial attention (max/avg pool across channels → conv → sigmoid)
- **Global reach**: ✅ Channel: global; Spatial: global along channel axis
- **Shift equivariance**: ✅ Yes if spatial convolution uses circular padding
- **Overhead**: <0.2% FLOPs increase
- **ONNX**: ✅ Trivial
- **Physics alignment**: ★★★ — Spatial attention helps focus on feature-dense regions

#### 2.2.3 Attention Gates (AG) for U-Net Skip Connections
- **Paper**: Oktay et al. (MIDL 2018)
- **Idea**: Gated skip connections: `α = σ(W_x · x + W_g · g + b)` where x is the skip feature and g is the decoder feature. Spatial attention on skip connections.
- **Global reach**: Indirect — attention is spatially local but informed by decoder features with larger RF
- **Shift equivariance**: ✅ Yes if implemented with circular-padded convolutions
- **Overhead**: Low — one extra 1×1 conv per skip connection
- **ONNX**: ✅ Trivial
- **Physics alignment**: ★★★ — Helps the decoder selectively use encoder features

#### 2.2.4 Restormer-style Multi-DConv Head Transposed Attention (MDTA)
- **Paper**: Zamir et al. (CVPR 2022 oral)
- **Idea**: Transpose attention: Q, K, V from depthwise convolutions, attention computed across channels (C×C matrix) instead of spatial tokens (N×N). Complexity O(HWC²) instead of O(H²W²C).
- **Global reach**: ✅ Full (each channel interacts with all spatial positions through dot products)
- **Shift equivariance**: ✅ Yes — depthwise convolutions + channel attention; spatial structure preserved
- **Overhead**: Moderate — C² attention matrix. For C=128: 128² = 16K multiply-adds per spatial position
- **ONNX**: ✅ Standard matmul ops
- **Physics alignment**: ★★★★ — Channel attention captures cross-frequency interactions relevant to diffraction

#### 2.2.5 Neighborhood Attention (NAT / DiNAT)
- **Paper**: Hassani et al. (CVPR 2023)
- **Idea**: Each token attends only to its local k×k neighborhood. DiNAT adds dilated neighborhoods for larger reach.
- **Global reach**: Partial — local windows, but dilated variant extends reach significantly
- **Shift equivariance**: ✅ Yes — same neighborhood pattern applied everywhere
- **Overhead**: O(N × k²) where k is window size, much cheaper than full attention
- **ONNX**: ⚠️ Custom kernels (NATTEN) needed; non-trivial ONNX export
- **Physics alignment**: ★★★ — Good for local OPE but doesn't capture truly global effects

### 2.3 MLP / Global Mixing Layers

#### 2.3.0 UNeXt / UNeXt-ILT (Tokenized MLP — Domain-Validated)
- **Papers**: Valanarasu & Patel "UNeXt: MLP-based Rapid Medical Image Segmentation Network" (MICCAI 2022); Lin et al. "UNeXt-ILT: fast and global context-aware inverse lithography solution" (J. Micro/Nanopatterning, Materials, and Metrology, Jan 2025)
- **Idea**: A hybrid Conv + MLP U-Net architecture. The encoder starts with convolutional stages (local features), then transitions to **Tokenized MLP** blocks in the latent/bottleneck stages. Each Tokenized MLP block: (1) projects spatial features into abstract tokens via learned tokenization, (2) applies MLPs across tokens for global mixing, (3) uses **shifted channels** (axial shifts) to inject local spatial awareness into the MLP.
- **UNeXt-ILT results**: Applied directly to inverse lithography. Compared to SOTA DL-based ILT: **-17.83% L2 error**, **-8.76% PV band**, **-34.48% turnaround time**. These results validate that MLP-based global mixing is highly effective for lithography tasks specifically.
- **Global reach**: ✅ Full — MLPs across all spatial tokens provide global context; tokenization compresses spatial extent
- **Shift equivariance**: ⚠️ Partial — The original UNeXt uses strided convolutions (not shift-equivariant). However, the Tokenized MLP mechanism itself is compatible with shift equivariance if: (a) we replace strided convs with dilated circular convs (as in our current architecture), (b) the shifted-channel operation uses circular shifts (`tf.roll`), and (c) tokenization is done via 1×1 convolutions (position-independent). **Adaptation required but feasible.**
- **Overhead**: Very Low — UNeXt achieves 72× parameter reduction and 68× FLOPs reduction vs transformer-based alternatives. The Tokenized MLP block has only O(T × C) parameters where T is the token count.
- **ONNX**: ✅ Trivial — standard ops (linear projections, reshape, MLP). No FFT needed.
- **Physics alignment**: ★★★★★ — **Directly validated on lithography ILT** with significant improvements. The tokenized MLP can learn the frequency-domain mixing that characterizes optical diffraction, while the shifted-channel mechanism captures local OPE interactions.

#### 2.3.1 ConvMixer-style Depthwise Mixing
- **Paper**: Trockman & Kolter (ICLR 2022 workshop)
- **Idea**: Large-kernel depthwise convolutions (e.g., 9×9) for spatial mixing + pointwise convolutions for channel mixing.
- **Global reach**: Partial — depends on kernel size. A 33×33 depthwise conv would cover half the 64×64 grid.
- **Shift equivariance**: ✅ Yes with circular padding
- **Overhead**: Very low — depthwise conv has only k²×C parameters. 9×9 depthwise with C=128: 10K params
- **ONNX**: ✅ Trivial
- **Physics alignment**: ★★★ — Increases spatial mixing efficiently

#### 2.3.2 PoolFormer / MetaFormer Token Mixing
- **Paper**: Yu et al. (CVPR 2022 oral)
- **Idea**: Replace attention with simple average pooling for token mixing. Shows that the *architecture* (MetaFormer) matters more than the specific mixing mechanism.
- **Global reach**: Partial — pooling window determines reach
- **Shift equivariance**: ✅ Yes — average pooling is shift-equivariant
- **Overhead**: Minimal — pooling has zero learnable parameters
- **ONNX**: ✅ Trivial
- **Physics alignment**: ★★ — Simple but may not capture directional diffraction effects

#### 2.3.3 Large-Kernel Convolutions (RepLKNet / SLaK)
- **Paper**: Ding et al. "RepLKNet" (CVPR 2022); Liu et al. "SLaK" (ICLR 2023)
- **Idea**: Very large depthwise convolution kernels (31×31 or 51×51) with structural re-parameterization for efficient training.
- **Global reach**: Near-global for 64×64 — a 51×51 kernel with circular padding covers 80% of the grid
- **Shift equivariance**: ✅ Yes with circular padding
- **Overhead**: Moderate — 51×51 depthwise conv with C=128: 333K params per layer. But depthwise, so FLOPs manageable.
- **ONNX**: ✅ Re-parameterized form is a standard large conv
- **Physics alignment**: ★★★★ — Large kernels can approximate the PSF (point spread function) of the optical system

#### 2.3.4 HorNet (Recursive Gated Convolution)
- **Paper**: Rao et al. (NeurIPS 2022)
- **Idea**: `output = x ⊙ g(DWConv(x))` — gated depthwise convolution with recursive application for growing receptive field. gnConv order can be increased for larger RF.
- **Global reach**: Tunable — higher-order gnConv approaches global reach
- **Shift equivariance**: ✅ Yes with circular padding
- **Overhead**: Low — only depthwise convs and pointwise operations
- **ONNX**: ✅ Standard ops
- **Physics alignment**: ★★★★ — Recursive gating models multi-order optical interactions

---

## 3. Comparative Summary

| Method | Global Reach | Shift Equivariant | Params Overhead | FLOPs Overhead | ONNX Export | Physics Fit |
|--------|-------------|-------------------|-----------------|---------------|-------------|-------------|
| **UNeXt Tokenized MLP** | ✅ Full | ⚠️ Needs adapt | Very Low | Very Low | ✅ | ★★★★★ |
| **FFC (Spectral Branch)** | ✅ Full | ✅ | Low (~5-10%) | Low | ⚠️ FFT ops | ★★★★★ |
| **AFNO** | ✅ Full | ✅ | Very Low | Very Low | ⚠️ FFT ops | ★★★★★ |
| **GFNet** | ✅ Full | ✅ | Very Low | Low | ⚠️ FFT ops | ★★★★★ |
| **SE Blocks** | ✅ Global stats | ✅ | Negligible | Negligible | ✅ | ★★★ |
| **CBAM** | ✅ Channel+Spatial | ✅ | Negligible | Negligible | ✅ | ★★★ |
| **Attention Gates** | Indirect | ✅ | Low | Low | ✅ | ★★★ |
| **Restormer MDTA** | ✅ Full | ✅ | Moderate | Moderate | ✅ | ★★★★ |
| **Large-Kernel Conv** | Near-global | ✅ | Moderate | Moderate | ✅ | ★★★★ |
| **HorNet gnConv** | Tunable | ✅ | Low | Low | ✅ | ★★★★ |
| **ConvMixer DW** | Partial | ✅ | Very Low | Very Low | ✅ | ★★★ |

---

## 4. Recommended Roadmap

We propose a **3-phase implementation plan**, ordered by impact-to-effort ratio:

### Phase 1: Quick Wins (Low Risk, High Compatibility)
**Goal**: Add global context with minimal code changes and guaranteed ONNX compatibility.

#### 1A. SE Blocks at Encoder/Decoder Boundaries
- **Where**: After each encoder stage and before each skip connection concatenation
- **Cost**: ~10K extra parameters (<2% increase)
- **Benefit**: Global channel recalibration captures which frequency bands are active
- **Implementation**: ~30 lines of code, drop-in after any conv block

#### 1B. Attention Gates on Skip Connections
- **Where**: Replace direct `Concatenate([decoder, encoder])` with gated skip connections
- **Cost**: ~15K extra parameters
- **Benefit**: Decoder can selectively suppress irrelevant encoder features; proven effective in medical image U-Nets with similar resolution
- **Implementation**: ~40 lines, replaces 3 concatenation points

**Expected outcome**: 5–15% MSE reduction from better feature selection, no shift-equivariance concerns, trivial ONNX export.

### Phase 2: Spectral Branch Integration (High Impact, Moderate Effort)
**Goal**: Add frequency-domain processing that directly mirrors the lithography physics.

#### 2A. GFNet-style Spectral Filter Layer at Bottleneck
- **Where**: Replace or augment the bottleneck (dilation=8) stage with a learnable frequency-domain filter
- **Design**:
  ```
  x_freq = FFT2D(x)                    # (64, 33, C) complex
  x_filtered = x_freq * W              # W is learnable (64, 33, C) complex
  x_spatial = iFFT2D(x_filtered)       # back to spatial
  output = x + x_spatial               # residual connection
  ```
- **Cost**: 64 × 33 × C_bottleneck = 64 × 33 × 256 ≈ 540K parameters (complex-valued → ~1M real params). Can reduce with low-rank factorization.
- **Why bottleneck**: This is where the network has the richest features and highest dilation. Adding global FFT here gives every bottleneck feature instant access to the full spatial extent.
- **Physics alignment**: This directly mimics `pupil_filter * FFT(mask)` — the core operation in lithography simulation.

#### 2B. FFC-style Dual Branch at Stage 3 and Bottleneck
- **Where**: Stages 3 and 4 (the deepest encoder/decoder stages)
- **Design**: Split channels 75% local / 25% spectral. The spectral branch processes in frequency domain using 1×1 convolutions.
- **Cost**: ~50K extra parameters per FFC block
- **Benefit**: Continuous local-global information exchange. More expressive than a single GFNet filter.
- **ONNX note**: Will need opset ≥17 for FFT support, or implement spectral branch as a matrix multiply (DFT matrix × input), which exports cleanly.

#### 2C. UNeXt-style Tokenized MLP at Bottleneck (Domain-Validated)
- **Where**: Replace bottleneck dilated convolutions with Tokenized MLP blocks
- **Design**:
  ```
  # Tokenize: project spatial features to tokens
  tokens = Conv1x1(x)                    # (B, H, W, C) → (B, H*W, T)
  # Shifted MLP: circular-shift channels, then MLP
  tokens_shifted = tf.roll(tokens, shift=s, axis=-1)  # circular channel shift
  tokens = MLP(tokens_shifted)           # global token mixing
  # De-tokenize: project back to spatial
  output = Conv1x1(tokens) + x           # residual connection
  ```
- **Cost**: Very low — for T=64 tokens, C=256: ~2×T×C ≈ 33K params per block
- **Why this is compelling**: UNeXt-ILT (Jan 2025) demonstrated **17.83% L2 error reduction** and **34.48% faster turnaround** on lithography ILT tasks using exactly this backbone. This is the only method in our survey with **direct lithography domain validation**.
- **Shift equivariance adaptation**: Replace UNeXt's original strided convolutions with our existing dilated circular convolutions. Use `tf.roll` for the channel-shifting operation (already shift-equivariant). Tokenization via 1×1 conv is position-independent → equivariant.
- **ONNX**: ✅ Standard ops only — no FFT needed, unlike Phases 2A/2B

**Expected outcome**: 15–30% MSE reduction, especially for patterns with strong periodicity (line/space gratings, contact arrays). The network can learn the physics rather than approximating it through many conv layers. Phase 2C (UNeXt MLP) offers the best risk-adjusted return given its domain validation.

### Phase 3: Advanced Global Mixing (Experimental)
**Goal**: Explore more expressive global mixing for further gains.

#### 3A. AFNO Block as Bottleneck Replacement
- **Where**: Replace the entire bottleneck with an AFNO-style block: FFT → MLP in frequency domain → iFFT
- **Design**:
  ```
  x_freq = FFT2D(x)                          # (64, 33, C) complex
  x_freq = MLP(x_freq)                       # 2-layer MLP per frequency bin
  x_spatial = iFFT2D(x_freq) + x             # residual
  ```
- **Benefit**: More expressive than GFNet (learnable nonlinear transform in frequency domain)
- **Cost**: ~2 × C² parameters for the MLP. C=256: ~130K params

#### 3B. Restormer-style Channel Attention at Decoder
- **Where**: Decoder stages, after skip connection fusion
- **Design**: Transposed attention (C×C matrix) computed from depthwise-conv features
- **Benefit**: Captures cross-channel (cross-frequency-band) interactions that are important for OPE
- **Cost**: Moderate — C² per head. With multi-head (4 heads), C=128: 4×32² = 4K per position

#### 3C. Large-Kernel Depthwise Convolution (RepLK-style)
- **Where**: Optional replacement for dilation-8 layers
- **Design**: Single 31×31 or 51×51 depthwise circular conv instead of dilated 3×3
- **Benefit**: Avoids gridding artifacts from high dilation rates; provides dense (not sparse) spatial coverage
- **Consideration**: Must use re-parameterization (sum of smaller kernels during training, merge for inference) for training stability

---

## 5. Implementation Priority Matrix

```
                    Low Effort ──────────────────── High Effort
                    │                                         │
High Impact    ★★★ │  (2C) UNeXt MLP ★★★★     (2B) FFC dual │ ★★★
                    │  (2A) GFNet bottleneck                   │
                    │  (1A) SE blocks                          │
                    │                                         │
                    │  (1B) Attention Gates     (3A) AFNO     │
                    │                          (3B) Restormer │
Low Impact     ★   │  ConvMixer DW             (3C) RepLK    │ ★
                    │  PoolFormer                              │
                    │                                         │
```

**(2C) UNeXt MLP is marked ★★★★ because it is the only method with direct lithography domain validation (UNeXt-ILT, Jan 2025).**

**Recommended implementation order**:
1. **SE Blocks** (1A) — easiest win, validate pipeline
2. **Attention Gates** (1B) — proven U-Net enhancement
3. **UNeXt Tokenized MLP** (2C) — **domain-validated** on lithography, lightweight, ONNX-friendly, no FFT export issues
4. **GFNet Bottleneck** (2A) — highest physics alignment, but needs FFT in ONNX
5. **FFC Dual Branch** (2B) — if GFNet shows promise, extend to dual-branch
6. **AFNO** (3A) — if spectral approach works, try nonlinear version
7. **Restormer MDTA** (3B) — if channel interactions matter
8. **RepLK** (3C) — alternative to dilation if gridding artifacts observed

> **Key insight from UNeXt-ILT**: The Tokenized MLP approach achieves comparable or better global context than FFT-based methods while being dramatically simpler to export (no FFT ops in ONNX). For our use case where ONNX compatibility is a hard requirement, this makes Phase 2C the **recommended primary path**, with FFT-based methods (2A, 2B) as complementary enhancements if further gains are needed.

---

## 6. Key Design Constraints

### 6.1 Shift Equivariance Preservation
All candidate methods preserve shift equivariance **provided**:
- Spatial convolutions use circular padding (already standard in our codebase)
- FFT is computed with periodic boundary (default for discrete FFT — matches our circular padding)
- No spatial pooling with stride > 1 (our architecture already avoids this)
- Attention is either global (position-independent) or uses relative position encoding with circular wrapping

### 6.2 ONNX Export Compatibility
- **Phase 1 methods**: All use standard ops (GAP, FC, sigmoid, conv). Zero ONNX issues.
- **Phase 2–3 FFT methods**: Two strategies:
  1. Use `tf.signal.rfft2d`/`irfft2d` with ONNX opset ≥ 17 (DFT op available)
  2. Pre-compute the DFT matrix and implement as `tf.matmul(DFT_matrix, x)` — exports as standard matmul. For 64×64, the DFT matrix is only 64×64 complex = 8KB. This is the **recommended approach** for maximum compatibility.

### 6.3 Computational Budget
Current model: ~488K parameters, ~X MFLOPs for 64×64 input.

| Phase | Additional Params | Relative Increase |
|-------|------------------|-------------------|
| 1 (SE + AG) | ~25K | +5% |
| 2C (UNeXt MLP) | ~33K per block | +7% |
| 2A (GFNet) | ~540K | +110% (but most are simple element-wise) |
| 2B (FFC) | ~100K | +20% |
| 3 (AFNO) | ~130K | +27% |

Phase 1 stays well within the "minimal overhead" goal. Phase 2C (UNeXt MLP) is the lightest Phase 2 option — only ~33K extra parameters per Tokenized MLP block, adding ~7% to the model. Phase 2A with GFNet doubles parameters but the new parameters are frequency-domain weights (element-wise multiply), not convolution weights, so FLOPs increase is modest.

---

## 7. Evaluation Plan

For each enhancement, measure:

1. **Prediction accuracy**: MSE, MAE, SSIM on held-out test set
2. **Shift equivariance**: Run `test_shift_equivariance()` — MAE should remain < 1e-3
3. **Per-pattern-type accuracy**: Evaluate separately on lines, contacts, L-shapes, random rectangles (different patterns stress different receptive field requirements)
4. **Inference speed**: Wall-clock time for single 64×64 inference (CPU)
5. **ONNX roundtrip**: Verify TF→ONNX conversion and numerical equivalence (MAE < 1e-5)
6. **Parameter count**: Track total model parameters

### Expected Key Result
The frequency-domain methods (Phase 2) should show the largest improvement on **periodic patterns** (line/space gratings, contact arrays) where global Fourier structure is strongest, while attention methods (Phase 1) should improve **isolated features** (L-shapes, random rectangles) where selective feature gating matters most.

---

## 8. References

1. Chi et al., "Fast Fourier Convolution," NeurIPS 2020
2. Suvorov et al., "Resolution-robust Large Mask Inpainting with Fourier Convolutions (LaMa)," WACV 2022
3. Guibas et al., "Adaptive Fourier Neural Operators," NeurIPS 2021 Workshop
4. Pathak et al., "FourCastNet: A Global Data-driven High-resolution Weather Forecasting Model," arXiv:2202.11214
5. Rao et al., "Global Filter Networks for Image Classification," NeurIPS 2021
6. Hu et al., "Squeeze-and-Excitation Networks," CVPR 2018
7. Woo et al., "CBAM: Convolutional Block Attention Module," ECCV 2018
8. Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas," MIDL 2018
9. Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration," CVPR 2022
10. Hassani et al., "Neighborhood Attention Transformer," CVPR 2023
11. Trockman & Kolter, "Patches Are All You Need?" ICLR 2022 Workshop
12. Yu et al., "MetaFormer is Actually What You Need for Vision," CVPR 2022
13. Ding et al., "Scaling Up Your Kernels to 31×31: Revisiting Large Kernel Design in CNNs (RepLKNet)," CVPR 2022
14. Liu et al., "More ConvNets in the 2020s: Scaling up Kernels Beyond 51×51 (SLaK)," ICLR 2023
15. Rao et al., "HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions," NeurIPS 2022
16. Valanarasu & Patel, "UNeXt: MLP-based Rapid Medical Image Segmentation Network," MICCAI 2022
17. Lin et al., "UNeXt-ILT: fast and global context-aware inverse lithography solution," J. Micro/Nanopatterning, Materials, and Metrology 24(1), 013201, January 2025
