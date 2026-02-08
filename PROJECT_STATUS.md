# LithographySimulator - Project Status & History

## Branch Info

| Field | Value |
|-------|-------|
| **Current branch** | `claude/shift-equivariant-unet-branch-1Nuq1` |
| **Parent branch** | `claude/shift-equivariant-unet-WkYqF` |
| **Base** | `master` |
| **Latest commit** | `f803510` — Fix ONNX export: replace GELU with tanh-approx GELUApprox layer |

---

## Project Overview

A lithography simulation toolkit built with TensorFlow/Keras. The project implements optical lithography image-formation models and a shift-equivariant U-Net architecture for lithography image segmentation, with an ongoing effort to integrate UNeXt-ILT (Tokenized MLP) blocks for improved global receptive fields.

### Key Components

| File | Purpose |
|------|---------|
| `shift_equivariant_unet.py` | Shift-equivariant U-Net model |
| `model_baseline.py` | Baseline model definition |
| `model_tokenized_mlp.py` | TokenizedMLP (UNeXt-ILT) model |
| `litho_sim_tf.py` | TensorFlow lithography simulator |
| `data_pipeline.py` | Training data pipeline |
| `train.py` | Training script |
| `export_onnx.py` | ONNX export and verification |
| `imageformation.py` | Image formation physics |
| `lightsource.py` | Light source modeling |
| `mask.py` | Mask representation |
| `pupil.py` | Pupil function modeling |

---

## Commit History (chronological, oldest first)

### 1. `01a18cb` — Add shift-equivariant U-Net for lithography image segmentation
- **Date:** 2026-02-06
- **Author:** Claude
- Initial implementation of the shift-equivariant U-Net architecture targeting lithography segmentation tasks.

### 2. `ebb412d` — Fix fundamental issues in shift-equivariant U-Net
- **Date:** 2026-02-06
- **Author:** Claude
- Addressed core bugs and correctness issues in the U-Net implementation.

### 3. `5925e86` — Add TF 2.12 compatibility via keras.ops shim
- **Date:** 2026-02-06
- **Author:** Claude
- Introduced a compatibility shim so the code works with TensorFlow 2.12's `keras.ops` API.

### 4. `ac07421` — Add documentation for shift-equivariant U-Net implementation
- **Date:** 2026-02-06
- **Author:** Claude
- Created `SHIFT_EQUIVARIANT_UNET.md` with architecture details and usage notes.

### 5. `491bd04` — Add TF lithography simulator, data pipeline, and training script
- **Date:** 2026-02-07
- **Author:** Claude
- Added `litho_sim_tf.py`, `data_pipeline.py`, and `train.py` for end-to-end training.

### 6. `8af063f` — Update .gitignore to exclude generated artifacts
- **Date:** 2026-02-07
- **Author:** Claude
- Ignored build/output artifacts from version control.

### 7. `d2304f3` — Add ONNX export/verification and register custom Keras layers
- **Date:** 2026-02-07
- **Author:** Claude
- Implemented `export_onnx.py` for model export; registered custom layers for serialization.

### 8. `eb93d6b` — Add ONNX and SavedModel artifacts to .gitignore
- **Date:** 2026-02-07
- **Author:** Claude
- Excluded exported model files from the repository.

### 9. `7153deb` — Fix CUDA availability checks
- **Date:** 2026-02-07
- **Author:** tangtangtang19871987-wq
- Corrected GPU detection logic for environments without CUDA.

### 10. `730a76e` — Merge pull request #1
- **Date:** 2026-02-07
- **Author:** tangtangtang19871987-wq
- Merged `codex/review-code-in-specific-branch` into the main development line.

### 11. `0717a4e` — Add roadmap report for global receptive field enhancements
- **Date:** 2026-02-07
- **Author:** Claude
- Created `GLOBAL_RECEPTIVE_FIELD_ROADMAP.md` outlining strategies for improving receptive fields.

### 12. `704dea3` — Add UNeXt-ILT (MICCAI 2022 + JMM Jan 2025) to roadmap
- **Date:** 2026-02-07
- **Author:** Claude
- Expanded the roadmap with references to the UNeXt-ILT paper and architecture.

### 13. `c4d5b28` — Update roadmap with UNeXt-ILT architecture details from paper diagrams
- **Date:** 2026-02-07
- **Author:** Claude
- Added detailed architecture diagrams and design notes from the UNeXt-ILT paper.

### 14. `cb75f60` — Implement shift-equivariant TokenizedMLP block (UNeXt-ILT adapted)
- **Date:** 2026-02-07
- **Author:** Claude
- Core implementation of the TokenizedMLP block adapted from UNeXt-ILT for shift-equivariant processing.

### 15. `01030e1` — Separate model definitions into model_baseline.py and model_tokenized_mlp.py
- **Date:** 2026-02-08
- **Author:** Claude
- Refactored model code into two files for clarity: baseline U-Net and TokenizedMLP variant.

### 16. `f803510` — Fix ONNX export: replace GELU with tanh-approx GELUApprox layer
- **Date:** 2026-02-08
- **Author:** Claude
- Fixed ONNX compatibility by replacing the standard GELU activation with a tanh-approximated custom layer.

---

## Current Status

- Shift-equivariant U-Net: **implemented and documented**
- TokenizedMLP (UNeXt-ILT) block: **implemented**
- TF lithography simulator: **integrated**
- Data pipeline & training: **functional**
- ONNX export: **working** (with GELUApprox fix)
- Global receptive field roadmap: **drafted**

## Next Steps

- Validate TokenizedMLP model accuracy against baseline
- End-to-end training benchmarks on lithography datasets
- Expand test coverage and CI integration
- Further ONNX/deployment optimization
