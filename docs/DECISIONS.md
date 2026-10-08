# DECISIONS — SIH26142 NTRO Super-Resolution Project

## D001 — SEN2SRLite as primary SR model (2026-10-03)

**Context**: T1.1 source verification.  
**Decision**: Use pretrained `SEN2SRLite` (`NonReference_RGBN_x4`) as the primary baseline model.  
**Reason**: Official ESAOpenSR pretrained weights. MIT license (code), CC0 1.0 (weights). Does not require training.  

---

## D002 — sr environment is NV_6939 symlink (2026-10-03)

**Context**: T1.1 environment setup. The `sr` conda environment specified in the project rules does not yet exist as a standalone Python 3.11 environment.  
**Decision**: Created `sr` as a symlink to `NV_6939` (Python 3.10.21, torch 2.3.0+cu121) which has verified CUDA support on the RTX 3050.  
**Action Required**: Create a proper `conda create -n sr python=3.11` environment at the next opportunity and verify CUDA support before removing the symlink. Document in HARDWARE.md.  

---

## D003 — fp16 DISABLED for SEN2SRLite (2026-10-03)

**Context**: T1.1 precision profiling on RTX 3050 (CUDA 12.1, PyTorch 2.3.0+cu121).  
**Decision**: fp16 autocast is **disabled** for SEN2SRLite inference. fp32 is the required operating mode.  
**Evidence**:  
- `ComplexHalf` arithmetic in `HardConstraint.forward()` (`sen2sr/models/tricks.py:223`) is experimental and produces NaN in band 4 (B08/NIR).  
- cuDNN reports `CUDNN_STATUS_NOT_SUPPORTED` for the sub-pixel convolution in `cnn.py:211`.  
- Measured: fp16 latency is **20× slower** than fp32 (216ms vs 10.6ms) with less VRAM saving than expected.  
**Consequence**: fp32 peak VRAM = 163.67 MB for a 128×128 tile — well within 6 GB budget.  

---

## D004 — Band permutation contract (2026-10-03)

**Context**: Official SEN2SRLite `load.py` visualization in `display_results()` uses `lr[0, [2, 1, 0]]` for RGB, confirming the model operates in RGBN = [B04, B03, B02, B08] order.  
**Decision**: Explicit bidirectional permutation `[B02,B03,B04,B08] ↔ [B04,B03,B02,B08]` is applied at the wrapper boundary. Permutation index `[2,1,0,3]` is its own inverse.  

---

## D005 — Input size handling strategy (2026-10-03)

**Context**: The HardConstraint low-pass mask is fixed at `512×512` (Fourier domain), requiring exactly 128×128 LR input tiles. Inputs smaller than 128 result in a shape mismatch RuntimeError.  
**Decision**:  
- Inputs == 128×128: direct model pass.  
- Inputs < 128: replicate-pad to 128, infer, crop output to `H×4, W×4`.  
- Inputs > 128: use official `sen2sr.predict_large()` (128-tile chunking with 32px overlap).  

---

## D008 — Reflectance Offset Handling & Guard (2026-10-05)

**Context**: TASK RAD-1 (Fix reflectance offset handling for Earth Search L2A).
**Decision**: 
1. `resolve_boa_offset()` strictly determines the BOA offset from `earthsearch:boa_offset_applied` and `s2:processing_baseline`, rather than blindly trusting the legacy extracted offset.
2. We enforce a guard where if >5% of valid pixels in any spectral band are negative after conversion, preparation aborts.
**Evidence**:
- T2.3 cached `-1000` without recording its source, and it equals the STAC `raster:bands` offset divided by scale. (Hypothesis until `download.py` is reviewed).
- The branch `applied=False`, PB ≥ `04.00` → `-1000` is unverified on real data.
- The 5% negative check guard is one-sided (it only catches a too-large subtraction, but cannot catch an offset left in the data biased +0.1).
**Consequence**:
- The invalid scene `S2B_43RFM_20230203_0_L2A` processed outputs (including ~9.5k tiles and stitched scene) have been moved to quarantine. The official-sample benchmark and T1.1 FP16 results remain unaffected.

---

## D009 — T5.1 GAN Training Stability (2026-10-08)

**Context**: T5.1 GAN finetuning scaffold preparation and stability validation.
**Decision**: 
1. **NaN Rejection**: The `WaldDataset` iterator performs rejection sampling on 64x64 crops. Any crop containing non-finite values (i.e. NoData boundaries) is instantly discarded. NaNs are *never* imputed or silently replaced with zeros.
2. **FP32 Training**: Training remains unconditionally FP32 (AMP disabled). Previous D003 analysis of the `HardConstraint` and CUDA kernel behavior prohibits stable half-precision.
3. **Scientific Limitation**: Wald 40m→10m is strictly a training proxy distribution. It does not mathematically prove deployment-scale 10m→2.5m superiority over bicubic, but is the sole authorized finetuning mechanism under Gate 3.

---

## D010 — T5.2 Gate 5 CLEARED & Checkpoint Retained (2026-10-09)

**Context**: T5.2 validation sweep on held-out geographic data (Pune validation AOI `aoi_pune_peri_urban`, scene `S2B_43QDA_20230114_0_L2A`) to determine if a fine-tuned checkpoint should be retained. Gate 5 requires the checkpoint to beat pretrained SEN2SR on the Wald set AND ensure consistency is not worse.
**Decision**: KEEP fine-tuned checkpoint 200 (`outputs/t5_2_finetune/checkpoint_step_200.pt`). Gate 5 is CLEARED.
**Evidence**:
- **Pune Wald metrics**:
  - Global RMSE: Step 200 = 0.016835 vs Pretrained = 0.017265 vs Bicubic = 0.017197 (Step 200 wins)
  - Global PSNR: Step 200 = 35.518757 vs Pretrained = 35.301703 vs Bicubic = 35.333060 (Step 200 wins)
  - SAM: Step 200 = 0.061868 vs Pretrained = 0.062298 vs Bicubic = 0.060394 (Step 200 wins against pretrained)
- **Consistency**:
  - RMSE_B02: Step 200 = 0.001691 vs Pretrained = 0.002602 (Step 200 strictly better)
  - SAM_RAD: Step 200 = 0.010940 vs Pretrained = 0.016669 (Step 200 strictly better)
  - REL_ERR_B02: Step 200 = 0.033421 vs Pretrained = 0.051417 (Step 200 strictly better)
**Consequence**: Checkpoint 200 is confirmed as the formally selected model moving forward into Phase 6. The distinction between training-scene evaluation and geographically held-out Pune validation results must remain preserved.
