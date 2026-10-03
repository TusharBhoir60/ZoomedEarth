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
