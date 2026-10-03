# TRD — Technical Requirements

Status: DRAFT v0.1 | Verify all package names/versions before pinning.

## 1. Stack
- Python 3.11, PyTorch 2.x (CUDA), Lightning optional
- Geo: rasterio, GDAL, pyproj, shapely, geopandas, xarray/rioxarray
- Data access: STAC (pystac-client). Sources: Copernicus Data Space, Earth Search (AWS), MS Planetary Computer. [Likely] any one works; pick one primary, one fallback.
- SR base: OpenSR (SEN2SR = fast/CNN-GAN path, LDSR-S2 = latent diffusion path). [Likely] verify pretrained weight license + band support in repo README.
- Building det: SegFormer or U-Net (segmentation_models_pytorch)
- API: FastAPI + Uvicorn
- Tile serving: TiTiler (COG)
- Frontend: MapLibre GL JS + React (or plain JS if time short)
- Tracking: MLflow or W&B; data versioning: DVC
- CI: GitHub Actions (lint + unit tests only; no GPU)
- Container: Docker (GPU: nvidia-container-toolkit)

## 2. Data
### Inputs
- Sentinel-2 L2A, bands B02,B03,B04,B08 @10m, SCL for cloud mask.
- Output grid: 2.5m (4x). Meets <4m.

### Training / eval sets
| Set | Use | Note |
|-----|-----|------|
| SEN2NAIP (OpenSR) | pretrain/finetune reference | US NAIP. Domain gap for India. [Likely] |
| India S2 tiles (self-built) | Wald-protocol eval + finetune | 10m→ downsample 40m; predict 10m; compare to real 10m |
| Building GT | downstream eval | Google Open Buildings / MS Building Footprints / OSM. Verify license. |
| Any HR India sample (Cartosat etc.) | true HR check | Ask NTRO/college. Optional. |

### Splits
- Split by geography (AOI-level), never random tiles. Leakage kills credibility.
- Val: 2 AOIs. Test: 3 AOIs. Never touched until W9.

## 3. Models
### 3.1 GAN fast path (P0)
- Start: SEN2SR pretrained. Baseline first, no training.
- Finetune on India tiles. Generator: RRDB/SwinIR-style or SEN2SR arch. Discriminator: PatchGAN / UNet-D.
- Loss: L1 + perceptual (low weight) + adversarial (low weight) + **consistency loss** (SR↓10m vs input).
- Hard-constraint option: low-freq of output forced = input (SEN2SR-style). [Likely]
- fp16, tile 128x128 LR → 512x512 HR, overlap 16px + feather blend.

### 3.2 Diffusion stretch (P1)
- LDSR-S2 pretrained (latent diffusion). Inference only first; finetune only if GPU budget allows.
- N samples (start N=8) → mean = SR, std = uncertainty.

### 3.3 Uncertainty
- GAN: ensemble of K finetuned checkpoints (K=3-5) OR TTA variance (flip/rot). [Guessing] TTA-only likely underestimates; ensemble better.
- Diffusion: multi-sample std.
- Calibrate: rank-correlate with abs error on Wald set. Report Spearman + reliability plot.

### 3.4 Downstream: building detection
- Train ONE segmentation model on **one** fixed input type (e.g. bicubic-upsampled 2.5m) OR train per-input; pick protocol in W4 and freeze.
- Compare inputs: raw 10m, bicubic 2.5m, GAN 2.5m, Diffusion 2.5m.
- Metrics: F1, IoU, precision, recall. Report per AOI type.

## 4. Metrics (implement in `eval/`)
- Consistency: SAM, per-band RMSE between SR↓10m and input. Downsample kernel fixed + documented.
- Reference-based (Wald set only): PSNR, SSIM, LPIPS, SAM.
- No-reference: NIQE/BRISQUE optional. Don't headline them.
- Downstream: F1/IoU.
- Uncertainty: Spearman, ECE-style reliability.
- Speed: s/tile, VRAM.

## 5. Compute
- Local/college GPUs. Need: GPU model, VRAM, queue system (SLURM?). Confirm W1.
- Budget rule: every experiment logs GPU-hours. Weekly cap set by Lead.
- Checkpoints to shared storage. No checkpoint on a laptop only.

## 6. Repo layout
```
repo/
  data/          # DVC-tracked, no raw in git
  src/
    ingest/      # STAC fetch, cloud mask, tiling
    models/      # gan/, diffusion/, uncertainty/
    train/
    infer/       # tile→stitch→COG
    eval/        # consistency, ref metrics, downstream
    api/         # FastAPI
  web/           # frontend
  configs/       # yaml, one per experiment
  notebooks/
  docs/          # PRD.md TRD.md SystemDesign.md Roadmap.md
  tests/
```

## 7. Engineering rules
- Branch per feature, PR + 1 review. main always demo-able.
- Config-driven runs. No hardcoded paths.
- Seed everything. Log git SHA in every run.
- Feature freeze: see Roadmap.md.

## 8. Team roles (6)
| # | Role | Owns |
|---|------|------|
| R1 | Lead / Integrator | scope, PRD, integration, demo script, final submission |
| R2 | Data Engineer | STAC ingest, cloud mask, AOI selection, tiling, India dataset, building GT |
| R3 | GAN Model Dev | SEN2SR baseline, finetune, consistency loss, fp16 inference |
| R4 | Diffusion + Uncertainty Dev | LDSR-S2 inference, ensembles, calibration |
| R5 | Evaluation / Downstream | metrics suite, building-detection model, result tables, failure-case analysis |
| R6 | Backend + Frontend / DevOps | FastAPI, TiTiler, viewer, Docker, demo hardening |

## 9. Known technical gotchas
- [Certain] SR of 20m/60m bands not in scope; only RGBN hits <4m here.
- [Likely] Reflectance scaling mismatch (L2A /10000, offset changes after Jan 2022 baseline 04.00) → normalize in ingest. Verify.
- [Likely] Tile seams → overlap + feather blend, test on stitched output.
- [Likely] Georeferencing: 2.5m grid must align exactly to 10m pixel corners (4x4 subpixels). Unit-test this.
- [Guessing] Registration offset vs building GT footprints may cost F1. Measure, don't assume.
