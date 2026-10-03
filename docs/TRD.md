# TRD — Technical Requirements

Status: v0.2 (solo, local 6 GB GPU, native Linux) | Supersedes v0.1
Verify every package name/version/API against its README before pinning. Nothing counts until you ran it.

## 1. Stack
- Python 3.11, PyTorch 2.x (CUDA build matching driver), conda/mamba env `sr`.
- Geo: rasterio, GDAL, pyproj, shapely, geopandas, rioxarray.
- STAC: pystac-client. Primary: Earth Search (AWS). Fallback: Planetary Computer. Cache everything to disk.
- **SR base: `sen2sr` package, NonReference RGBN x4 (Lite) variant.** Includes a tiled large-image predict function: use it. [Verify exact name/signature in README]
- Tiling/geo pipeline helper: `opensr-utils` is an option; evaluate in T1.1 before writing your own. [Verify]
- Diffusion (optional, pre-render only): `opensr-model` (LDSR-S2), 128→512 on 4-channel tensors, no geo IO.
- Benchmark: `opensr-test` for Tier A (real-HR) evaluation. [Verify API and metric names]
- Building det: `segmentation_models_pytorch`, U-Net with resnet18/34.
- App: **Streamlit** viewer + simple pipeline CLI. No FastAPI/TiTiler/React/Redis in v1.
- Tracking: CSV or MLflow. CI: lint + unit tests only (no GPU).
- OS: native Linux for all GPU work (see ExecutionPlan §0b).

## 2. Data
### Inputs
- S2 L2A B02,B03,B04,B08 @10 m + SCL. Output grid 2.5 m (4x), aligned to 4x4 subpixels of the 10 m grid.
- Reflectance: /10000 with the processing-baseline offset handled (baseline 04.00, Jan 2022). Verify against ESA docs, cite source in code.

### Sets
| Set | Use | Note |
|-----|-----|------|
| opensr-test datasets (NAIP, SPOT, others) | **Tier A**: real 10→2.5 m reference validation | Non-India. HR differs in sensor/date from S2, which caps absolute scores; compare methods on the same data |
| WorldStrat (S2 + SPOT 6/7) | Optional extra Tier A data | Check coverage and license (reference imagery is non-commercial) |
| India S2 tiles (self-built, 5 AOIs) | **Tier B** Wald (40→10 m), **Tier C** downstream, demo | Dry-season scenes |
| Building GT | Tier C | Open Buildings / OSM / Overture. Verify license and India coverage. Measure misalignment before trusting |
| Any HR India sample (Cartosat) | True-HR India check | Ask NTRO/college. Optional |

### AOI roles
- At least **3 building-dense AOIs** (e.g. 2 dense urban + 1 peri-urban) carry the quantitative Tier C results.
- Flood / coastal / hill AOIs: qualitative figures, consistency, uncertainty maps. Not used for F1 claims.
- Exact splits are defined in `EVAL_PROTOCOL.md` §3 and frozen at W4.

## 3. Models
### 3.1 SR (P0)
- Run pretrained SEN2SR first. No training.
- Finetune is **conditional** (Phase 3 gate: only if pretrained is not clearly better than bicubic on Tier B and Tier A). Cap ~10 GPU-hours. Keep only if it beats pretrained on val **and** consistency does not regress.
- Finetune pairs come from Wald (40→10 m), which trains the wrong scale. Treat as risky; validate on Tier A and visually.
- Loss if finetuning: L1 + low-weight perceptual + low-weight adversarial + consistency.
- Budget (6 GB, fp16, batch 1): LR tile 128 / overlap 16 for inference; finetune LR crop 64, batch 4, grad-accum 4 (fallback crop 48).

### 3.2 Uncertainty (P0)
- Primary: TTA (8 flip/rot90 variants, invert, per-pixel std).
- Ensemble: at most 2 checkpoints. Not a reliable method at K=2: report as a comparison, not as the main claim.
- Optional: LDSR-S2 multi-sample std (N=4–8) on demo AOIs, run on Kaggle/Colab.
- **Calibrate on Tier A (real HR), not Wald.** Report Spearman (per-tile and subsampled-pixel), sparsification curve, reliability plot.

### 3.3 Diffusion (P2, cut first)
- Local diffusion is cut. Pre-render on a free cloud GPU only if every earlier gate passed.

### 3.4 Downstream building detection (P0)
- Protocol fixed in `EVAL_PROTOCOL.md` §6: same architecture and hyperparameters per input type, one model per input, leave-one-AOI-out over building-dense AOIs.
- Inputs: raw 10 m (nearest), bicubic 2.5 m, SEN2SR 2.5 m, optional finetuned.

## 4. Metrics (`src/eval/`)
- Consistency (sanity gate): per-band RMSE, SAM, relative error; fixed 4x4 box-average downsample kernel, documented.
- Tier A/B reference metrics: PSNR, SSIM, LPIPS, SAM, plus hallucination/synthesis-oriented metrics from `opensr-test` if available [Verify]; otherwise a documented high-frequency-energy and edge-agreement check.
- Downstream: F1, IoU, precision, recall, each with block-bootstrap CI.
- Uncertainty: Spearman, sparsification error curve, reliability plot.
- Speed: s/tile, peak VRAM.

## 5. Compute and ops
- RTX 3050 6 GB, 16 GB RAM: windowed rasterio reads, one tile at a time, float32.
- Checkpoint every 500 steps with resume. Laptop sleep off, plugged in, performance mode.
- Log GPU-hours, config, git SHA for every run.

## 6. Repo layout
```
repo/
  docs/        PRD.md TRD.md EVAL_PROTOCOL.md DECISIONS.md HARDWARE.md
  configs/     aois.yaml, finetune.yaml, eval.yaml
  data/        not in git
  src/
    ingest/    fetch.py preprocess.py
    infer/     run_sen2sr.py bicubic.py stitch.py   (stitch only if sen2sr's tiling is insufficient)
    models/    uncertainty.py ldsr_infer.py
    train/     finetune_gan.py
    eval/      consistency.py tier_a.py wald.py gt.py building_det.py calibration.py
    app/       pipeline.py viewer.py
  scripts/     reproduce_all.py failure_cases.py
  tests/
```

## 7. Engineering rules
- Branch per feature; main always demo-able. One task = one commit.
- Config-driven, seeded, git SHA in every run.
- **Human runs the package's own example before the agent wraps it.** Agent must cite file/line for any library API used.
- Test AOIs/blocks untouched until the freeze phase.
- Every metric gets a unit test (identity SR → ~0 consistency error; synthetic seams; georeference alignment).

## 8. Known gotchas
- Reflectance offset change in L2A processing baseline.
- 2.5 m grid must align exactly to 10 m pixel corners (test it).
- Tile seams: overlap + feather; regression test on stitched output.
- Building GT misalignment vs imagery: measure by edge cross-correlation, then decide.
- Wald trains/tests a different scale than the deployment task: never headline it.
- SEN2SR is consistent by design, so consistency metrics won't separate methods.
