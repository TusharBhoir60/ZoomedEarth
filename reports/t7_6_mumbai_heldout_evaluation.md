# ZoomedEarth — T7.6 Mumbai Held-Out Evaluation Report

**Date Generated:** October 10, 2026

## 1. Frozen Checkpoint Identity
- **File:** `checkpoints/t7_3_real_run_v1/best_model.pt`
- **SHA-256 Hash:** `8145eb85b1a2c8f4d30ee42105e552a0396d956fda98a63f60779d914f82dc57`
- **Epoch:** `1`
- **Pune Selection Score:** `0.9238`

## 2. Execution Paths & Functions
This final evaluation was dispatched exclusively utilizing the rigidly frozen, deterministic project implementation modules:
- **Inference Wrapper:** `src.train.train_detector.generate_validation_predictions`
- **Metric Extraction:** `src.eval.tier_c.score_tier_c`

### Active Evaluation Contracts:
1. **Input Rasters (Independent Passes):**
   - SEN2SR Source: `data/outputs/sen2sr/mosaic_aoi_mumbai_urban/SEN2SR_scene.tif`
   - Bicubic Source: `data/outputs/sen2sr/mosaic_aoi_mumbai_urban/Bicubic_scene.tif`
2. **Canonical Labels:** `data/processed/eval_labels/aoi_mumbai_urban_buildings_2_5m.tif`
3. **Canonical Cloud Mask:** `data/processed/sentinel2/mosaic_aoi_mumbai_urban/cloud_mask.tif`
4. **Frozen Budget File:** `data/processed/eval_labels/pu_area_budget.json`
5. **Output Predictions Target:** `data/processed/predictions/t7_6_mumbai_eval/`

## 3. Results: Tier C Area-Budgeted Recall
The metrics below represent the recovery fractions of open positive labels achieved under the restricted $K$ Area-Budget. 

- **Mumbai SEN2SR Tier C Recall:** **`0.6190`** (61.90%)
- **Mumbai Bicubic Tier C Recall:** **`0.5362`** (53.62%)
- **Absolute Modality Difference:** `+0.0828` (+8.28 percentage points directly attributable to SEN2SR upsampling performance over baseline Bicubic)

## 4. Candidate Pool and Label Semantics
These scores represent model selection behavior against Positive-Unlabeled (PU) conditions derived explicitly from the pre-calculated Delhi distributions. 
- **Frozen Area-Budget Scalar:** `0.1973425628900519`
- **Candidate Pool (N):** `57,468,096` pixels *(Exactly identical across SEN2SR and Bicubic evaluations due to strict `cloud_mask.tif` overlap validating inference artifact bounds)*
- **Eligible Known-Positive Target Pixels:** `5,828,816` pixels (Pixels representing valid building footprint ground-truth target coordinates)
- **Selected Area-Budget Candidates (K):** `11,340,901` pixels (Pixels allowed to be flagged as positive predictions under the PU scalar limit)

## 5. Output-Grid Verification and Resource Measurements
The final inference grids perfectly overlap the established standard canonical labels exactly preserving validation integrity.
- **Dimensions:** `6432 × 8936`
- **Geospatial Format:** `float32`, `EPSG:32643`
- **Transform Check:** Explicit $4\times$ matching 2.5m spacing scaling.

**Resource Impact:**
- **Peak RAM:** `15.2%` (~2.4 GB).
- CPU overhead securely bounded inside Python environments performing sequential Numpy index arrays mapping.

## 6. Known Limitations
**Semantic Limitations under the PU Protocol:** 
These values reflect strictly *held-out Open Buildings annotation recovery*. Because Open Buildings contains inherent systematic positive-unlabeled bias (failing to natively annotate all structures equally across sub-geographies), the reported recall fractions systematically underestimate absolute real-world spatial completeness. This test evaluates the robustness of semantic extraction relative to a budgeted scalar and serves essentially to validate relative generalizability over baseline, **not** absolute universal precision, precision-recall curve topologies, or $F_1$ scoring models.

## 7. Statement of Integrity
**The held-out results derived from this AOI (Mumbai) were categorically never utilized during any phase of training, checkpoint selection logic, threshold tuning, or configuration alignment.** All weights utilized within the neural architectures evaluated originated cleanly from isolated upstream dependencies decoupled fundamentally from Mumbai testing grids.

***

### Artifact Manifest & Hashes (SHA-256)
- **Checkpoint (`best_model.pt`):** `8145eb85b1a2c8f4d30ee42105e552a0396d956fda98a63f60779d914f82dc57`
- **SEN2SR Input Raster:** `427974c3cd4295f8f67dcbb65f41fac3cfe3acdc9e2b53e4782c8890aa4b032a`
- **Bicubic Input Raster:** `654ebeb2ea2317ded65e14b6520b4c4486a18cd7fe5cd9825f1e3688a5723c7c`
- **SEN2SR Output Predictions:** `613839d3a2d5013415677c6210db053195dea8919ff445c43715399f32737ac8`
- **Bicubic Output Predictions:** `a3bc29435500a10920acb6a852ae81b812d3fdc78a214d776d0318ce18472c86`
