# Project Rules — Sentinel-2 Super-Resolution + Trust Harness (SIH26142, NTRO)

## 1. PROJECT IDENTITY

Project:
Sentinel-2 10m → 2.5m super-resolution and trust/validation harness for Indian AOIs.

## 2. SOLO DEVELOPMENT

* This is a one-person project.
* The human developer is the sole project owner and decision maker.
* Antigravity is an implementation assistant, not another developer.
* Never assume another developer exists.
* Never create tasks that depend on another developer/team.
* Never divide work into human R1–R6 roles.
* Optimize implementation for one developer and one local 6 GB GPU.

## 3. DOCUMENT AUTHORITY

Before planning any implementation task, read:

* docs/PRD.md
* docs/TRD.md

Read docs/EVAL_PROTOCOL.md for evaluation-related work.

Use docs/PRD.md as the product source of truth.

Never silently resolve a conflict between documents.
If a conflict materially affects implementation, report it to the human developer.

## 4. PYTHON / ENVIRONMENT

* Python 3.11
* Environment name: sr
* Native Linux
* Do not use WSL
* Do not create Windows-specific implementation instructions
* Use configuration-driven execution
* YAML configuration belongs under configs/

## 5. API / PACKAGE VERIFICATION

Never invent library APIs.

Before using an unfamiliar package API:

* Read its official README/documentation/source.
* Verify the installed/versioned API.
* Record the verified version.
* Pin the version where appropriate.
* If the API cannot be verified, stop and ask the human developer rather than guessing.

## 6. DATA SAFETY

* Never modify raw files under data/ unless explicitly instructed.
* Never delete project data automatically.
* Never overwrite raw datasets.
* Do not fabricate datasets, metrics, benchmark results, timings, VRAM numbers, or model performance.
* All reported numerical results must come from an actual execution.

## 7. GEOSPATIAL RULES

Input Sentinel-2 bands:

* B02
* B03
* B04
* B08

Input resolution:

10 m

Target resolution:

2.5 m

Target scale:

4× spatial super-resolution

Output must preserve correct:

* CRS
* affine transform
* spatial extent
* pixel alignment
* band ordering
* geospatial metadata

The 2.5 m grid must be aligned to the 4×4 subpixels of the original 10 m grid.

## 8. DATA SPLITS

Use geography/AOI-based splits.

Never use random tile splitting when an AOI/geographic split is required.

Test AOIs must remain untouched until the evaluation freeze specified in the project documents.

## 9. MODEL SCOPE

Primary baseline:

Pretrained SEN2SR.

Do not invent a new super-resolution architecture.

GAN/SEN2SR fine-tuning is conditional on the project's evaluation gate.

Diffusion is optional and inference/pre-render only if the appropriate later gate passes.

Do not implement live diffusion.

Do not implement diffusion fine-tuning.

## 10. V1 ARCHITECTURE

Keep the implementation lightweight and solo-maintainable.

KEEP:

* Sentinel-2 ingestion
* preprocessing
* SEN2SR baseline
* bicubic baseline
* consistency evaluation
* uncertainty estimation
* building-detection evaluation
* failure analysis
* Streamlit viewer
* offline demo
* reproducible evaluation pipeline

CUT FROM V1:

* TiTiler
* React frontend
* Redis
* Celery/RQ
* complex backend services
* live diffusion
* diffusion fine-tuning
* unnecessary batch infrastructure

Do not reintroduce these components unless the human developer explicitly changes the PRD.

## 11. OUTPUTS / REPRODUCIBILITY

Experiment outputs must contain enough information to reproduce the result.

Where applicable, save:

* configuration
* experiment/run ID
* git SHA
* software versions
* timing
* GPU information
* relevant metrics

Use:

outputs/<run_id>/

for generated experiment artifacts unless the task specifies another documented location.

## 12. TESTING

Every implementation module must have appropriate pytest coverage.

Tests should verify, where applicable:

* numerical behavior
* tensor shapes
* band ordering
* CRS
* affine transforms
* pixel sizes
* 4× spatial scaling
* geospatial alignment
* edge cases

Run relevant tests before declaring a task complete.

## 13. DEVELOPMENT DISCIPLINE

* One task at a time.
* One logical change per commit.
* Keep changes small.
* Do not perform unrelated refactoring.
* Do not add speculative infrastructure.
* Do not change project architecture without explicit approval.
* Do not silently expand scope.

## 14. GPU CONSTRAINTS

The local GPU has 6 GB VRAM.

Treat VRAM as a hard engineering constraint.

Prefer:

* fp16/AMP where supported
* small tiles
* batch size 1 for inference where necessary
* windowed raster processing
* memory-conscious processing
* one tile at a time

Never assume that a configuration fits in VRAM without testing it.

## 15. TEST / EVALUATION INTEGRITY

Never tune against the frozen test AOIs.

Never change evaluation criteria after seeing test results.

Never exclude poor results without documenting the reason.

Never fabricate a successful experiment.

If an experiment fails:

* preserve the failure information
* report the actual error/result
* identify the likely cause
* propose the smallest next debugging step

## 16. HUMAN VERIFICATION

For every task, provide:

* files changed
* commands executed
* tests executed
* actual results
* known limitations
* exact verification step for the human developer

Never declare a task complete merely because code was written.
