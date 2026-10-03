# EVAL_PROTOCOL.md

Status: **DRAFT v0.1. Becomes FROZEN at the Week 4 gate.** After freeze, only typo fixes. Anything marked `[FILL @ W4]` is decided before freeze and then never changed.
Every change after freeze must be logged in `DECISIONS.md` as a deviation, with the reason, and reported in the results.

## 1. Purpose and claim hierarchy
We evaluate pretrained (and optionally finetuned) Sentinel-2 10 m → 2.5 m SR. Evidence is reported in separate tiers. **Tiers are never averaged or blended into one score.**

| Tier | Question | Data | Strength of claim |
|------|----------|------|-------------------|
| A | Does SR match real HR at 10→2.5 m? | opensr-test datasets (+ WorldStrat if usable) | Strongest, but non-India |
| B | Does it behave sensibly on India imagery at a proxy scale? | India Wald (40→10 m) | Supporting only |
| C | Does SR help a real task on India? | Building detection | Task evidence, noisy labels |
| D | Is uncertainty informative? | Tier A errors | Calibration evidence |
| E | Is SR radiometrically consistent? | All outputs | Sanity gate, not a result |

## 2. Methods compared (same list in every tier)
1. Raw 10 m (nearest-neighbour resample where a 2.5 m grid is needed)
2. Bicubic 2.5 m
3. SEN2SR pretrained (RGBN x4)
4. SEN2SR finetuned (only if it passed the keep gate)
5. LDSR-S2 (demo AOIs only, not in quantitative tiers unless it runs on Tier A)

## 3. Splits
- Never split by random tile.
- **Tier A:** use the benchmark's own scenes; no training involved. Report per dataset.
- **Tier B:** Wald on India AOIs. Finetune (if any) uses train AOIs only; val AOI chooses checkpoint.
- **Tier C:** leave-one-AOI-out (LOAO) across the building-dense AOIs `[FILL @ W4: list ≥3 AOIs]`.
  - For each held-out AOI, train on the others, select checkpoint/threshold on **validation blocks** drawn from the training AOIs (see below), report on the held-out AOI once.
  - Hyperparameters are fixed in advance (§6). No tuning on held-out AOIs.
  - Non-urban AOIs (flood/coastal/hill) are qualitative only.
- **Block structure:** divide each AOI into square blocks `[FILL @ W4: e.g. 2 km]`. Validation blocks are separated from training blocks by at least one block of buffer to limit spatial leakage.

## 4. Fixed processing choices
- Downsample kernel (SR→10 m, 10 m→40 m): 4x4 / 4x4 box average, documented and unit-tested.
- Reflectance scaling and baseline-offset handling: as implemented in `src/ingest/preprocess.py`, version-pinned.
- Grid: 2.5 m, aligned to 4x4 subpixels of the 10 m grid.
- Seeds: `[FILL @ W4: e.g. 3 fixed seeds]`. Report mean ± spread across seeds.
- Software versions pinned in `requirements.txt`; git SHA logged with every result.

## 5. Tier A: real-HR validation
- Use `opensr-test` datasets (NAIP, SPOT, others as available) with its published loaders and metrics. [Verify API]
- Report, per method and dataset: PSNR, SSIM, LPIPS, SAM, and the benchmark's consistency / hallucination-oriented measures (use its names if present; if not, implement and document a high-frequency-energy check and an edge-agreement check).
- HR reference and S2 differ in sensor, date and spectral response, so absolute scores are capped. Compare **methods on identical inputs**, not against external papers.
- Caveat to state in every figure caption: results are non-India.

## 6. Tier C: downstream building detection
- Model: U-Net, resnet18 encoder (pretrained), identical architecture and hyperparameters for every input type: `[FILL @ W4: lr, epochs, loss, augmentation, crop size]`.
- One model per input type, trained from the same initialization and seeds.
- Crop sizes cover the same ground area (e.g. 64 px at 10 m ↔ 256 px at 2.5 m).
- All predictions compared on the 2.5 m grid. 10 m model outputs are upsampled by nearest neighbour for scoring.
- Labels: building footprints rasterized at 2.5 m. Source `[FILL @ W4]`, license verified. Dataset and any confidence threshold fixed.
- **Alignment check before use:** estimate footprint-vs-image offset by edge cross-correlation. If offset exceeds ~1 LR pixel (10 m) in an AOI, either apply a documented global shift or drop that AOI from quantitative results. Decide once, record in `DECISIONS.md`.
- Metrics: F1, IoU, precision, recall at a fixed threshold chosen on validation blocks.

## 7. Statistics
- Unit of resampling: spatial blocks (not pixels, not tiles), since neighbouring pixels are correlated.
- Report 95% **block-bootstrap CIs** for every Tier A/B/C metric `[FILL @ W4: n_boot ≥ 1000]`.
- Compare methods with **paired** differences on the same blocks (SR − bicubic), with CI.
- No claim of improvement unless the paired CI excludes zero. Otherwise write "no significant difference".
- Report all AOIs and methods. No dropping of results after seeing them (exclusions only via the pre-declared alignment rule in §6).

## 8. Pre-registered interpretation (decided before seeing results)
| Outcome (Tier C, paired SR − bicubic) | We will say |
|--------------------------------------|-------------|
| CI > 0 | SR gives a measurable downstream gain on these AOIs |
| CI spans 0 | No significant gain detected; SR may still help visual analysis, not demonstrated for this task |
| CI < 0 | SR hurts this task; investigate misalignment, label noise, hallucination; report plainly |

Additionally report SR vs **raw 10 m** (the "why not just use 10 m directly" comparison). Any outcome is reportable. Do **not** change the protocol, labels, or model to improve a result.

## 9. Tier D: uncertainty calibration
- Error = per-pixel absolute error vs HR reference on Tier A, after the benchmark's alignment handling.
- Report: Spearman(uncertainty, error) per tile and on a subsample of pixels (note spatial autocorrelation); sparsification curve (remove highest-uncertainty pixels first, plot remaining error, compare to random and oracle) and area under it; reliability plot (binned uncertainty vs mean error).
- Gate: Spearman < 0.3 → try the next method (ensemble); if still low, report as is.
- Visual sanity: high on edges/fine texture, low on flat areas. Uniform noise = bug.

## 10. Tier E: consistency (sanity gate)
- Per-band RMSE, SAM, relative error between SR↓10 m and input, all methods, all AOIs.
- Expected near-zero for SEN2SR by design, so a pass is not evidence against hallucination. Any finetuned checkpoint must not regress here.

## 11. Failure gallery (mandatory)
- ≥20 tiles manually reviewed.
- Export 6 cases: tiles where SR F1 < bicubic F1, and tiles where SR shows structure absent from the 10 m input. Each with input, bicubic, SR, uncertainty, GT.
- Pick one for the demo and be able to explain it.

## 12. Freeze checklist (Week 4)
- [ ] All `[FILL @ W4]` items completed.
- [ ] AOI list, roles, block size, buffer, seeds fixed.
- [ ] Alignment check run on all candidate AOIs; inclusion decided.
- [ ] Interpretation table (§8) unchanged.
- [ ] `reproduce_all.py` regenerates Tier A–E tables from configs.
- [ ] Tag repo `protocol-freeze`.

## 13. Reporting template (one row per method per tier)
`method | dataset/AOI | metric | value | 95% CI | n blocks | seeds | git SHA`
Caption every table with the tier, the data origin (India / non-India), and the caveats from §5–§6.
