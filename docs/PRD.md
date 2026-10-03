# PRD — Sentinel-2 Super-Resolution + Trust Harness (SIH26142, NTRO)

Status: v0.2 (solo-aligned, supersedes v0.1) | Owner: you | Finale: Dec 2026 (confirm exact date)
Single source of truth. If another doc conflicts with this one, this one wins. Conflicts get fixed, not tolerated.

## 1. Problem
- Sentinel-2 RGB+NIR is 10 m. Too coarse for building-level urban and disaster analysis.
- PS SIH26142 asks for DL super-resolution to <4 m (we target 2.5 m, 4x) that preserves geospatial + spectral consistency, manages uncertainty explicitly, and is **validated against high-resolution references**.
- Core risk: SR models invent plausible detail. Analysts need evidence of when to trust the output.

## 2. Positioning (what makes this project different)
Many teams will ship "SR model + uncertainty map + dashboard". Ours is the **validation and trust harness for Sentinel-2 SR over India**:
- Real-HR validation (not only Wald).
- Frozen eval protocol, geography-aware splits, confidence intervals.
- Calibrated uncertainty, with the calibration evidence shown.
- Honest downstream test, including "SR gives no significant gain" as a valid outcome.
- A prepared failure case.

The SR model itself is a pretrained open model (SEN2SR). We do not claim a new architecture.

## 3. Goals
- **G1** 4x SR of S2 L2A B02/B03/B04/B08 → 2.5 m, georeferenced COG. Base: pretrained SEN2SR (RGBN x4 variant). Finetune only if the Phase 3 gate says it is needed.
- **G2** Consistency as a **sanity gate** (SR↓10 m ≈ input). Not a headline result: the base model is consistent by design.
- **G3** Per-pixel uncertainty raster with every SR output, plus calibration evidence (rank correlation, sparsification curve, reliability plot).
- **G4** Validation in three tiers, reported separately and never blended (see `EVAL_PROTOCOL.md`):
  - A: real 10 m → 2.5 m HR pairs (non-India).
  - B: India Wald protocol (40 m → 10 m).
  - C: downstream building detection on India AOIs.
- **G5** Offline demo on Indian AOIs from cached outputs.
- **G6 (stretch, droppable)** LDSR-S2 diffusion pre-render for demo AOIs only.

## 4. Non-goals (v1)
- No live diffusion, no diffusion finetuning.
- No TiTiler / React stack (Streamlit viewer).
- No SWIR / 20 m / 60 m bands, no SAR fusion, no multi-temporal SR (all listed under Future Scope).
- No new foundation model trained from scratch.
- No batch CLI beyond the one-command pipeline.

## 5. Users
- SIH/NTRO judges: want a clear demo, honest numbers, a trust story.
- Analyst persona: loads an AOI, gets SR + uncertainty, exports GeoTIFF.

## 6. Functional requirements
| ID | Requirement | Priority |
|----|-------------|----------|
| FR1 | AOI (bbox) + date range → fetch S2 L2A (B02,B03,B04,B08,SCL) | P0 |
| FR2 | Cloud mask + best-scene selection | P0 |
| FR3 | SEN2SR inference → 2.5 m GeoTIFF (COG), grid aligned to 4x4 subpixels of the 10 m grid | P0 |
| FR4 | Uncertainty raster co-registered with SR | P0 |
| FR5 | Streamlit viewer: 10 m / bicubic / SR / uncertainty, swipe, offline | P0 |
| FR6 | Tier A / B / C evaluation per `EVAL_PROTOCOL.md` → metrics tables with CIs | P0 |
| FR7 | Consistency sanity report | P0 |
| FR8 | Failure-case gallery | P0 |
| FR9 | Export GeoTIFF + PNG + metrics JSON | P1 |
| FR10 | LDSR-S2 pre-rendered demo AOI | P2 (cut first) |

## 7. Success criteria
Numeric targets below are **hypotheses**, not pass/fail gates, until baselines exist (W3). Replace with measured values at protocol freeze (W4).

| Area | Hypothesis | Notes |
|------|-----------|-------|
| Consistency | SAM(SR↓10 m, input) small; per-band RMSE small | Sanity gate. Must not regress after any finetune. |
| Tier A (real HR) | SR beats bicubic on reference-based metrics | Report which metrics, with CIs. |
| Tier C (downstream) | SR ≥ bicubic on building F1/IoU | **"No significant difference" is an acceptable, reportable result.** Interpretation rules are fixed in `EVAL_PROTOCOL.md` §8. |
| Uncertainty | Positive rank correlation with error on Tier A; sparsification curve beats random | Gate: Spearman < 0.3 → switch method or report honestly. |
| Hallucination | Failure gallery produced and explained | Manual review of ≥20 tiles. |
| Latency | SEN2SR < 60 s per 10 km x 10 km tile on the local GPU | Measure in T1.3. Likely easy with the Lite model. |

## 8. Acceptance criteria
- End-to-end run on ≥5 Indian AOIs (urban, peri-urban, flood, coastal, hill).
- One command regenerates all tables and figures from configs.
- Demo works with WiFi off.
- 3 dry runs on the demo machine by Dec 7.
- Failure case prepared and explained.

## 9. Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| No HR reference for India | High | Tier A on opensr-test (NAIP/SPOT) + WorldStrat if coverage fits; ask NTRO/college for any Cartosat sample; label results "non-India" |
| Wald tests wrong scale (40→10 m) | High | Never headline it. Use as India-specific supporting evidence only |
| Downstream test underpowered (few test AOIs, noisy labels) | High | Block bootstrap CIs, leave-one-AOI-out over building-dense AOIs, GT alignment check, qualitative-only for non-urban AOIs |
| Hallucinated structures | High | Uncertainty + failure gallery + Tier A hallucination-oriented metrics |
| Pretrained weights trained on US NAIP → domain gap | High | Measure it (Tier B, C), report honestly |
| Agent writes unverified model/API code | High | Human runs package's own example first; agent wraps only after |
| Solo + 6 GB laptop GPU | High | Pretrained-first, finetune conditional and capped (~10 GPU-h), no local diffusion |
| Docs drift out of sync | Med | This PRD is the source of truth; edit it before changing scope |
| Monsoon cloud | Med | Dry-season scenes; flood AOI uses a clear post-event scene (stated openly as a limitation) |

## 10. Open questions
- Does NTRO or the college provide any HR reference (Cartosat etc.)?
- Finale date and demo format (live vs video)?
- Confirm SEN2SR weight license on the model card; confirm building-GT license for submission.
- WorldStrat: any Indian coverage?

## 11. Future scope (deck material)
1. Multi-temporal SR (fuse several dates): the real route to added information.
2. SAR (Sentinel-1) fusion for monsoon and flood scenes, where optical is blind.
3. 20 m bands via SEN2SR for spectral indices at 2.5 m.
4. Few-shot adaptation on Indian HR (Cartosat) if available.
5. Publish the Tier A/B/C harness as an open SR trust benchmark for India.
