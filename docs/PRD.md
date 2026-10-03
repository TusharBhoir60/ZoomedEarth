# PRD — Sentinel-2 Super-Resolution (SIH 2026, NTRO)

Status: DRAFT v0.1 | Owner: Team Lead | Finale: Dec 2026

## 1. Problem
- Sentinel-2 = 10m. Too coarse for building-level disaster + urban analysis.
- Need DL super-resolution: 10m → <4m (target 2.5m, 4x) over Indian AOIs.
- Risk: SR models hallucinate. Analysts need trust signal → uncertainty map mandatory.

## 2. Goals
- G1: 4x SR of Sentinel-2 L2A RGB+NIR (B02,B03,B04,B08) → 2.5m.
- G2: Spectral consistency: SR output downsampled to 10m ≈ input.
- G3: Per-pixel uncertainty map with every SR output.
- G4: Prove value on downstream task (building detection), not just PSNR/SSIM.
- G5: Live demo on Indian AOIs, GAN fast path (<60s per 10km x 10km tile, target).
- G6 (stretch): Diffusion high-fidelity path, uncertainty via multi-sample variance.

## 3. Non-goals
- No 3D globe viz.
- No SWIR/20m/60m band SR (out of scope v1).
- No new foundation model training from scratch.
- No SAR fusion.

## 4. Users
- NTRO/SIH judges: want clear demo + metrics + trust story.
- Analyst persona: loads AOI, gets SR + uncertainty, exports GeoTIFF.

## 5. Functional Requirements
| ID | Requirement | Priority |
|----|-------------|----------|
| FR1 | Input AOI (bbox/GeoJSON) + date range → fetch S2 L2A | P0 |
| FR2 | Cloud mask + best-scene selection | P0 |
| FR3 | GAN SR inference → 2.5m GeoTIFF (COG) | P0 |
| FR4 | Uncertainty raster co-registered with SR | P0 |
| FR5 | Side-by-side viewer: 10m / bicubic / SR / uncertainty | P0 |
| FR6 | Building detection run on 10m, bicubic, SR → metrics table | P0 |
| FR7 | Consistency report (SR↓10m vs input) | P0 |
| FR8 | Diffusion mode toggle (slow, high-fi) | P1 |
| FR9 | Export GeoTIFF + PNG + metrics JSON | P1 |
| FR10 | Batch AOI run via CLI | P2 |

## 6. Success Metrics (baselines TBD in W3; targets are [Guessing] until then)
- Consistency: SAM(SR↓10m, input) < 1°; per-band RMSE < 1% reflectance.
- Downstream: building F1/IoU ≥ +10% relative vs bicubic baseline.
- Uncertainty: Spearman(uncertainty, abs error) > 0.5 on held-out Wald-protocol set.
- Hallucination check: no new-structure artifacts in 20-tile manual review.
- Latency: GAN < 60s/tile on 1 GPU.

## 7. Acceptance Criteria
- Runs end-to-end on ≥5 Indian AOIs (mix: dense urban, peri-urban, flood, coastal, hill).
- All metrics reproducible from one command.
- Demo works offline (cached tiles) if network dies.

## 8. Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Pretrained weights trained on US NAIP → domain gap on India | High | Wald-protocol eval; fine-tune on India S2 pairs; report gap honestly |
| No HR reference for India | High | Wald protocol; downstream task; ask NTRO/college for any Cartosat/HR sample |
| Hallucinated buildings | High | Uncertainty map; consistency constraint; show failure cases |
| GPU access limits | Med | Fixed compute budget per week; small-tile training; fp16 |
| Diffusion too slow for demo | Med | GAN is primary; diffusion pre-rendered |
| 6-person coordination | Med | Roles fixed in Roadmap.md; weekly gates |
| Cloud cover on monsoon scenes | Med | Choose dry-season AOIs; flood AOI via post-event clear scene |

## 9. Open Questions
- Does NTRO PS supply any data/HR reference? (check problem statement text)
- Finale date + demo format (live vs video)?
- Exact GPU spec at college cluster?
- Building GT license OK for submission? (verify Open Buildings / MS footprints / OSM terms)
