# System Design

Status: DRAFT v0.1

## 1. Architecture

```
 [User: AOI + dates]
        |
   [Web UI: MapLibre] <----- COG tiles ------ [TiTiler]
        |                                          ^
     REST/JSON                                     |
        v                                          |
   [FastAPI]  --enqueue--> [Job Queue (RQ/Celery+Redis)]
        |                          |
        |                    [GPU Worker]
        |                     1 ingest
        |                     2 tile + normalize
        |                     3 SR (GAN | Diffusion)
        |                     4 uncertainty
        |                     5 stitch + write COG
        |                     6 consistency + downstream eval
        v                          |
   [Metadata DB: SQLite/Postgres]  v
                          [Artifact store: /data/outputs/<job_id>/]
```

## 2. Components
| Component | Job | Owner |
|-----------|-----|-------|
| Ingest | STAC query, download B02/B03/B04/B08/SCL, best-scene pick, cloud mask | R2 |
| Tiler | 128px LR tiles, 16px overlap, normalize | R2 |
| SR-GAN | fast inference, fp16 | R3 |
| SR-Diff | N-sample inference | R4 |
| Uncertainty | ensemble/TTA/multi-sample std → raster | R4 |
| Stitcher | feather blend → 2.5m COG, georef exact | R3 |
| Eval | consistency, ref metrics, building det | R5 |
| API | jobs, status, download | R6 |
| Viewer | 4-way compare, swipe, uncertainty overlay, metrics panel | R6 |

## 3. Data flow (one job)
1. POST /jobs {bbox, date_range, mode: gan|diffusion}
2. Ingest → cached scene in `/data/raw/<scene_id>/`
3. Tile → `/data/tiles/<job_id>/`
4. Model → SR tiles + uncertainty tiles
5. Stitch → `sr_2p5m.tif`, `uncertainty.tif` (COG, EPSG same as source UTM)
6. Eval → `metrics.json` (consistency always; downstream if building GT exists)
7. Viewer loads COGs via TiTiler

## 4. API (v0)
| Method | Path | Purpose |
|--------|------|---------|
| POST | /jobs | create job |
| GET | /jobs/{id} | status + progress |
| GET | /jobs/{id}/metrics | metrics.json |
| GET | /jobs/{id}/download?type=sr\|unc\|png | files |
| GET | /aois | preset AOIs (demo) |

Job states: `queued → ingesting → tiling → inferring → stitching → evaluating → done | failed`

## 5. Output artifacts
```
outputs/<job_id>/
  sr_2p5m.tif          # COG, RGBN, float32 or uint16 scaled
  uncertainty.tif      # COG, 1 band
  bicubic_2p5m.tif     # baseline
  metrics.json
  consistency_report.png
  buildings_{10m,bicubic,gan,diff}.geojson   # if run
  config.yaml + git_sha.txt
```

## 6. Deployment
- Dev: Docker compose = api + redis + worker (GPU) + titiler + web.
- Demo: single college GPU box, all services local. **Cached demo AOIs pre-rendered** → no live network dependency.
- Diffusion: pre-rendered for demo AOIs; live only if latency OK.
- Fallback: recorded video + static COGs.

## 7. Failure handling
| Failure | Response |
|---------|----------|
| STAC down | fall back to 2nd provider / local cache |
| Cloud > threshold | reject scene, pick next, else warn |
| GPU OOM | halve tile batch, retry |
| Seam artifact | increase overlap; regression test on stitched sample |
| Model timeout | return job failed + partial tiles, no silent garbage |

## 8. Demo script (5 min)
1. Pick AOI (urban) → 10m view. Zoom in: blurry.
2. Run GAN → 2.5m in <60s. Swipe compare.
3. Toggle uncertainty: high on fine texture/edges.
4. Building detection: 10m vs SR, F1 table.
5. Show consistency report + one **failure case** (credibility).
6. Diffusion pre-render: sharper, slower, uncertainty from samples.

## 9. Design decisions
- D1: GAN primary. Reason: latency + demo reliability. Trade-off: less fidelity than diffusion.
- D2: Consistency constraint in training AND reported in eval. Reason: defends against hallucination claims.
- D3: Downstream metric is headline; PSNR/SSIM secondary.
- D4: Geography-split eval. Reason: leakage.
- D5: Pre-render demo AOIs. Reason: live demos die.
