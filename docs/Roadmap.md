# Roadmap

Start: Mon Sep 28, 2026 | Finale: Dec 2026 (exact date [Guessing] — confirm; plan assumes worst case early Dec)
Rule: **feature freeze Nov 23. Demo freeze Nov 30.** Buffer after = insurance.

## Phases
| Wk | Dates | Phase | Gate |
|----|-------|-------|------|
| 1 | Sep 28 | Setup | Repo, envs, GPU access confirmed, OpenSR runs on 1 tile |
| 2 | Oct 5 | Data | 5 India AOIs ingested, cloud-masked, tiled |
| 3 | Oct 12 | Baseline | SEN2SR pretrained output on India AOIs + bicubic + metrics v0 |
| 4 | Oct 19 | Eval protocol | Wald set built, building GT aligned, protocol FROZEN |
| 5 | Oct 26 | GAN finetune 1 | Consistency loss in, first India-finetuned ckpt |
| 6 | Nov 2 | Uncertainty + API | Uncertainty raster works; API + queue up |
| 7 | Nov 9 | Downstream + UI | Building det results on 4 input types; viewer v1 |
| 8 | Nov 16 | Diffusion + integration | LDSR-S2 inference on demo AOIs; end-to-end pipeline |
| 9 | Nov 23 | Test-set eval + FREEZE | Test AOIs run once; numbers locked |
| 10 | Nov 30 | Demo hardening | Pre-rendered AOIs, offline mode, video backup |
| 11 | Dec 7 | Rehearsal + docs | 3 full dry runs; submission pack |
| 12 | Dec 14 | Buffer | Fixes only |

## Per-role plan
### R1 Lead
- W1: lock scope, confirm PS text/data, GPU access, finale date.
- W2-8: weekly Monday planning, Friday gate review. Kill scope creep.
- W9+: demo script, Q&A prep, submission.

### R2 Data
- W1: STAC client, pick primary provider.
- W2: 5 AOIs (dense urban, peri-urban, flood, coastal, hill). Dry-season scenes.
- W3-4: building GT, alignment check, Wald set, geography splits.
- W5+: India finetune tile sets, cache demo AOIs.

### R3 GAN
- W1-3: SEN2SR inference reproduced; fp16; tile+stitch.
- W5: finetune + consistency loss.
- W6-7: ensemble ckpts (K=3), speed tuning <60s/tile.
- W8+: freeze best ckpt.

### R4 Diffusion + Uncertainty
- W2-3: LDSR-S2 inference reproduced.
- W6: uncertainty rasters (GAN ensemble/TTA + diffusion std).
- W7: calibration vs error, reliability plot.
- W8: pre-render demo AOIs.

### R5 Eval / Downstream
- W1-3: metrics lib (consistency, SAM, PSNR/SSIM/LPIPS) with unit tests.
- W4: freeze building-det protocol.
- W5-7: train seg model, run 4 input types.
- W8-9: failure-case gallery, result tables, test-set run.

### R6 Backend / Frontend / DevOps
- W1: Docker base, CI.
- W3-6: FastAPI, queue, TiTiler.
- W6-8: viewer (swipe, uncertainty overlay, metrics panel).
- W9-10: offline mode, one-command startup, video backup.

## Decision gates
- **W3:** if pretrained SEN2SR on India already visually fine → spend less on finetune, more on eval + uncertainty.
- **W4:** if no usable building GT alignment → fall back to OSM in urban AOIs only.
- **W6:** if uncertainty Spearman < 0.3 → switch method (ensemble > TTA), else report honestly.
- **W8:** diffusion not working → drop to pre-render-only or cut. GAN + uncertainty + downstream is a complete story alone.
- **W9:** results worse than bicubic on downstream → report honestly, analyze why. Do not tune on test.

## Cut list (in order)
1. Diffusion finetune (inference only)
2. Live diffusion in demo (pre-render)
3. Batch CLI
4. Fancy UI (keep swipe + uncertainty toggle)
5. Extra AOIs beyond 3

## Rituals
- Mon 15 min: plan. Fri 30 min: demo of what works, gate check.
- Daily async: 3-line update (done / next / blocked).
- Every experiment: config + git SHA + GPU-hours logged.

## Definition of done
- One command reproduces all metrics.
- Demo runs offline from cached AOIs.
- 3 rehearsals passed by Dec 7.
- Failure case prepared and explained.
