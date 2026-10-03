# Execution Plan — Solo + Antigravity

Setup: solo, all 6 roles | Windows | local NVIDIA GPU | Antigravity IDE
Companion docs: PRD.md, TRD.md, SystemDesign.md, Roadmap.md → put in `docs/`

Legend: **YOU** = human only. **AG** = paste prompt to Antigravity agent. **GATE** = must pass before next phase.

---

## 0. Truth first
- Solo = 6 roles in ~11 weeks. Roadmap.md assumed 6 people. Not all of it fits.
- **Solo cut line (locked):**
  - KEEP: ingest, SEN2SR baseline, consistency metrics, uncertainty, building-det eval, simple viewer, offline demo.
  - CUT: diffusion finetune, live diffusion, TiTiler, React frontend, batch CLI.
  - Diffusion = inference-only, pre-rendered, only if W8 gate passes.
  - Viewer = Streamlit (or plain MapLibre HTML). Not the SystemDesign.md stack. [Likely] Saves ~1 week.
- Local GPU = VRAM is the bottleneck. Check in T0.2. Everything below assumes small tiles + fp16 + batch 1.
- Agents invent package APIs and metrics. Rule: nothing counts until you ran it and saw the number.

---

## 0b. Hardware profile — OVERRIDES other sections
Machine: i5 12th gen, 16 GB RAM, RTX 3050 6 GB VRAM, dual boot (Windows + Linux).

### OS decision
- Do all GPU work in **native Linux**. Skip WSL. Skip T0.1 WSL install + T0.4 WSL test.
  - WSL2 default eats up to ~50% RAM → 8 GB gone. Antigravity + training + WSL on 16 GB = swap thrash.
  - Native Linux = no agent-terminal-to-WSL problem.
- Linux driver: Ubuntu → `ubuntu-drivers devices` then `sudo ubuntu-drivers install`, reboot, `nvidia-smi`. Secure Boot on → enroll MOK or disable.
- Hybrid graphics laptop: confirm training uses the NVIDIA GPU (`nvidia-smi` shows python process).
- Windows/WSL = fallback only. If WSL: `%UserProfile%\.wslconfig` → `[wsl2]` / `memory=10GB` / `swap=8GB`.
- Need ≥60 GB free on Linux partition: `df -h`.
- Plug into power, performance mode, cooling pad. [Likely] laptop 3050 throttles on long runs.
- Close Chrome etc. during training. Antigravity is heavy.

### Budgets (VRAM 6 GB) — [Guessing] until measured in T1.3
| Job | Setting |
|-----|---------|
| SEN2SR inference | fp16, LR tile 128, overlap 16, batch 1 |
| GAN finetune | LR crop 64 (HR 256), batch 4, grad accum 4, fp16. OOM → LR crop 48, freeze early layers |
| Seg model (T7.1) | U-Net resnet18/34, 256 crop, batch 8, AMP |
| Uncertainty | TTA primary. Ensemble max 2 ckpts (not 3-5) |
| Diffusion (LDSR-S2) | test 1 tiny tile locally only. Real pre-render on Kaggle/Colab free GPU (T4-class, quota limited). Cut if it fails |
| RAM (16 GB) | windowed rasterio reads, one tile at a time, float32 not float64 |

### Changed cut line
- Diffusion locally = **cut**. Diffusion = optional Kaggle/Colab pre-render only.
- GAN finetune = conditional (T3.3) and capped at ~10 GPU-hours. Pretrained SEN2SR already fine → skip.
- Overnight runs: checkpoint every 500 steps, resume support. Laptop sleep off.
- T1.3 becomes the truth: if SEN2SR inference doesn't fit at tile 64, stop and re-plan before Phase 2.

---

## 1. How the loop works
1. Open Antigravity → Agent Manager → new task scoped to repo folder.
2. Paste the task prompt. Agent plans → **read the plan artifact before approving**.
3. Agent runs, produces artifacts (diffs, logs, tests).
4. YOU verify the "Done when" line yourself. Commit if passes: `git commit -m "T<id>: <what>"`.
5. Fail? Comment on the artifact or re-prompt with the error text. Don't restart from scratch.

Settings [Likely, verify in UI]:
- Mode: **Agent-assisted** (not fully agent-driven).
- Terminal command policy: **request review**. Never auto-run `rm`, `del`, `git push --force`, or anything touching data/.
- Model: use strongest available for model/eval code; cheaper/faster for boilerplate. Switch if agent loops.
- Parallel agents OK only for independent tasks (e.g. tests + docs). Never two agents editing the same file.

---

## 2. Phase 0 — Environment (Day 1–2)

### T0.1 YOU — Install base
- [ ] Install/update NVIDIA **Windows** driver (not a Linux driver inside WSL).
- [ ] PowerShell admin: `wsl --install -d Ubuntu` → reboot → create user.
- [ ] Install Antigravity (Windows installer, sign in with Google).
- [ ] Install Git, Windows Terminal (optional).
- Done when: in Ubuntu shell `nvidia-smi` shows your GPU.

### T0.2 YOU — Record hardware
Run in WSL, paste output into `docs/HARDWARE.md`:
```bash
nvidia-smi
free -h
df -h ~
```
- Write down: GPU model, VRAM, RAM, free disk.
- Rule of thumb [Guessing]: VRAM < 8 GB → tiles 64–96 px LR, no diffusion batch >1. Free disk < 100 GB → limit to ~5 AOIs, delete raw after tiling.

### T0.3 YOU — Python env (in WSL)
```bash
# Miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
# restart shell
mamba create -n sr python=3.11 -y
mamba activate sr
mamba install -c conda-forge rasterio gdal geopandas shapely pyproj xarray rioxarray pystac-client -y
# torch: pick CUDA build matching your driver from pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
- Done when: prints `True` + GPU name.
- Fallback if WSL is painful: native Windows Miniforge, same `mamba` line, torch from pytorch.org Windows wheel.

### T0.4 YOU — Open repo in Antigravity
```bash
mkdir -p ~/sr-s2 && cd ~/sr-s2 && git init
mkdir -p docs src/{ingest,models,train,infer,eval,app} configs tests data notebooks
cp <your md files> docs/
```
- Open `~/sr-s2` in Antigravity. [Guessing] WSL folder access works via WSL remote or `\\wsl$\Ubuntu\home\<user>\sr-s2`. Test that the **agent terminal** can run `nvidia-smi` inside WSL.
- If agent terminal is Windows-only and can't reach WSL → switch whole project to native Windows (T0.3 fallback). Don't fight this for >1 hour.
- Done when: agent can run `python -c "import torch; print(torch.cuda.is_available())"` and gets `True`.

### T0.5 AG — Project rules
Add as workspace rule (Customizations / rules; path [Likely] `.agent/rules/project.md`):
```
Project: Sentinel-2 10m -> 2.5m super-resolution for Indian AOIs.
Always read docs/TRD.md and docs/PRD.md before planning.
Rules:
- Python 3.11, env "sr". Config-driven (yaml in configs/), no hardcoded paths.
- Never invent library APIs. Read the package source/README first. Pin versions in requirements.txt.
- Never fabricate results. Report only numbers from actual runs. Save outputs to outputs/<run_id>/ with config + git SHA.
- Do not touch data/ raw files. Do not delete anything without asking.
- Split by AOI, never random tiles. Test AOIs are untouchable until Phase 9.
- Write a pytest for every module. Run it before finishing.
- Keep changes small. One task = one commit.
- Only bands B02,B03,B04,B08 at 10m. Output grid 2.5m, aligned to 4x4 subpixels of the 10m grid.
```
- Done when: rule visible in Antigravity; new agent quotes it back when asked.

**GATE 0:** GPU works in agent terminal, repo + docs + rules in place.

---

## 3. Phase 1 — OpenSR baseline (Day 3–5)

### T1.1 AG — Reproduce SEN2SR inference
Prompt:
```
Read docs/TRD.md. Find the OpenSR project (SEN2SR / LDSR-S2) on GitHub and read its README and install docs first. Install what's needed in env "sr" and pin versions in requirements.txt. Write src/infer/run_sen2sr.py that takes a 4-band (B02,B03,B04,B08) Sentinel-2 GeoTIFF and writes a 4x upscaled GeoTIFF with correct georeferencing. Use fp16 if supported. Use the sample data the repo provides first. Do not invent any API — cite the file/line you used. Print peak VRAM and seconds per tile.
```
- Done when: sample tile → SR output opens in QGIS/rasterio, sharper than input, coordinates align.
- YOU: confirm pretrained weights license + which bands the model supports. Note in `docs/DECISIONS.md`.

### T1.2 AG — Bicubic baseline
```
Write src/infer/bicubic.py producing a 4x bicubic upscale with identical grid/georef as the SR output. Add pytest checking output transform = input transform with pixel size /4 and origin unchanged.
```
- Done when: pytest passes.

### T1.3 YOU — Speed/VRAM check
- Log s/tile + VRAM from T1.1 into `docs/HARDWARE.md`.
- If OOM at small tiles: tell AG to add tiled inference with overlap (T2.3 handles it).

**GATE 1:** one tile through SEN2SR + bicubic, georeferenced, timed.

---

## 4. Phase 2 — Data ingest (Week 2)

### T2.1 YOU — Pick 5 AOIs
Pick ~10km x 10km boxes. Suggested [Guessing on cloud-free availability]:
| # | Type | Idea |
|---|------|------|
| 1 | Dense urban | Pune / Mumbai / Delhi core |
| 2 | Peri-urban | Hyderabad / Bengaluru fringe |
| 3 | Flood | Assam (Brahmaputra) or a known recent flood |
| 4 | Coastal | Chennai / Mumbai coast |
| 5 | Hill | Dehradun / Shimla area |
- Save bboxes in `configs/aois.yaml` (`name, bbox[minx,miny,maxx,maxy], type, split`).
- Split: 3 train, 1 val, 1 test now. Rename later only before touching test.
- Dry-season scenes (Nov–Mar) for urban/hill; post-event clear scene for flood.

### T2.2 AG — STAC ingest
```
Write src/ingest/fetch.py using pystac-client. Primary: Earth Search AWS (https://earth-search.aws.element84.com/v1, collection sentinel-2-l2a). Fallback: Microsoft Planetary Computer STAC (needs planetary-computer signing). For each AOI in configs/aois.yaml and date range, find scenes, rank by low cloud cover, download B02,B03,B04,B08 and SCL clipped to the AOI into data/raw/<aoi>/<scene_id>/. Log scene ID, date, cloud %, CRS. Cache; skip if exists. Handle STAC errors by falling back. Add tests with a tiny mocked response.
```
- Done when: 5 AOIs each have a cached scene + metadata JSON.
- YOU: open one in QGIS. Looks right? Cloud-free enough?

### T2.3 AG — Cloud mask, normalization, tiling
```
Write src/ingest/preprocess.py: (1) mask cloud/shadow using SCL, (2) convert to reflectance handling the L2A offset change (check scene processing baseline; verify against ESA docs, cite source), (3) tile into LR tiles with configurable size and 16px overlap, keeping georef per tile, (4) write manifest CSV (tile_id, aoi, split, bounds, cloud_frac). Add tests: tiles reassemble to original; 2.5m grid aligns to 4x4 subpixels of 10m.
```
- Done when: tests pass, manifest exists, you eyeball 5 tiles.

### T2.4 AG — Stitcher
```
Write src/infer/stitch.py that blends overlapping SR tiles with feather weights into a single COG with correct georef. Test on synthetic tiles: no seams (max abs diff at overlaps < 1e-6 for constant input).
```

**GATE 2:** end-to-end script: AOI → tiles → SR → stitched COG. No metrics yet.

---

## 5. Phase 3 — Metrics + baselines (Week 3)

### T3.1 AG — Consistency metrics
```
Write src/eval/consistency.py. Downsample SR (2.5m) to 10m with a fixed, documented kernel (area/box average over 4x4), compare with the input: per-band RMSE, SAM in degrees, relative error. Return dict + save consistency_report.png (scatter + error map). pytest: identity SR (nearest upsample of input) gives ~0 error.
```
- Done when: bicubic and SEN2SR both scored on all 5 AOIs. Numbers in `outputs/baseline_v0/metrics.json`.

### T3.2 AG — Wald-protocol set
```
Write src/eval/wald.py: from real 10m tiles, downsample to 40m (fixed kernel), run each SR method 4x back to 10m, compute PSNR, SSIM, LPIPS, SAM vs the real 10m. Save per-AOI metrics. Use train/val/test from manifest; do not run on test AOI yet.
```
- Done when: table for bicubic vs SEN2SR on train+val AOIs.
- YOU: this is your India-specific evidence. Screenshot + save.

### T3.3 YOU — Baseline review (decision)
- SEN2SR already clearly beats bicubic on India Wald set? → finetune is optional, spend time on uncertainty + downstream.
- Not clearly? → finetune (Phase 5) is required.
- Write verdict in `docs/DECISIONS.md`.

**GATE 3:** baseline table exists. Metrics reproducible with one command.

---

## 6. Phase 4 — Eval protocol freeze (Week 4)

### T4.1 YOU — Building ground truth
- Pick source: Google Open Buildings / Microsoft footprints / OSM (Overture bundles some). **Check license + India coverage.** [Likely] Google Open Buildings covers South Asia.
- Download only for your 5 AOIs into `data/gt/`.

### T4.2 AG — Rasterize + align GT
```
Write src/eval/gt.py: clip building footprints to each AOI, rasterize at 2.5m on the SR grid (binary mask), save COG. Add an alignment-check script that overlays footprint edges on the 10m image and reports pixel offset estimate via cross-correlation of edge maps. Output PNGs for review.
```
- YOU: view overlays. Offset > ~1 LR pixel (10m)? Note it. Urban-only eval fallback = OSM in urban AOIs.

### T4.3 YOU — Freeze protocol
Write in `docs/EVAL_PROTOCOL.md` and **do not change after**:
- Inputs compared: raw 10m (resampled nearest), bicubic 2.5m, SEN2SR 2.5m, (later) fine-tuned, (later) diffusion.
- Seg model: same architecture + same hyperparams per input, trained on train AOIs, tuned on val AOI, reported on test AOI once.
- Metrics: F1, IoU, precision, recall.
- Downsample kernel, seeds, split — fixed.

**GATE 4:** protocol frozen + GT aligned.

---

## 7. Phase 5 — GAN finetune (Week 5, conditional on T3.3)

### T5.1 AG — Training scaffold
```
Read docs/TRD.md section 3.1 and the OpenSR/SEN2SR training code. Write src/train/finetune_gan.py: finetune from pretrained on train-AOI tiles using LR->HR pairs built by Wald protocol (40m->10m) and, if available, real pairs. Loss = L1 + low-weight perceptual + low-weight adversarial + consistency loss (SR downsampled to LR vs LR). fp16, batch size fits VRAM, gradient accumulation, checkpoint every N steps, resume, MLflow or CSV logging. Config in configs/finetune.yaml. Run 200 steps smoke test and report loss curves.
```
- Done when: smoke test runs, loss decreases, no NaNs.
- Note [Guessing]: Wald-protocol training teaches 4x on 40m->10m, not 10m->2.5m. Scale gap. Validate visually + downstream. Report honestly.

### T5.2 YOU — Real run
- Launch overnight. Log GPU-hours. Keep best val ckpt only.
- Done when: fine-tuned ckpt beats pretrained on val Wald set **and** consistency not worse. Else keep pretrained.

**GATE 5:** keep/discard decision recorded.

---

## 8. Phase 6 — Uncertainty (Week 6)

### T6.1 AG — GAN uncertainty
```
Write src/models/uncertainty.py with two methods: (a) TTA: 8 flips/rot90 variants, invert, per-pixel std; (b) checkpoint ensemble if >1 checkpoints exist. Output uncertainty COG (same grid as SR). Add a calibration script src/eval/calibration.py: on Wald val set compute Spearman between uncertainty and abs error, and a reliability plot (binned uncertainty vs mean error).
```
- Done when: Spearman + plot for TTA, saved.
- Gate rule: Spearman < 0.3 → try ensemble (train 2 more seeds if time) or report honestly. [Guessing] TTA-only often underestimates.

### T6.2 YOU — Look at maps
- Uncertainty high on edges/roofs/fine texture, low on flat fields? Plausible. Uniform noise = bug.

**GATE 6:** uncertainty raster + calibration number.

---

## 9. Phase 7 — Downstream building detection (Week 7)

### T7.1 AG — Seg training
```
Write src/eval/building_det.py using segmentation_models_pytorch (U-Net, pretrained encoder). Follow docs/EVAL_PROTOCOL.md exactly. Train one model per input type (10m-nearest, bicubic, SEN2SR [, finetuned]) on train AOIs, select on val AOI, save predictions. Report F1, IoU, precision, recall per AOI. Fix seeds. Save everything under outputs/downstream_v1/.
```
- Done when: table with all input types on val.
- YOU: SR not beating bicubic? Don't tune the protocol. Investigate (misalignment, GT quality, overfit). If still not better, report honestly + why.

### T7.2 AG — Failure gallery
```
Write scripts/failure_cases.py: find tiles where SR building F1 is worse than bicubic and where SR shows structures absent in the 10m input (large local difference after downsampling). Export a 6-image gallery with input, bicubic, SR, uncertainty, GT.
```
- Done when: 1 failure case you can explain in the demo.

**GATE 7:** downstream table + failure gallery.

---

## 10. Phase 8 — Serving/UI (Week 8)

### T8.1 AG — One-command pipeline
```
Write src/app/pipeline.py: given AOI name/bbox + date range, run ingest -> tile -> SR (mode gan) -> uncertainty -> stitch -> consistency -> save outputs/<job>/ with sr.tif, uncertainty.tif, bicubic.tif, metrics.json. CLI: python -m src.app.pipeline --aoi pune. Job status printed. Graceful errors.
```

### T8.2 AG — Viewer
```
Build a Streamlit app (src/app/viewer.py) that loads a finished job and shows: swipe/side-by-side of 10m, bicubic, SR; uncertainty overlay toggle; metrics panel (consistency + downstream F1 table); download buttons. Works fully offline from cached outputs. Add a "Demo AOIs" dropdown.
```
- Done when: full demo flow works with WiFi off.
- Upgrade path only if time: TiTiler + MapLibre.

**GATE 8:** end-to-end demo path on 2 AOIs, offline.

---

## 11. Phase 8b — Diffusion stretch (Week 8–9, conditional)
Only if Gates 1–8 pass and VRAM allows.

### T9.1 AG — LDSR-S2 inference
```
Read OpenSR LDSR-S2 docs and source. Write src/models/ldsr_infer.py to run inference on one tile with N samples (start N=4, then 8 if time allows), output mean SR and per-pixel std. Report VRAM and seconds/tile. Do not invent APIs; cite files.
```
- Done when: 1 demo AOI rendered, saved as pre-rendered COG.
- Fails/OOM/too slow → **cut**. Say "future work" in the deck. Not a failure.

---

## 12. Phase 9 — Test run + freeze (Week 9)
- [ ] YOU: run test AOI **once** through downstream + consistency. Save. No retuning after.
- [ ] YOU: tag repo `v1.0-freeze`. Feature freeze.
- [ ] AG: `python scripts/reproduce_all.py` regenerates every table/figure from configs. Done when: same numbers within seed noise.

## 13. Phase 10 — Demo + docs (Weeks 10–11)
- [ ] AG: README with setup + run + results. Architecture diagram source (Mermaid).
- [ ] YOU: 5-min demo script per SystemDesign.md §8. Include failure case.
- [ ] YOU: pre-render all demo AOIs. Record backup video.
- [ ] YOU: 3 full dry runs on the demo machine, WiFi off.
- [ ] YOU: prepare answers: hallucination, why GAN, why not just PSNR, domain gap, license/data source.

---

## 14. Prompt template (reuse)
```
CONTEXT: read docs/TRD.md + docs/EVAL_PROTOCOL.md.
TASK: <one thing>.
FILES: create/modify only <paths>.
CONSTRAINTS: no invented APIs (cite source), configs in yaml, fp16, VRAM <= <X> GB.
DELIVER: code + pytest + short run log with real numbers.
DONE WHEN: <checkable condition>.
```

## 15. Debug prompts
- Error: "Here is the full traceback: <paste>. Explain the root cause first, then propose the smallest fix. Don't rewrite other files."
- OOM: "Reduce peak VRAM without changing outputs: smaller tile, fp16, batch 1, gradient checkpointing. Report before/after VRAM."
- Suspicious result: "This metric looks too good. List 3 possible leakage/bugs and write a test for each."

## 16. Weekly YOU checklist
- [ ] Every new number: opened output, looked at it.
- [ ] Test AOI untouched.
- [ ] Commit per task. `main` runs.
- [ ] GPU-hours logged.
- [ ] Decisions in `docs/DECISIONS.md`.
- [ ] Behind schedule? Apply cut list in §0. Don't add scope.

## 17. Timeline (solo)
| Wk | Phase | Gate |
|----|-------|------|
| 1 | 0 + 1 | env + baseline tile |
| 2 | 2 | ingest + stitch |
| 3 | 3 | baseline metrics |
| 4 | 4 | protocol frozen |
| 5 | 5 | finetune decision |
| 6 | 6 | uncertainty |
| 7 | 7 | downstream + failures |
| 8 | 8 (+8b) | offline demo |
| 9 | 9 | freeze, test run |
| 10–11 | 10 | rehearsals |
| 12 | buffer | fixes only |
