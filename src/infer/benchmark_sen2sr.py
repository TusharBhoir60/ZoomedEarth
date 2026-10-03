"""
T1.3 — SEN2SR Speed & VRAM Verification Benchmark.

Project: Sentinel-2 Super-Resolution + Trust Harness (SIH26142, NTRO)
Task:    T1.3 — Measurement only. Do NOT modify model, weights, or T1.1 behavior.

Uses the exact T1.1 inference path:
  - load_sen2sr_lite_model() from src.infer.run_sen2sr
  - Direct model(input_tensor) call for 128x128 inputs
  - fp32 precision (T1.1 mandated — fp16 is UNSAFE on RTX 3050/CUDA 12.1)
  - Band order: [B04, B03, B02, B08] fed to model (permuted from repo convention)

CUDA timing uses explicit torch.cuda.synchronize() around each measurement.
Warm-up runs are excluded from reported statistics.
"""

import json
import pathlib
import statistics
import sys
import time
from typing import Dict, Any

import numpy as np
import torch

# ── Ensure src.* imports work when script is called directly ──────────────────
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.infer.run_sen2sr import (
    load_sen2sr_lite_model,
    permute_bgrn_to_rgbn,
)

# ── Configuration ─────────────────────────────────────────────────────────────
WEIGHTS_DIR = "models/SEN2SRLite"
WARMUP_RUNS = 2
MEASURED_RUNS = 5
PRECISION = "fp32"   # T1.1 mandated — see run_summary.json precision_findings
SCALE = 4.0

# T1.1 validated input: [B02, B03, B04, B08] float32 normalized reflectance
# These values mirror the T1.1 test fixture (test_run_sen2sr_e2e).
REPO_BAND_VALUES = [0.12, 0.15, 0.18, 0.35]  # B02, B03, B04, B08
INPUT_H, INPUT_W = 128, 128   # The validated T1.1 tile size


def make_input_tensor(device: torch.device) -> torch.Tensor:
    """
    Create the T1.1-convention input tensor for benchmarking.

    Repository order: [B02, B03, B04, B08] → permuted to [B04, B03, B02, B08]
    for the model, exactly as run_sen2sr.py does.
    Shape fed to model: [1, 4, 128, 128], float32.
    """
    bgrn = np.zeros((4, INPUT_H, INPUT_W), dtype=np.float32)
    for i, v in enumerate(REPO_BAND_VALUES):
        bgrn[i, :, :] = v

    # Apply T1.1's band permutation before model input
    permuted = permute_bgrn_to_rgbn(bgrn)  # → [B04, B03, B02, B08]
    tensor = torch.from_numpy(permuted).unsqueeze(0).to(device)  # [1, 4, H, W]
    return tensor


def get_memory_mb(device: torch.device) -> Dict[str, float]:
    """Collect current and peak CUDA memory statistics in MB."""
    mb = 1024 ** 2
    props = torch.cuda.get_device_properties(device)
    return {
        "gpu_name": props.name,
        "gpu_total_memory_mb": round(props.total_memory / mb, 2),
        "allocated_mb": round(torch.cuda.memory_allocated(device) / mb, 2),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / mb, 2),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / mb, 2),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / mb, 2),
    }


def assess_large_tile_feasibility() -> Dict[str, Any]:
    """
    Assess feasibility of 1000×1000 LR inference based on predict_large source
    and T1.1 validated behavior. Does NOT run a 1000×1000 inference.

    Source-level observations from sen2sr.utils.predict_large:
      1. Tiles processed sequentially in a single loop.
      2. Each tile result is moved to CPU immediately (result.detach().cpu()),
         so only one 128×128 tile (and its activations) is on GPU at a time.
      3. The full output tensor is pre-allocated on CPU at index=0 with shape:
           (C, X.shape[1] * res_n, X.shape[1] * res_n)
         For 1000×1000 input: (4, 4000, 4000) float32 ≈ 244 MB RAM.
         This is within the 16 GB RAM budget.
      4. Non-square behavior: the output tensor uses X.shape[1] for BOTH
         spatial dimensions (observed during T1.1 validation — not documented
         as a bug in the sen2sr source). For a square 1000×1000 input this
         does not produce incorrect dimensions.

    Conclusion: Based on source inspection, GPU VRAM usage during predict_large
    is bounded by one tile at a time (~163 MB peak at 128×128 fp32), not the
    full image. This suggests VRAM should not be the limiting factor for a
    1000×1000 input. However:
      - This has NOT been measured on actual 1000×1000 input on this hardware.
      - Intermediate activations in the HardConstraint FFT path scale with
        tile size (fixed at 512×512 Fourier space for 128×128 tiles) — tile
        size is fixed, so this is constant.
      - The feasibility of a full 1000×1000 inference via predict_large
        therefore CANNOT be confirmed without an actual benchmark run.
    Recommendation: A dedicated large-tile benchmark task (T1.x) is required
    to verify end-to-end behavior on a 1000×1000 (or real Sentinel-2 scene)
    input before treating it as production-ready.
    """
    return {
        "target_tile_lr_px": "1000x1000",
        "target_tile_hr_px": "4000x4000",
        "inference_path": "predict_large (sen2sr.utils)",
        "tile_size_used_by_predict_large": "128x128 LR",
        "overlap": 32,
        "tiles_processed": "sequentially (one at a time)",
        "per_tile_gpu_memory_bound": "~163 MB peak allocated (measured at 128x128 fp32)",
        "result_moved_to_cpu_per_tile": True,
        "full_output_buffer_location": "CPU",
        "full_output_buffer_size_for_1000x1000": "~244 MB RAM (float32, 4x4000x4000)",
        "non_square_behavior": (
            "predict_large pre-allocates output using X.shape[1] for both spatial "
            "dimensions. For a non-square input this would produce an incorrect output "
            "shape. Observed and identified during T1.1 validation. Not documented as "
            "a bug in the sen2sr source."
        ),
        "1000x1000_gpu_vram_feasibility": "UNVERIFIED — source analysis suggests per-tile VRAM is bounded, but actual behavior on 1000x1000 input on this hardware has not been measured.",
        "1000x1000_ram_feasibility": "Likely within budget (est. ~244 MB for output buffer on 16 GB system), but not measured.",
        "recommendation": (
            "A dedicated large-tile benchmark task (T1.x) is required. "
            "Do not treat 1000x1000 inference as verified without an actual measurement "
            "on this hardware."
        ),
    }


def run_benchmark(device_str: str = "cuda:0") -> Dict[str, Any]:
    """Run the full T1.3 benchmark and return the complete results dict."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. T1.3 requires GPU (RTX 3050).")

    device = torch.device(device_str)
    mb = 1024 ** 2

    # ── Software/environment metadata ────────────────────────────────────────
    import platform
    try:
        import sen2sr
        sen2sr_ver = sen2sr.__version__
    except AttributeError:
        sen2sr_ver = "0.8.5 (installed, __version__ not exposed)"

    props = torch.cuda.get_device_properties(device)
    env_info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": str(torch.backends.cudnn.version()),
        "numpy_version": np.__version__,
        "sen2sr_version": sen2sr_ver,
        "gpu_name": props.name,
        "gpu_total_memory_mb": round(props.total_memory / mb, 2),
        "gpu_total_memory_gb": round(props.total_memory / (1024 ** 3), 3),
    }

    # ── Git SHA ───────────────────────────────────────────────────────────────
    try:
        import subprocess
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_sha = "unavailable"

    # ── Model loading ─────────────────────────────────────────────────────────
    print(f"[T1.3] Loading SEN2SRLite model on {device} ...")
    torch.cuda.synchronize(device)
    t_load_start = time.perf_counter()
    model = load_sen2sr_lite_model(weights_dir=WEIGHTS_DIR, device=device)
    model.eval()
    torch.cuda.synchronize(device)
    model_load_time_s = time.perf_counter() - t_load_start
    print(f"[T1.3] Model loaded in {model_load_time_s*1000:.1f} ms")

    # ── Build input tensor ────────────────────────────────────────────────────
    input_tensor = make_input_tensor(device)
    input_shape = list(input_tensor.shape)   # [1, 4, 128, 128]
    input_dtype = str(input_tensor.dtype)

    # ── Memory before any inference ───────────────────────────────────────────
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    memory_before_mb = round(torch.cuda.memory_allocated(device) / mb, 2)

    # ── Warm-up runs (excluded from stats) ───────────────────────────────────
    print(f"[T1.3] Running {WARMUP_RUNS} warm-up inference passes ...")
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            torch.cuda.synchronize(device)
            _ = model(input_tensor)
            torch.cuda.synchronize(device)

    # ── Measured runs ─────────────────────────────────────────────────────────
    print(f"[T1.3] Running {MEASURED_RUNS} measured inference passes ...")
    run_records = []
    output_shape = None

    with torch.no_grad():
        for run_idx in range(1, MEASURED_RUNS + 1):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            output = model(input_tensor)

            torch.cuda.synchronize(device)
            elapsed_s = time.perf_counter() - t0

            peak_alloc_mb = round(torch.cuda.max_memory_allocated(device) / mb, 2)
            peak_reserved_mb = round(torch.cuda.max_memory_reserved(device) / mb, 2)

            if output_shape is None:
                output_shape = list(output.shape)

            record = {
                "run": run_idx,
                "inference_time_s": round(elapsed_s, 6),
                "inference_time_ms": round(elapsed_s * 1000, 3),
                "peak_allocated_mb": peak_alloc_mb,
                "peak_reserved_mb": peak_reserved_mb,
            }
            run_records.append(record)
            print(
                f"  run {run_idx}/{MEASURED_RUNS}: "
                f"{elapsed_s*1000:.3f} ms | "
                f"peak alloc: {peak_alloc_mb:.1f} MB | "
                f"peak reserved: {peak_reserved_mb:.1f} MB"
            )

    # ── Aggregate statistics ──────────────────────────────────────────────────
    times_s = [r["inference_time_s"] for r in run_records]
    times_ms = [r["inference_time_ms"] for r in run_records]
    peak_allocs = [r["peak_allocated_mb"] for r in run_records]
    peak_reserved = [r["peak_reserved_mb"] for r in run_records]

    stats = {
        "mean_inference_time_s": round(statistics.mean(times_s), 6),
        "median_inference_time_s": round(statistics.median(times_s), 6),
        "min_inference_time_s": round(min(times_s), 6),
        "max_inference_time_s": round(max(times_s), 6),
        "mean_inference_time_ms": round(statistics.mean(times_ms), 3),
        "median_inference_time_ms": round(statistics.median(times_ms), 3),
        "min_inference_time_ms": round(min(times_ms), 3),
        "max_inference_time_ms": round(max(times_ms), 3),
        "mean_peak_allocated_mb": round(statistics.mean(peak_allocs), 2),
        "max_peak_allocated_mb": round(max(peak_allocs), 2),
        "mean_peak_reserved_mb": round(statistics.mean(peak_reserved), 2),
        "max_peak_reserved_mb": round(max(peak_reserved), 2),
    }

    print(
        f"\n[T1.3] Summary:\n"
        f"  mean: {stats['mean_inference_time_ms']:.3f} ms | "
        f"  median: {stats['median_inference_time_ms']:.3f} ms | "
        f"  min: {stats['min_inference_time_ms']:.3f} ms | "
        f"  max: {stats['max_inference_time_ms']:.3f} ms\n"
        f"  peak VRAM allocated (mean): {stats['mean_peak_allocated_mb']:.1f} MB | "
        f"  reserved (mean): {stats['mean_peak_reserved_mb']:.1f} MB"
    )

    # ── Large-tile feasibility ────────────────────────────────────────────────
    feasibility = assess_large_tile_feasibility()

    # ── Assemble full metadata ────────────────────────────────────────────────
    metadata = {
        "task_id": "T1.3",
        "run_id": "t1_3_performance_v0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git_sha,
        "environment": env_info,
        "benchmark_config": {
            "weights_dir": WEIGHTS_DIR,
            "device": device_str,
            "precision": PRECISION,
            "scale": SCALE,
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS,
            "input_shape": input_shape,
            "input_band_order_to_model": ["B04", "B03", "B02", "B08"],
            "input_band_values_normalized": {
                "B04": REPO_BAND_VALUES[2],
                "B03": REPO_BAND_VALUES[1],
                "B02": REPO_BAND_VALUES[0],
                "B08": REPO_BAND_VALUES[3],
            },
            "input_dtype": input_dtype,
            "inference_call": "model(input_tensor) — direct call, same as T1.1 run_sen2sr for 128x128",
            "timing_method": "time.perf_counter() bracketed by torch.cuda.synchronize()",
        },
        "output_shape": output_shape,
        "model_load_time_s": round(model_load_time_s, 6),
        "model_load_time_ms": round(model_load_time_s * 1000, 3),
        "memory_allocated_before_inference_mb": memory_before_mb,
        "individual_runs": run_records,
        "aggregate": stats,
        "vram_budget_mb": 6144,
        "vram_budget_utilization_pct": round(
            stats["max_peak_allocated_mb"] / 6144 * 100, 2
        ),
        "large_tile_feasibility": feasibility,
        "notes": [
            "All measurements taken on NVIDIA RTX 3050 6GB Laptop GPU, native Linux.",
            "fp32 is the only safe precision for SEN2SRLite on this GPU (fp16 produces NaN in B08 — see T1.1 run_summary.json).",
            "Timing is CUDA-synchronized via torch.cuda.synchronize() before and after each inference call.",
            "Warm-up runs are excluded from all reported statistics.",
            "Peak memory statistics are reset before each measured run via torch.cuda.reset_peak_memory_stats().",
        ],
    }
    return metadata


def main():
    print("=" * 60)
    print("T1.3 — SEN2SR Speed & VRAM Verification")
    print("=" * 60)

    metadata = run_benchmark(device_str="cuda:0")

    out_dir = pathlib.Path("outputs/t1_3_performance_v0")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metadata.json"

    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[T1.3] Metadata written to: {out_path}")
    print("[T1.3] DONE.")


if __name__ == "__main__":
    main()
