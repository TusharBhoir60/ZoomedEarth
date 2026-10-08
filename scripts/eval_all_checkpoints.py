import sys
import logging
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import torch

sys.path.append(str(Path(__file__).parent.parent))

import src.eval.wald as wald_mod
import src.infer.infer_tile as infer_tile_mod
from src.infer.run_sen2sr import load_sen2sr_lite_model
from src.eval.wald import run_wald_experiment
from src.eval.consistency import evaluate_scene_consistency

def _cleanup_rasters(step_eval_dir):
    print(f"  [Disk-Safe] Cleaning up raster intermediates for {step_eval_dir.name}")
    try:
        lr_dir = step_eval_dir / "lr_40m"
        sr_dir = step_eval_dir / "sen2sr_10m"
        bicubic_link = step_eval_dir / "bicubic_10m.tif"
        if lr_dir.exists():
            shutil.rmtree(lr_dir)
        if sr_dir.exists():
            shutil.rmtree(sr_dir)
        if bicubic_link.exists() or bicubic_link.is_symlink():
            bicubic_link.unlink()
        print(f"  [Disk-Safe] Cleanup completed for {step_eval_dir.name}")
    except Exception as e:
        print(f"  [Disk-Safe] Error during cleanup: {e}")

def find_best_checkpoint(ckpt_dir, hr_scene_dir, eval_dir):
    ckpt_dir = Path(ckpt_dir)
    eval_dir = Path(eval_dir)
    baselines_dir = eval_dir / "baselines"
    checkpoints_dir = eval_dir / "checkpoints"
    
    baselines_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints = list(ckpt_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        print("No checkpoints found.")
        return
        
    # Sort checkpoints by step number
    checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
        
    original_load = load_sen2sr_lite_model
    original_run_bicubic = wald_mod.run_bicubic
    
    shared_bicubic_path = baselines_dir / "bicubic_10m.tif"
    
    def make_patched_run_bicubic():
        def patched_run_bicubic(input_path, output_path, *args, **kwargs):
            if not shared_bicubic_path.exists():
                print("  [Disk-Safe] Generating shared bicubic...")
                original_run_bicubic(input_path, shared_bicubic_path, *args, **kwargs)
            
            output_path = Path(output_path)
            if output_path.exists() or output_path.is_symlink():
                output_path.unlink()
            output_path.symlink_to(shared_bicubic_path.resolve())
            
        return patched_run_bicubic
        
    # Also evaluate pretrained baseline
    print("Evaluating pretrained baseline...")
    pretrained_out = baselines_dir / "pretrained"
    
    with patch.object(wald_mod, 'run_bicubic', side_effect=make_patched_run_bicubic()):
        try:
            res_pretrained = run_wald_experiment(hr_scene_dir, pretrained_out, device="cuda:0")
            cons_pretrained = evaluate_scene_consistency(
                pretrained_out / "lr_40m" / "sen2sr_tile",
                pretrained_out / "sen2sr_10m" / "SEN2SR.tif"
            )
            # Serialize complete result immediately
            with open(pretrained_out / "metrics.json", "w") as f:
                json.dump({"wald": res_pretrained, "consistency": cons_pretrained}, f, indent=2)
        except Exception as e:
            print(f"Failed on pretrained baseline: {e}")
            raise
    
    best_ckpt = None
    best_rmse = float('inf')
    results_map = {}
    
    previous_best_dir = None
    
    for ckpt_path in checkpoints:
        print(f"Evaluating {ckpt_path.name}...")
        step_eval_dir = checkpoints_dir / ckpt_path.stem
        step_eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Need to capture the current ckpt_path in the closure
        def make_patched_load(path_to_load):
            def patched_load(weights_dir, device=None):
                model = original_load(weights_dir, device)
                ckpt = torch.load(path_to_load, map_location=device)
                model.sr_model.load_state_dict(ckpt["generator"])
                return model
            return patched_load
            
        success = False
        try:
            with patch.object(infer_tile_mod, 'load_sen2sr_lite_model', side_effect=make_patched_load(ckpt_path)):
                with patch.object(wald_mod, 'run_bicubic', side_effect=make_patched_run_bicubic()):
                    print(f"  [Disk-Safe] Starting evaluation for {ckpt_path.name}")
                    res = run_wald_experiment(hr_scene_dir, step_eval_dir, device="cuda:0")
                
            rmse_val = res["metrics"]["rmse_global"]["sen2sr"]["mean"]
            
            # Calculate consistency before deleting
            print(f"Evaluating consistency for {ckpt_path.name}...")
            cons_res = evaluate_scene_consistency(
                step_eval_dir / "lr_40m" / "sen2sr_tile",
                step_eval_dir / "sen2sr_10m" / "SEN2SR.tif"
            )
            
            ckpt_results = {
                "wald": res,
                "consistency": cons_res
            }
            results_map[ckpt_path.name] = ckpt_results
            
            # Serialize COMPLETE metric result immediately
            metrics_path = step_eval_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(ckpt_results, f, indent=2)
                
            # Verify the result JSON was successfully written
            if not metrics_path.exists() or metrics_path.stat().st_size == 0:
                raise RuntimeError(f"Metrics file {metrics_path} was not written properly.")
            
            print(f"  [Disk-Safe] Metrics written for {ckpt_path.name}")
            success = True
            
            print(f"{ckpt_path.name} Global RMSE: {rmse_val}")
            
            is_best = False
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_ckpt = ckpt_path
                is_best = True
                
            # Safely cleanup temporary rasters
            if is_best:
                # Cleanup the previous best, keep the current best
                if previous_best_dir and previous_best_dir != step_eval_dir:
                    _cleanup_rasters(previous_best_dir)
                previous_best_dir = step_eval_dir
            else:
                # Cleanup this non-best checkpoint
                _cleanup_rasters(step_eval_dir)
                
        except Exception as e:
            print(f"Evaluation failed for {ckpt_path.name}: {e}")
            if not success:
                print(f"  [Disk-Safe] Preserving temporary outputs in {step_eval_dir} for debugging.")
            raise
            
    if best_ckpt is None:
        print("Warning: All checkpoints produced NaN RMSE or failed.")
        best_ckpt_name = "None"
        finetuned_results = None
    else:
        print(f"Best checkpoint is {best_ckpt.name} with RMSE {best_rmse}")
        best_ckpt_name = str(best_ckpt)
        finetuned_results = {
            "wald": results_map[best_ckpt.name]["wald"],
            "consistency": results_map[best_ckpt.name]["consistency"]
        }
    
    full_results = {
        "best_checkpoint": best_ckpt_name,
        "pretrained": {
            "wald": res_pretrained,
            "consistency": cons_pretrained
        },
        "finetuned": finetuned_results,
        "all_checkpoints": results_map
    }
    
    with open(eval_dir / "t5_2_final_results.json", "w") as f:
        json.dump(full_results, f, indent=2)
        
    print("Done. Results saved to t5_2_final_results.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--hr-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    
    find_best_checkpoint(args.ckpt_dir, args.hr_dir, args.out_dir)
