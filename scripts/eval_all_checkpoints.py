import sys
import logging
import json
from pathlib import Path
from unittest.mock import patch
import torch

sys.path.append(str(Path(__file__).parent.parent))

import src.infer.infer_tile as infer_tile_mod
from src.infer.run_sen2sr import load_sen2sr_lite_model
from src.eval.wald import run_wald_experiment
from src.eval.consistency import evaluate_scene_consistency

def find_best_checkpoint(ckpt_dir, hr_scene_dir, eval_dir):
    ckpt_dir = Path(ckpt_dir)
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints = list(ckpt_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        print("No checkpoints found.")
        return
        
    # Sort checkpoints by step number
    checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
        
    original_load = load_sen2sr_lite_model
    
    best_ckpt = None
    best_rmse = float('inf')
    results_map = {}
    
    for ckpt_path in checkpoints:
        print(f"Evaluating {ckpt_path.name}...")
        step_eval_dir = eval_dir / ckpt_path.stem
        
        # Need to capture the current ckpt_path in the closure
        def make_patched_load(path_to_load):
            def patched_load(weights_dir, device=None):
                model = original_load(weights_dir, device)
                ckpt = torch.load(path_to_load, map_location=device)
                model.sr_model.load_state_dict(ckpt["generator"])
                return model
            return patched_load
            
        with patch.object(infer_tile_mod, 'load_sen2sr_lite_model', side_effect=make_patched_load(ckpt_path)):
            res = run_wald_experiment(hr_scene_dir, step_eval_dir, device="cuda:0")
            
        rmse_val = res["metrics"]["rmse_global"]["sen2sr"]["mean"]
        
        # Calculate consistency before deleting
        print(f"Evaluating consistency for {ckpt_path.name}...")
        cons_res = evaluate_scene_consistency(
            step_eval_dir / "lr_40m" / "sen2sr_tile",
            step_eval_dir / "sen2sr_10m" / "SEN2SR.tif"
        )
        
        results_map[ckpt_path.name] = {
            "wald": res,
            "consistency": cons_res
        }
        
        # Directory is preserved for detailed evidence
            
        print(f"{ckpt_path.name} Global RMSE: {rmse_val}")
        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_ckpt = ckpt_path
            
    print(f"Best checkpoint is {best_ckpt.name} with RMSE {best_rmse}")
    
    # Also evaluate pretrained baseline
    print("Evaluating pretrained baseline...")
    pretrained_out = eval_dir / "pretrained"
    res_pretrained = run_wald_experiment(hr_scene_dir, pretrained_out, device="cuda:0")
    
    cons_pretrained = evaluate_scene_consistency(
        pretrained_out / "lr_40m" / "sen2sr_tile",
        pretrained_out / "sen2sr_10m" / "SEN2SR.tif"
    )
    
    # Pretrained baseline directory is preserved
    
    full_results = {
        "best_checkpoint": str(best_ckpt),
        "pretrained": {
            "wald": res_pretrained,
            "consistency": cons_pretrained
        },
        "finetuned": {
            "wald": results_map[best_ckpt.name]["wald"],
            "consistency": results_map[best_ckpt.name]["consistency"]
        },
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
