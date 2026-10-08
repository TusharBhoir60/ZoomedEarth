import pytest
import tempfile
from pathlib import Path
import json
from unittest.mock import patch, MagicMock
from scripts.eval_all_checkpoints import find_best_checkpoint

def test_evaluator_preserves_results_and_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "checkpoint_step_100.pt").touch()
        (ckpt_dir / "checkpoint_step_200.pt").touch()
        (ckpt_dir / "checkpoint_step_300.pt").touch()
        
        hr_dir = tmp_path / "hr_data"
        hr_dir.mkdir()
        out_dir = tmp_path / "eval_out"
        
        # 100 has RMSE 0.05
        mock_res_100 = {"metrics": {"rmse_global": {"sen2sr": {"mean": 0.05}}, "psnr": {"sen2sr": {"mean": 30.5}}}}
        # 200 has RMSE 0.04 (best)
        mock_res_200 = {"metrics": {"rmse_global": {"sen2sr": {"mean": 0.04}}, "psnr": {"sen2sr": {"mean": 32.1}}}}
        # 300 has RMSE 0.045
        mock_res_300 = {"metrics": {"rmse_global": {"sen2sr": {"mean": 0.045}}, "psnr": {"sen2sr": {"mean": 31.0}}}}
        
        mock_res_pretrained = {"metrics": {"rmse_global": {"sen2sr": {"mean": 0.06}}, "psnr": {"sen2sr": {"mean": 29.5}}}}
        mock_cons = {"psnr": 40.0}
        
        def mock_run_wald_experiment(hr, out, device):
            # Create a dummy file in out to verify if it gets deleted
            out = Path(out)
            out.mkdir(parents=True, exist_ok=True)
            lr_dir = out / "lr_40m"
            sr_dir = out / "sen2sr_10m"
            lr_dir.mkdir(exist_ok=True)
            sr_dir.mkdir(exist_ok=True)
            
            (lr_dir / "dummy_lr.tif").touch()
            (sr_dir / "dummy_sr.tif").touch()
            
            # wald also typically creates bicubic_10m.tif via run_bicubic, but eval_all_checkpoints patches run_bicubic to make symlinks
            # Let's mock a file there
            (out / "bicubic_10m.tif").touch()
            
            name = out.name
            if name == "checkpoint_step_100": return mock_res_100
            elif name == "checkpoint_step_200": return mock_res_200
            elif name == "checkpoint_step_300": return mock_res_300
            else: return mock_res_pretrained
                
        def mock_evaluate_scene_consistency(lr, sr):
            return mock_cons

        with patch("scripts.eval_all_checkpoints.run_wald_experiment", side_effect=mock_run_wald_experiment), \
             patch("scripts.eval_all_checkpoints.evaluate_scene_consistency", side_effect=mock_evaluate_scene_consistency), \
             patch("scripts.eval_all_checkpoints.patch") as mock_patch:
            # We mock 'patch.object' context manager used inside the loop
            mock_cm = MagicMock()
            mock_cm.__enter__.return_value = None
            mock_patch.object.return_value = mock_cm
            
            find_best_checkpoint(ckpt_dir, hr_dir, out_dir)
            
        # 1. Directory persistence check
        baselines_dir = out_dir / "baselines"
        checkpoints_dir = out_dir / "checkpoints"
        
        # Pretrained should keep its dir and rasters
        assert (baselines_dir / "pretrained").exists()
        assert (baselines_dir / "pretrained" / "lr_40m").exists()
        
        # Checkpoints dirs should all exist because metrics.json is kept
        assert (checkpoints_dir / "checkpoint_step_100").exists()
        assert (checkpoints_dir / "checkpoint_step_200").exists()
        assert (checkpoints_dir / "checkpoint_step_300").exists()
        
        # Rasters for non-best should be deleted (100 and 300)
        assert not (checkpoints_dir / "checkpoint_step_100" / "lr_40m").exists()
        assert not (checkpoints_dir / "checkpoint_step_100" / "sen2sr_10m").exists()
        assert not (checkpoints_dir / "checkpoint_step_100" / "bicubic_10m.tif").exists()
        
        assert not (checkpoints_dir / "checkpoint_step_300" / "lr_40m").exists()
        assert not (checkpoints_dir / "checkpoint_step_300" / "sen2sr_10m").exists()
        assert not (checkpoints_dir / "checkpoint_step_300" / "bicubic_10m.tif").exists()
        
        # Rasters for best should be kept (200 is best because it has RMSE 0.04)
        assert (checkpoints_dir / "checkpoint_step_200" / "lr_40m").exists()
        assert (checkpoints_dir / "checkpoint_step_200" / "sen2sr_10m").exists()
        
        # Metrics json should be written for ALL checkpoints
        for step in [100, 200, 300]:
            metrics_file = checkpoints_dir / f"checkpoint_step_{step}" / "metrics.json"
            assert metrics_file.exists(), f"metrics.json missing for {step}"
            with open(metrics_file) as f:
                ckpt_data = json.load(f)
            assert "wald" in ckpt_data
            assert "consistency" in ckpt_data
        
        # 2. Results format check
        results_file = out_dir / "t5_2_final_results.json"
        assert results_file.exists()
        
        with open(results_file) as f:
            data = json.load(f)
            
        assert data["best_checkpoint"] == str(ckpt_dir / "checkpoint_step_200.pt")
        
        # detailed metrics preserved
        all_ckpts = data["all_checkpoints"]
        assert "checkpoint_step_100.pt" in all_ckpts
        assert "checkpoint_step_200.pt" in all_ckpts
        assert "checkpoint_step_300.pt" in all_ckpts
        
        # psnr metric not silently discarded
        assert "psnr" in all_ckpts["checkpoint_step_100.pt"]["wald"]["metrics"]
        assert all_ckpts["checkpoint_step_100.pt"]["wald"]["metrics"]["psnr"]["sen2sr"]["mean"] == 30.5
        
        # existing result schema compatible (pretrained and finetuned)
        assert data["finetuned"]["wald"]["metrics"]["rmse_global"]["sen2sr"]["mean"] == 0.04
        assert data["pretrained"]["wald"]["metrics"]["rmse_global"]["sen2sr"]["mean"] == 0.06
