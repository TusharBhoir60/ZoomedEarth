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
        
        hr_dir = tmp_path / "hr_data"
        hr_dir.mkdir()
        out_dir = tmp_path / "eval_out"
        
        mock_res_100 = {
            "metrics": {
                "rmse_global": {"sen2sr": {"mean": 0.05}},
                "psnr": {"sen2sr": {"mean": 30.5}}
            }
        }
        mock_res_200 = {
            "metrics": {
                "rmse_global": {"sen2sr": {"mean": 0.04}},
                "psnr": {"sen2sr": {"mean": 32.1}}
            }
        }
        mock_res_pretrained = {
            "metrics": {
                "rmse_global": {"sen2sr": {"mean": 0.06}},
                "psnr": {"sen2sr": {"mean": 29.5}}
            }
        }
        mock_cons = {"psnr": 40.0}
        
        def mock_run_wald_experiment(hr, out, device):
            # Create a dummy file in out to verify it isn't deleted
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "dummy.txt").touch()
            name = Path(out).name
            if name == "checkpoint_step_100":
                return mock_res_100
            elif name == "checkpoint_step_200":
                return mock_res_200
            else:
                return mock_res_pretrained
                
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
        assert (out_dir / "checkpoint_step_100" / "dummy.txt").exists(), "Checkpoint 100 dir was deleted"
        assert (out_dir / "checkpoint_step_200" / "dummy.txt").exists(), "Checkpoint 200 dir was deleted"
        assert (out_dir / "pretrained" / "dummy.txt").exists(), "Pretrained dir was deleted"
        
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
        
        # psnr metric not silently discarded
        assert "psnr" in all_ckpts["checkpoint_step_100.pt"]["wald"]["metrics"]
        assert all_ckpts["checkpoint_step_100.pt"]["wald"]["metrics"]["psnr"]["sen2sr"]["mean"] == 30.5
        
        # existing result schema compatible (pretrained and finetuned)
        assert data["finetuned"]["wald"]["metrics"]["rmse_global"]["sen2sr"]["mean"] == 0.04
        assert data["pretrained"]["wald"]["metrics"]["rmse_global"]["sen2sr"]["mean"] == 0.06

