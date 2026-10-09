import argparse
import collections
import pathlib
import sys
import tempfile
import random
import shutil

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.eval.tier_c import score_tier_c
from src.models.detector import BuildingDetectorUNet
from src.train.detector_dataset import DetectorDataset

def proxy_bce_loss(logits, proxy_labels, valid_mask):
    """
    Computes masked BCE loss.
    - valid_mask: Excludes clouds and NaNs from the loss calculation entirely.
    
    Distinction Documented (D017):
    - proxy_labels contain 0 for both verified negatives AND unknown (255) pixels.
    - This is a training-only proxy-supervision transformation.
    - These proxy 0s are NOT independently verified non-buildings.
    """
    logits = logits.squeeze(1)
    
    bce = F.binary_cross_entropy_with_logits(logits, proxy_labels, reduction='none')
    masked_bce = bce * valid_mask.float()
    valid_count = valid_mask.sum()
    
    if valid_count > 0:
        return masked_bce.sum() / valid_count
    else:
        return logits.sum() * 0.0

@torch.no_grad()
def generate_validation_predictions(model, input_path, output_path, device, patch_size=512):
    """
    Runs model inference strictly sequentially over an AOI and writes predictions.
    Avoids loading full mosaic into RAM.
    """
    model.eval()
    
    with rasterio.open(input_path) as src:
        profile = src.profile
        profile.update(
            count=1,
            dtype='float32',
            compress='deflate',
            nodata=np.nan,
            tiled=True,
            blockxsize=patch_size,
            blockysize=patch_size
        )
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            for _, window in dst.block_windows(1):
                data = src.read(window=window)
                img_nan = np.isnan(data).any(axis=0)
                
                _, h, w = data.shape
                pad_h = (8 - (h % 8)) % 8
                pad_w = (8 - (w % 8)) % 8
                
                data = np.nan_to_num(data, nan=0.0).astype(np.float32)
                
                if pad_h > 0 or pad_w > 0:
                    data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                    
                tensor = torch.from_numpy(data).unsqueeze(0).to(device)
                
                logits = model(tensor)
                pred_arr = logits.squeeze(0).squeeze(0).cpu().numpy()
                
                if pad_h > 0 or pad_w > 0:
                    pred_arr = pred_arr[:h, :w]
                
                pred_arr[img_nan] = np.nan
                dst.write(pred_arr, 1, window=window)

def evaluate_pune_tier_c(model, pune_input_path, pune_label_path, pune_cloud_mask_path, budget_path, device):
    """
    Generates predictions for Pune and runs the official Tier C scorer.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_path = str(pathlib.Path(tmpdir) / "pune_preds.tif")
        generate_validation_predictions(model, pune_input_path, pred_path, device)
        
        results = score_tier_c(
            pred_path=pred_path,
            label_path=pune_label_path,
            cloud_mask_path=pune_cloud_mask_path,
            budget_path=budget_path,
            aoi_id="mosaic_aoi_pune_peri_urban"
        )
        return results

def train_model(config):
    """
    Executes the Tier C building detector training lifecycle.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = DetectorDataset(
        bicubic_path=config.train_bicubic,
        sen2sr_path=config.train_sen2sr,
        label_path=config.train_labels,
        cloud_mask_path=config.train_cloud_mask,
        patch_size=config.patch_size,
        seed=config.seed
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        drop_last=False
    )
    
    model = BuildingDetectorUNet().to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr)
    
    best_val_score = -1.0
    start_epoch = 0
    
    if config.resume_from and pathlib.Path(config.resume_from).exists():
        print(f"Resuming from {config.resume_from}")
        ckpt = torch.load(config.resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        else:
            raise ValueError("Attempted to resume from a checkpoint lacking optimizer state. Resuming from best_model.pt is not supported for full training continuation. Use last_checkpoint.pt instead.")
            
        best_val_score = ckpt.get('best_val_score', -1.0)
        start_epoch = ckpt.get('epoch', 0) + 1
        
        if 'python_rng' in ckpt:
            random.setstate(ckpt['python_rng'])
        if 'numpy_rng' in ckpt:
            np.random.set_state(ckpt['numpy_rng'])
        if 'torch_rng' in ckpt:
            torch.set_rng_state(ckpt['torch_rng'].cpu())
        if 'cuda_rng' in ckpt and torch.cuda.is_available():
            cuda_rngs = [r.cpu() if isinstance(r, torch.Tensor) else r for r in ckpt['cuda_rng']]
            torch.cuda.set_rng_state_all(cuda_rngs)
            
    print(f"Starting training from epoch {start_epoch} to {config.epochs}")
    
    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        skipped_batches = 0
        modality_counts = collections.Counter()
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            proxy_labels = batch['proxy_labels'].to(device)
            valid_mask = batch['valid_imagery_mask'].to(device)
            
            for mod in batch['modality']:
                modality_counts[mod] += 1
                
            optimizer.zero_grad()
            
            logits = model(images)
            loss = proxy_bce_loss(logits, proxy_labels, valid_mask)
            
            # D017: If a batch has zero valid pixels, skip the entire optimization update.
            if valid_mask.sum() == 0:
                skipped_batches += 1
                continue
                
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss detected at epoch {epoch}, batch {batch_idx}")
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            valid_batches += 1
            
        avg_loss = epoch_loss / valid_batches if valid_batches > 0 else float('nan')
        print(f"Epoch {epoch}: Avg Loss {avg_loss:.4f} | Valid Batches: {valid_batches} | Skipped: {skipped_batches}")
        print(f"Epoch {epoch} Modality Counts: {dict(modality_counts)}")
        
        # Validation
        if (epoch + 1) % config.val_cadence == 0:
            print("Running dual-modality validation on Pune...")
            try:
                # 1. SEN2SR Evaluation (Primary Checkpoint Metric)
                sen2sr_results = evaluate_pune_tier_c(
                    model=model,
                    pune_input_path=config.val_sen2sr, 
                    pune_label_path=config.val_labels,
                    pune_cloud_mask_path=config.val_cloud_mask,
                    budget_path=config.budget_path,
                    device=device
                )
                current_score = sen2sr_results["recall"]
                print(f"[Epoch {epoch}] Validation Recall (SEN2SR): {current_score:.4f}")
                
                # 2. Bicubic Evaluation (Diagnostic Only)
                try:
                    bicubic_results = evaluate_pune_tier_c(
                        model=model,
                        pune_input_path=config.val_bicubic, 
                        pune_label_path=config.val_labels,
                        pune_cloud_mask_path=config.val_cloud_mask,
                        budget_path=config.budget_path,
                        device=device
                    )
                    bicubic_score = bicubic_results["recall"]
                    print(f"[Epoch {epoch}] Validation Recall (Bicubic): {bicubic_score:.4f}")
                except Exception as b_e:
                    print(f"Bicubic evaluation failed: {b_e}. This does not affect SEN2SR checkpointing.")
                
                # Checkpoint Selection (Strictly SEN2SR)
                if current_score > best_val_score:
                    best_val_score = current_score
                    ckpt_path = pathlib.Path(config.checkpoint_dir) / "best_model.pt"
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # For best model, we do not need to persist optimizer/rng state (unless users want it, but we strictly separate them now)
                    torch.save({
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'best_val_score': best_val_score,
                        'config': vars(config)
                    }, ckpt_path)
                    print(f"Saved new best checkpoint with SEN2SR recall {best_val_score:.4f}")
                    
            except Exception as e:
                print(f"SEN2SR Validation failed: {e}. Validation skipped for this epoch.")

        # Always save last_checkpoint.pt at the end of the epoch to persist the latest training state
        ckpt_dir = pathlib.Path(config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        last_ckpt_dict = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_score': best_val_score,
            'config': vars(config),
            'seed': config.seed,
            'python_rng': random.getstate(),
            'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            last_ckpt_dict['cuda_rng'] = torch.cuda.get_rng_state_all()
            
        temp_ckpt = ckpt_dir / "last_checkpoint.tmp"
        final_ckpt = ckpt_dir / "last_checkpoint.pt"
        torch.save(last_ckpt_dict, temp_ckpt)
        shutil.move(str(temp_ckpt), str(final_ckpt))
        print(f"Saved latest training state to {final_ckpt}")
                
    print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-bicubic', required=True)
    parser.add_argument('--train-sen2sr', required=True)
    parser.add_argument('--train-labels', required=True)
    parser.add_argument('--train-cloud-mask', required=True)
    parser.add_argument('--val-sen2sr', required=True)
    parser.add_argument('--val-bicubic', required=True)
    parser.add_argument('--val-labels', required=True)
    parser.add_argument('--val-cloud-mask', required=True)
    parser.add_argument('--budget-path', required=True)
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--val-cadence', type=int, default=1)
    parser.add_argument('--resume-from', type=str, default=None)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    train_model(args)
