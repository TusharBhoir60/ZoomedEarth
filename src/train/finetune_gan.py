import argparse
import csv
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torchvision.models as models

from src.infer.run_sen2sr import load_sen2sr_lite_model, permute_bgrn_to_rgbn, permute_rgbn_to_bgrn

def _get_git_sha():
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('ascii').strip()
    except Exception:
        return 'unknown'

class WaldDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, crop_size=64, samples_per_epoch=200):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.crop_size = crop_size
        self.samples_per_epoch = samples_per_epoch
        self.bands = ["B02", "B03", "B04", "B08"]

        self.lr_paths = {b: self.lr_dir / f"{b}.tif" for b in self.bands}
        self.hr_paths = {b: self.hr_dir / f"{b}.tif" for b in self.bands}
        
        with rasterio.open(self.lr_paths["B02"]) as src:
            self.lr_h, self.lr_w = src.shape

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        max_y = self.lr_h - self.crop_size
        max_x = self.lr_w - self.crop_size
        
        retries = 0
        while retries < 100:
            y0 = random.randint(0, max_y)
            x0 = random.randint(0, max_x)
            
            lr_window = Window(x0, y0, self.crop_size, self.crop_size)
            hr_window = Window(x0 * 4, y0 * 4, self.crop_size * 4, self.crop_size * 4)
            
            lr_arrs = []
            hr_arrs = []
            
            try:
                for b in self.bands:
                    with rasterio.open(self.lr_paths[b]) as src:
                        lr_arrs.append(src.read(1, window=lr_window))
                for b in self.bands:
                    with rasterio.open(self.hr_paths[b]) as src:
                        hr_arrs.append(src.read(1, window=hr_window))
            except Exception:
                retries += 1
                logging.info(f"[REJECT] x0={x0} y0={y0} due to exception")
                continue
            
            lr_arr = np.stack(lr_arrs, axis=0).astype(np.float32)
            hr_arr = np.stack(hr_arrs, axis=0).astype(np.float32)
            
            if np.max(hr_arr) > 2.0:
                hr_arr /= 10000.0
            if np.max(lr_arr) > 2.0:
                lr_arr /= 10000.0
                
            if not np.isfinite(lr_arr).all() or not np.isfinite(hr_arr).all():
                logging.info(f"[REJECT] x0={x0} y0={y0} due to NaN")
                retries += 1
                continue
                
            if random.random() > 0.5:
                lr_arr = np.flip(lr_arr, axis=1).copy()
                hr_arr = np.flip(hr_arr, axis=1).copy()
            if random.random() > 0.5:
                lr_arr = np.flip(lr_arr, axis=2).copy()
                hr_arr = np.flip(hr_arr, axis=2).copy()
                
            return torch.from_numpy(lr_arr), torch.from_numpy(hr_arr)
        
        raise RuntimeError("Failed to load a valid patch after multiple retries. The dataset may be largely invalid.")

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=4, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )
        
    def forward(self, x):
        return self.model(x)

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:18], vgg[18:27]])
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        # BGRN -> B02, B03, B04, B08. RGB is B04, B03, B02, which is indices 2, 1, 0
        x_rgb = x[:, [2, 1, 0], :, :]
        y_rgb = y[:, [2, 1, 0], :, :]
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x_rgb = (x_rgb - mean) / std
        y_rgb = (y_rgb - mean) / std
        
        loss = 0.0
        for block in self.blocks:
            x_rgb = block(x_rgb)
            y_rgb = block(y_rgb)
            loss += torch.nn.functional.l1_loss(x_rgb, y_rgb)
        return loss

def downsample_box_4x4_tensor(x):
    return torch.nn.functional.avg_pool2d(x, kernel_size=4, stride=4)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune.yaml")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps for smoke testing")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
        
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
        
    set_seed(cfg["seed"])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # 1. Load Generator
    logging.info(f"Loading pretrained generator from {cfg['pretrained_weights']}")
    # model is SRModelWithConstraint
    generator = load_sen2sr_lite_model(cfg["pretrained_weights"], device=device)
    # Enable gradients for sr_model
    for param in generator.sr_model.parameters():
        param.requires_grad = True
    generator.train()
    
    # 2. Discriminator
    discriminator = PatchGANDiscriminator().to(device)
    discriminator.train()
    
    # 3. Loss Functions
    l1_loss_fn = nn.L1Loss()
    perceptual_loss_fn = VGGPerceptualLoss().to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    
    # 4. Optimizers
    opt_g = optim.Adam(generator.sr_model.parameters(), lr=float(cfg["learning_rate_g"]))
    opt_d = optim.Adam(discriminator.parameters(), lr=float(cfg["learning_rate_d"]))
    
    scaler = GradScaler(enabled=cfg.get("amp_enabled", False))
    
    # Dataset
    lr_dir = Path(cfg["train_data_dir"]) / "lr_40m" / "sen2sr_tile"
    hr_dir = "data/processed/sentinel2/S2B_43RFM_20230203_0_L2A" # We use the explicit HR directory as specified in wald experiment
    
    try:
        train_ds = WaldDataset(lr_dir, hr_dir, crop_size=cfg["crop_size"], samples_per_epoch=cfg["max_steps"] * cfg["batch_size"])
    except Exception as e:
        logging.warning(f"Failed to use crop_size {cfg['crop_size']}, falling back to {cfg['fallback_crop_size']}. Error: {e}")
        train_ds = WaldDataset(lr_dir, hr_dir, crop_size=cfg["fallback_crop_size"], samples_per_epoch=cfg["max_steps"] * cfg["batch_size"])
        
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], num_workers=0, shuffle=True)
    
    # Setup state
    start_step = 0
    if cfg.get("resume_checkpoint") and Path(cfg["resume_checkpoint"]).exists():
        logging.info(f"Resuming from {cfg['resume_checkpoint']}")
        ckpt = torch.load(cfg["resume_checkpoint"], map_location=device)
        generator.sr_model.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_step = ckpt["step"]
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

    # CSV Logging
    csv_path = out_dir / cfg["logging"]["csv_file"]
    file_exists = csv_path.exists()
    csv_f = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_f)
    if not file_exists:
        csv_writer.writerow(["step", "loss_g_l1", "loss_g_perceptual", "loss_g_adv", "loss_g_cons", "loss_g_total", "loss_d", "peak_vram_mb"])

    logging.info(f"Starting training from step {start_step} to {cfg['max_steps']}")
    
    t0 = time.time()
    
    # Training Loop
    step = start_step
    data_iter = iter(train_dl)
    
    def check_finite(tensor, name, step, context=""):
        if tensor is None:
            return True
        if not torch.isfinite(tensor).all():
            logging.error(f"[{context}] Non-finite value in {name} at step {step}")
            try:
                logging.error(f"Shape: {tensor.shape}, Min: {torch.min(tensor).item()}, Max: {torch.max(tensor).item()}, Finite Count: {torch.isfinite(tensor).sum().item()} / {tensor.numel()}")
            except Exception as e:
                logging.error(f"Could not compute stats: {e}")
            return False
        return True

    def check_model_params(model, name, step, context=""):
        for n, p in model.named_parameters():
            if p.requires_grad:
                if not check_finite(p, f"{name} param {n}", step, context):
                    return False
                if p.grad is not None:
                    if not check_finite(p.grad, f"{name} grad {n}", step, context):
                        return False
        return True

    def get_grad_norm(model):
        total_norm = 0.0
        max_grad = 0.0
        max_param = 0.0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                max_grad = max(max_grad, p.grad.data.abs().max().item())
                max_param = max(max_param, p.data.abs().max().item())
        total_norm = total_norm ** 0.5
        return total_norm, max_grad, max_param

    while step < cfg["max_steps"]:
        try:
            lr_b, hr_b = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            lr_b, hr_b = next(data_iter)
            
        lr_b = lr_b.to(device)
        hr_b = hr_b.to(device)

        if not check_finite(lr_b, "lr_b", step, "data_load") or not check_finite(hr_b, "hr_b", step, "data_load"):
            logging.error(f"Data contains NaNs at step {step}! LR min/max/mean: {lr_b.min().item()}/{lr_b.max().item()}/{lr_b.mean().item()} HR min/max/mean: {hr_b.min().item()}/{hr_b.max().item()}/{hr_b.mean().item()}")
            break
        
        # Generator Forward
        with autocast(enabled=cfg.get("amp_enabled", False)):
            lr_rgbn = permute_bgrn_to_rgbn(lr_b)
            sr_rgbn = generator.sr_model(lr_rgbn)
            sr_b = permute_rgbn_to_bgrn(sr_rgbn)
            
            if not check_finite(sr_b, "sr_b", step, "generator_forward"): break
            
            loss_l1 = l1_loss_fn(sr_b, hr_b)
            if not check_finite(loss_l1, "loss_l1", step, "generator_forward"): break
            
            loss_perceptual = perceptual_loss_fn(sr_b, hr_b)
            if not check_finite(loss_perceptual, "loss_perceptual", step, "generator_forward"): break
            
            cons_down = downsample_box_4x4_tensor(sr_b)
            loss_cons = l1_loss_fn(cons_down, lr_b)
            if not check_finite(loss_cons, "loss_cons", step, "generator_forward"): break
            
            pred_fake = discriminator(sr_b)
            if not check_finite(pred_fake, "pred_fake_g", step, "generator_forward"): break
            loss_g_adv = bce_loss_fn(pred_fake, torch.ones_like(pred_fake))
            if not check_finite(loss_g_adv, "loss_g_adv", step, "generator_forward"): break
            
            w_l1, w_p, w_adv, w_cons = cfg["loss_weights"]["l1"], cfg["loss_weights"]["perceptual"], cfg["loss_weights"]["adversarial"], cfg["loss_weights"]["consistency"]
            loss_g = (w_l1 * loss_l1) + (w_p * loss_perceptual) + (w_adv * loss_g_adv) + (w_cons * loss_cons)
            loss_g = loss_g / cfg["gradient_accumulation_steps"]
            
        if not check_finite(loss_g, "loss_g_total", step, "generator_forward"): break
            
        scaler.scale(loss_g).backward()
        if not check_model_params(generator.sr_model, "Generator", step, "after_g_backward"): break
        
        # Discriminator Forward
        with autocast(enabled=cfg.get("amp_enabled", False)):
            pred_real = discriminator(hr_b)
            if not check_finite(pred_real, "pred_real_d", step, "discriminator_forward"): break
            
            pred_fake_d = discriminator(sr_b.detach())
            if not check_finite(pred_fake_d, "pred_fake_d", step, "discriminator_forward"): break
            
            loss_d_real = bce_loss_fn(pred_real, torch.ones_like(pred_real))
            loss_d_fake = bce_loss_fn(pred_fake_d, torch.zeros_like(pred_fake_d))
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d = loss_d / cfg["gradient_accumulation_steps"]
            
        if not check_finite(loss_d, "loss_d_total", step, "discriminator_forward"): break
            
        scaler.scale(loss_d).backward()
        if not check_model_params(discriminator, "Discriminator", step, "after_d_backward"): break
        
        if (step + 1) % cfg["gradient_accumulation_steps"] == 0:
            if step >= 100:
                g_norm, g_max_g, g_max_p = get_grad_norm(generator.sr_model)
                d_norm, d_max_g, d_max_p = get_grad_norm(discriminator)
                logging.info(f"Step {step} Gradients - G norm: {g_norm:.4f}, max_grad: {g_max_g:.4f}, max_param: {g_max_p:.4f} | D norm: {d_norm:.4f}, max_grad: {d_max_g:.4f}, max_param: {d_max_p:.4f}")
                
            scaler.step(opt_g)
            scaler.step(opt_d)
            scaler.update()
            
            if not check_model_params(generator.sr_model, "Generator", step, "after_optimizer_step"): break
            if not check_model_params(discriminator, "Discriminator", step, "after_optimizer_step"): break
            
            opt_g.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)
            
        step += 1
        
        # Logging
        if step % cfg["logging"]["log_interval"] == 0:
            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            else:
                peak_vram = 0.0
                
            csv_writer.writerow([
                step,
                loss_l1.item(), loss_perceptual.item(), loss_g_adv.item(), loss_cons.item(), loss_g.item() * cfg["gradient_accumulation_steps"],
                loss_d.item() * cfg["gradient_accumulation_steps"],
                peak_vram
            ])
            csv_f.flush()
            logging.info(f"Step {step}/{cfg['max_steps']} | Loss G: {loss_g.item() * cfg['gradient_accumulation_steps']:.4f} | Loss D: {loss_d.item() * cfg['gradient_accumulation_steps']:.4f}")
            
        # Checkpointing
        if step % cfg["checkpoint_interval"] == 0 or step == cfg["max_steps"]:
            ckpt_path = out_dir / f"checkpoint_step_{step}.pt"
            torch.save({
                "step": step,
                "generator": generator.sr_model.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "config": cfg,
                "git_sha": _get_git_sha()
            }, ckpt_path)
            logging.info(f"Saved checkpoint to {ckpt_path}")
            
    csv_f.close()
    
    total_time = time.time() - t0
    logging.info(f"Training completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
