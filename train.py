import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch import optim
import torch.nn.functional as F
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
import csv
import os
from tqdm import tqdm
import time

from config import DATASET_CONFIG, TRAINING_CONFIG, print_config
from models.cnn_dehaze import EnhancedCNNDehaze
from utils.dataset import DehazeDataset

print_config()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# ------------------------------------------------
# LOSS FUNCTION
# ------------------------------------------------
def loss_function(pred, target):
    l1 = F.l1_loss(pred, target)
    ssim_loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    return l1 + 0.5 * ssim_loss


# ------------------------------------------------
# PSNR CALCULATION
# ------------------------------------------------
def calculate_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()


# ------------------------------------------------
# SSIM CALCULATION
# ------------------------------------------------
def calculate_ssim(pred, target):
    return ssim(pred, target, data_range=1, size_average=True).item()


# ------------------------------------------------
# TRAINING FUNCTION
# ------------------------------------------------
def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for hazy, clean in pbar:
        hazy = hazy.to(device)
        clean = clean.to(device)
        
        output = model(hazy)
        loss = loss_function(output, clean)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        psnr = calculate_psnr(output, clean)
        ssim_val = calculate_ssim(output, clean)
        
        total_loss += loss.item()
        total_psnr += psnr
        total_ssim += ssim_val
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(train_loader)
    avg_psnr = total_psnr / len(train_loader)
    avg_ssim = total_ssim / len(train_loader)
    
    return avg_loss, avg_psnr, avg_ssim


# ------------------------------------------------
# VALIDATION FUNCTION
# ------------------------------------------------
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for hazy, clean in pbar:
            hazy = hazy.to(device)
            clean = clean.to(device)
            
            output = model(hazy)
            loss = loss_function(output, clean)
            
            psnr = calculate_psnr(output, clean)
            ssim_val = calculate_ssim(output, clean)
            
            total_loss += loss.item()
            total_psnr += psnr
            total_ssim += ssim_val
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)
    
    return avg_loss, avg_psnr, avg_ssim


# ------------------------------------------------
# MAIN TRAINING LOOP
# ------------------------------------------------
def main():
    os.makedirs("samples", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    print("📦 Loading dataset...")
    
    # 🔥 ABSOLUTE PATH FIX FOR COLAB
    full_dataset = DehazeDataset(
        "/content/image-dehazing-project/data/reside/hazy",
        "/content/image-dehazing-project/data/reside/clean"
    )
    
    dataset = full_dataset
    print(f"✅ Using FULL dataset: {len(dataset)} images")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    print(f"📊 Dataset split: Train={train_size}, Val={val_size}")
    
    train_loader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )
    
    print("🧠 Initializing model...")
    model = EnhancedCNNDehaze().to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )
    
    epochs = 25   # 🔥 FINAL FIX
    
    print(f"🚀 Starting training for {epochs} epochs...\n")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss, _, _ = train_epoch(model, train_loader, optimizer, device)
        val_loss, _, _ = validate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}\n")
    
    print("\n🎉 Training complete!")


# ------------------------------------------------
# 🚨 THIS WAS MISSING IN YOUR CODE
# ------------------------------------------------
if __name__ == "__main__":
    main()