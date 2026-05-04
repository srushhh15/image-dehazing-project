import torch
import torch.multiprocessing as mp
mp.set_start_method('fork', force=True)

torch.backends.cudnn.benchmark = True

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
    full_dataset = DehazeDataset(
        "data/reside/hazy",
        "data/reside/clean"
    )
    
    if DATASET_CONFIG["use_full_dataset"]:
        dataset = full_dataset
        print(f"✅ Using FULL dataset: {len(dataset)} images")
    else:
        dataset = Subset(full_dataset, list(range(min(DATASET_CONFIG["subset_size"], len(full_dataset)))))
        print(f"✅ Using SUBSET: {len(dataset)} images")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    print(f"📊 Dataset split: Train={train_size}, Val={val_size}")
    
    train_loader = DataLoader(
        train_data,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True
    )
    
    print("🧠 Initializing model...")
    model = EnhancedCNNDehaze().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
    
    epochs = 25   # 🔥 ONLY CHANGE
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 25
    
    train_losses = []
    val_losses = []
    train_psnrs = []
    val_psnrs = []
    train_ssims = []
    val_ssims = []
    
    csv_file = "metrics.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch", "Train Loss", "Train PSNR", "Train SSIM",
            "Val Loss", "Val PSNR", "Val SSIM", "LR"
        ])
    
    print("🚀 Starting training for 25 epochs...\n")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 60)
        
        train_loss, train_psnr, train_ssim = train_epoch(
            model, train_loader, optimizer, device
        )
        
        val_loss, val_psnr, val_ssim = validate(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_psnrs.append(train_psnr)
        val_psnrs.append(val_psnr)
        train_ssims.append(train_ssim)
        val_ssims.append(val_ssim)
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"Train PSNR: {train_psnr:.2f} | Val PSNR: {val_psnr:.2f}")
        print(f"Train SSIM: {train_ssim:.4f} | Val SSIM: {val_ssim:.4f}")
        print(f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
            print("✅ Best model saved!")
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            torch.save(
                model.state_dict(),
                f"checkpoints/model_epoch_{epoch+1}.pth"
            )
            print(f"✅ Checkpoint saved at epoch {epoch+1}")
        
        if patience_counter >= early_stop_patience:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break
        
        scheduler.step(val_loss)
        
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, train_loss, train_psnr, train_ssim,
                val_loss, val_psnr, val_ssim, current_lr
            ])
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    torch.save(model.state_dict(), "checkpoints/final_model.pth")
    torch.save(model.state_dict(), "enhanced_cnn_dehaze.pth")
    
    print("\n🎉 Training complete!")
if __name__ == "__main__":
    main()