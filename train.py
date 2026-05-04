import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch import optim
import torch.nn.functional as F
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
import csv
import os
from tqdm import tqdm

from models.cnn_dehaze import EnhancedCNNDehaze
from utils.dataset import DehazeDataset

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
    
    # Use subset for testing (change range to use more images)
    dataset = Subset(full_dataset, list(range(min(50, len(full_dataset)))))
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    print(f"📊 Dataset split: Train={train_size}, Val={val_size}")
    
    train_loader = DataLoader(
        train_data,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == "cuda" else False
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
    
    epochs = 50  # ← CHANGED FROM 10 TO 50
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10
    
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
    
    print("🚀 Starting training...\n")
    
    for epoch in range(epochs):
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
        
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"Train PSNR: {train_psnr:.2f} | Val PSNR: {val_psnr:.2f}")
        print(f"Train SSIM: {train_ssim:.4f} | Val SSIM: {val_ssim:.4f}")
        print(f"LR: {current_lr:.2e}\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
            print("✅ Best model saved!")
        else:
            patience_counter += 1
        
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
    
    torch.save(model.state_dict(), "checkpoints/final_model.pth")
    torch.save(model.state_dict(), "enhanced_cnn_dehaze.pth")
    
    print("\n📊 Plotting results...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(train_losses, label="Train Loss", marker='o', markersize=3)
    axes[0, 0].plot(val_losses, label="Val Loss", marker='s', markersize=3)
    axes[0, 0].set_title("Loss Curve")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(train_psnrs, label="Train PSNR", marker='o', markersize=3)
    axes[0, 1].plot(val_psnrs, label="Val PSNR", marker='s', markersize=3)
    axes[0, 1].set_title("PSNR Curve")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(train_ssims, label="Train SSIM", marker='o', markersize=3)
    axes[1, 0].plot(val_ssims, label="Val SSIM", marker='s', markersize=3)
    axes[1, 0].set_title("SSIM Curve")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("SSIM")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(train_losses, label="Train", marker='o', markersize=3)
    axes[1, 1].plot(val_losses, label="Validation", marker='s', markersize=3)
    axes[1, 1].set_title("Overall Loss Comparison")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150, bbox_inches='tight')
    print("✅ Results saved to training_results.png")
    
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_graph.png")
    
    print("\n" + "="*60)
    print("🎉 Training complete!")
    print(f"📈 Best validation loss: {best_val_loss:.6f}")
    print(f"📈 Final validation PSNR: {val_psnr:.2f} dB")
    print(f"📈 Final validation SSIM: {val_ssim:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()