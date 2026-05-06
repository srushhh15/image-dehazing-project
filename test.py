import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from pytorch_msssim import ssim
import csv

from models.cnn_dehaze import EnhancedCNNDehaze

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def calculate_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()

def plot_training_metrics():
    """Plot training metrics from metrics.csv"""
    print("\n📊 Generating training metrics plots...")
    
    try:
        # Read CSV file
        epochs = []
        train_losses = []
        val_losses = []
        train_psnrs = []
        val_psnrs = []
        train_ssims = []
        val_ssims = []
        
        with open('metrics.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['Epoch']))
                train_losses.append(float(row['Train Loss']))
                val_losses.append(float(row['Val Loss']))
                train_psnrs.append(float(row['Train PSNR']))
                val_psnrs.append(float(row['Val PSNR']))
                train_ssims.append(float(row['Train SSIM']))
                val_ssims.append(float(row['Val SSIM']))
        
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Loss Curve
        axes[0, 0].plot(epochs, train_losses, 'b-o', label="Train Loss", linewidth=2, markersize=5)
        axes[0, 0].plot(epochs, val_losses, 'r-s', label="Val Loss", linewidth=2, markersize=5)
        axes[0, 0].set_title("Loss Curve", fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel("Epoch", fontsize=12)
        axes[0, 0].set_ylabel("Loss", fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: PSNR Curve
        axes[0, 1].plot(epochs, train_psnrs, 'b-o', label="Train PSNR", linewidth=2, markersize=5)
        axes[0, 1].plot(epochs, val_psnrs, 'r-s', label="Val PSNR", linewidth=2, markersize=5)
        axes[0, 1].set_title("PSNR Curve (dB)", fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel("Epoch", fontsize=12)
        axes[0, 1].set_ylabel("PSNR (dB)", fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: SSIM Curve
        axes[1, 0].plot(epochs, train_ssims, 'b-o', label="Train SSIM", linewidth=2, markersize=5)
        axes[1, 0].plot(epochs, val_ssims, 'r-s', label="Val SSIM", linewidth=2, markersize=5)
        axes[1, 0].set_title("SSIM Curve", fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel("Epoch", fontsize=12)
        axes[1, 0].set_ylabel("SSIM", fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Summary Statistics
        axes[1, 1].axis('off')
        summary_text = f"""
TRAINING SUMMARY

Total Epochs: {len(epochs)}

Best Val Loss: {min(val_losses):.6f}
Best Val PSNR: {max(val_psnrs):.2f} dB
Best Val SSIM: {max(val_ssims):.4f}

Final Train Loss: {train_losses[-1]:.6f}
Final Train PSNR: {train_psnrs[-1]:.2f} dB
Final Train SSIM: {train_ssims[-1]:.4f}

Final Val Loss: {val_losses[-1]:.6f}
Final Val PSNR: {val_psnrs[-1]:.2f} dB
Final Val SSIM: {val_ssims[-1]:.4f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig("training_metrics_detailed.png", dpi=150, bbox_inches='tight')
        print("✅ Saved: training_metrics_detailed.png")
        plt.close()
        
        # Create simple loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-o', label="Train Loss", linewidth=2, markersize=5)
        plt.plot(epochs, val_losses, 'r-s', label="Val Loss", linewidth=2, markersize=5)
        plt.title("Loss Curve", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("loss_curve_detailed.png", dpi=150, bbox_inches='tight')
        print("✅ Saved: loss_curve_detailed.png")
        plt.close()
        
    except FileNotFoundError:
        print("⚠️  metrics.csv not found. Skipping training metrics plots.")
    except Exception as e:
        print(f"⚠️  Error generating plots: {e}")

def main():
    print("🧠 Loading model...")
    model = EnhancedCNNDehaze().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    print("✅ Model loaded!")
    
    # ← UPDATED: Resize all images to 256×256 for consistent display
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    input_folder = "data/reside/hazy/"
    clean_folder = "data/reside/clean/"
    output_folder = "outputs/"
    comparison_folder = "comparisons/"
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(comparison_folder, exist_ok=True)
    
    # Test on first 10 images (matches train.py)
    files = sorted(os.listdir(input_folder))[:10]
    
    print(f"\n🔄 Processing {len(files)} images...\n")
    
    total_psnr = 0
    total_ssim = 0
    count = 0
    
    for name in tqdm(files, desc="Dehazing"):
        hazy_path = os.path.join(input_folder, name)
        
        # ← UPDATED: Load and resize hazy image to 256×256
        hazy_img = Image.open(hazy_path).convert("RGB")
        hazy_img_resized = hazy_img.resize((256, 256), Image.Resampling.LANCZOS)
        
        x = transform(hazy_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            y = model(x)
        
        y_clamped = torch.clamp(y, 0, 1)
        out_img = transforms.ToPILImage()(y_clamped.squeeze(0).cpu())
        out_img.save(os.path.join(output_folder, name))
        
        # Compute metrics if clean image exists
        clean_id = name.split("_")[0]
        clean_name = clean_id + ".png"
        clean_path = os.path.join(clean_folder, clean_name)
        
        if os.path.exists(clean_path):
            # ← UPDATED: Load and resize clean image to 256×256
            clean_img = Image.open(clean_path).convert("RGB")
            clean_img_resized = clean_img.resize((256, 256), Image.Resampling.LANCZOS)
            
            clean_tensor = transform(clean_img).unsqueeze(0).to(device)
            
            psnr = calculate_psnr(y_clamped, clean_tensor)
            ssim_val = ssim(y_clamped, clean_tensor, data_range=1).item()
            
            total_psnr += psnr
            total_ssim += ssim_val
            count += 1
            
            # ← UPDATED: Save comparison with all images same size (256×256)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            axes[0].imshow(hazy_img_resized)
            axes[0].set_title("Hazy", fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(out_img)
            axes[1].set_title(f"Dehazed (PSNR: {psnr:.2f})", fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(clean_img_resized)
            axes[2].set_title("Ground Truth", fontsize=12, fontweight='bold')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_folder, f"comp_{name}"), dpi=100, bbox_inches='tight')
            plt.close()
    
    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        print(f"\n✅ Average PSNR: {avg_psnr:.2f} dB")
        print(f"✅ Average SSIM: {avg_ssim:.4f}")
    
    print(f"✅ Outputs saved to {output_folder}")
    print(f"✅ Comparisons saved to {comparison_folder}")
    
    # Generate training metrics plots
    plot_training_metrics()

if __name__ == "__main__":
    main()