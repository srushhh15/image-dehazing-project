# test.py (UPDATED)
import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from pytorch_msssim import ssim

from models.cnn_dehaze import EnhancedCNNDehaze

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def calculate_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()

def main():
    print("🧠 Loading model...")
    model = EnhancedCNNDehaze().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    print("✅ Model loaded!")
    
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
    
    files = sorted(os.listdir(input_folder))[:10]
    
    print(f"\n🔄 Processing {len(files)} images...\n")
    
    total_psnr = 0
    total_ssim = 0
    count = 0
    
    for name in tqdm(files, desc="Dehazing"):
        hazy_path = os.path.join(input_folder, name)
        
        hazy_img = Image.open(hazy_path).convert("RGB")
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
            clean_img = Image.open(clean_path).convert("RGB")
            clean_tensor = transform(clean_img).unsqueeze(0).to(device)
            
            psnr = calculate_psnr(y_clamped, clean_tensor)
            ssim_val = ssim(y_clamped, clean_tensor, data_range=1).item()
            
            total_psnr += psnr
            total_ssim += ssim_val
            count += 1
            
            # Save comparison
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(hazy_img)
            axes[0].set_title("Hazy")
            axes[0].axis('off')
            
            axes[1].imshow(out_img)
            axes[1].set_title(f"Dehazed (PSNR: {psnr:.2f})")
            axes[1].axis('off')
            
            axes[2].imshow(clean_img)
            axes[2].set_title("Ground Truth")
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

if __name__ == "__main__":
    main()