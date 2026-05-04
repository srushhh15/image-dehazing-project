import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class DehazeDataset(Dataset):
    def __init__(self, hazy_dir, clean_dir, size=256, augment=True):
        self.hazy_dir = hazy_dir
        self.clean_dir = clean_dir
        self.augment = augment
        
        self.hazy_images = sorted([
            f for f in os.listdir(hazy_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ])
        
        # Simple transforms - no normalization
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),  # Converts to [0, 1] automatically
        ])
    
    def __len__(self):
        return len(self.hazy_images)
    
    def __getitem__(self, idx):
        hazy_name = self.hazy_images[idx]
        hazy_path = os.path.join(self.hazy_dir, hazy_name)
        
        # Extract clean image id (assumes naming like: hazy_001.png -> 001.png)
        clean_id = hazy_name.split("_")[0] if "_" in hazy_name else hazy_name.split(".")[0]
        clean_name = clean_id + ".png"
        clean_path = os.path.join(self.clean_dir, clean_name)
        
        # Debug: Print first few loads
        if idx < 3:
            print(f"Loading: {hazy_name} -> {clean_name}")
            print(f"Exists: {os.path.exists(clean_path)}")
        
        if not os.path.exists(clean_path):
            raise FileNotFoundError(f"Missing: {clean_name} for {hazy_name}")
        
        hazy = Image.open(hazy_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")
        
        hazy = self.transform(hazy)
        clean = self.transform(clean)
        
        return hazy, clean