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
        
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.hazy_images)
    
    def __getitem__(self, idx):
        hazy_name = self.hazy_images[idx]
        hazy_path = os.path.join(self.hazy_dir, hazy_name)
        
        clean_id = hazy_name.split("_")[0] if "_" in hazy_name else hazy_name.split(".")[0]
        clean_name = clean_id + ".png"
        clean_path = os.path.join(self.clean_dir, clean_name)
        
        if not os.path.exists(clean_path):
            return self.__getitem__((idx + 1) % len(self.hazy_images))
        
        hazy = Image.open(hazy_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")
        
        hazy = self.transform(hazy)
        clean = self.transform(clean)
        
        return hazy, clean