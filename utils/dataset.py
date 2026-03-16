import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class DehazeDataset(Dataset):
    def __init__(self, hazy_dir, clean_dir, size=256):
        self.hazy_dir = hazy_dir
        self.clean_dir = clean_dir

        # Only keep image files
        self.hazy_images = sorted([
            f for f in os.listdir(hazy_dir)
            if f.endswith(".png") or f.endswith(".jpg")
        ])

        self.transform = T.Compose([
    T.Resize((256,256)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    T.ToTensor()
])

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):

        hazy_name = self.hazy_images[idx]
        hazy_path = os.path.join(self.hazy_dir, hazy_name)

        # Extract clean image id
        clean_id = hazy_name.split("_")[0]
        clean_name = clean_id + ".png"

        clean_path = os.path.join(self.clean_dir, clean_name)

        # Safety check
        if not os.path.exists(clean_path):
            raise FileNotFoundError(f"Missing clean image: {clean_name}")

        hazy = Image.open(hazy_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")

        hazy = self.transform(hazy)
        clean = self.transform(clean)

        return hazy, clean