import torch
from torchvision import transforms
from PIL import Image
import os

from models.cnn_dehaze import EnhancedCNNDehaze

device = "cuda" if torch.cuda.is_available() else "cpu"

model = EnhancedCNNDehaze().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

input_folder = "data/reside/hazy/"
output_folder = "outputs/"

os.makedirs(output_folder, exist_ok=True)

files = os.listdir(input_folder)[:10]

for i, name in enumerate(files):

    print(f"Processing {i+1}/10")

    img = Image.open(os.path.join(input_folder, name)).convert("RGB")

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(x)

    y = y.squeeze(0).cpu()

    out = transforms.ToPILImage()(y)

    out.save(os.path.join(output_folder, name))

print("Outputs saved.")