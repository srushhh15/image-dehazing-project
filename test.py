import os
import torch
from PIL import Image
import torchvision.transforms as T


from models.cnn_dehaze import CNNDehaze


# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def main():

    # Load trained model
    model = CNNDehaze().to(device)

    state_dict = torch.load("cnn_dehaze.pth", map_location=device)
    model.load_state_dict(state_dict)

    model.eval()

    # Dataset paths
    hazy_dir = "data/reside/hazy"
    output_dir = "outputs"

    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    # Image transform
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    # Test only first 20 images
    files = sorted(os.listdir(hazy_dir))[:20]

    for name in files:

        path = os.path.join(hazy_dir, name)

        # Load image
        img = Image.open(path).convert("RGB")

        # Preprocess
        x = transform(img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            pred = model(x)

        # Convert tensor to image
        pred = pred.squeeze(0).cpu().clamp(0, 1)

        output_img = T.ToPILImage()(pred)

        save_path = os.path.join(output_dir, name)

        output_img.save(save_path)

        print("Saved:", save_path)


if __name__ == "__main__":
    main()