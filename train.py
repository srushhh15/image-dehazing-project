import torch
from torch.utils.data import DataLoader, Subset
from torch import nn, optim

from models.unet import UNet
from utils.dataset import DehazeDataset


# Device (CPU/GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def main():

    # Load full dataset
    full_dataset = DehazeDataset(
        "data/reside/hazy",
        "data/reside/clean"
    )

    print("Total images:", len(full_dataset))

    # Use only first 20 images for debugging
    dataset = Subset(full_dataset, list(range(20)))

    # DataLoader (Windows-safe)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    # Model
    model = UNet().to(device)

    # Loss and optimizer
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop (small epochs for debug)
    epochs = 2

    for epoch in range(epochs):

        model.train()
        total_loss = 0.0

        for batch_idx, (hazy, clean) in enumerate(loader):

            hazy = hazy.to(device)
            clean = clean.to(device)

            # Forward
            output = model(hazy)
            loss = loss_fn(output, clean)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Batch [{batch_idx+1}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        print("-" * 50)

    # Save model
    torch.save(model.state_dict(), "baseline_unet_debug.pth")
    print("Model saved as baseline_unet_debug.pth")


if __name__ == "__main__":
    main()