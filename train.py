import torch
from torch.utils.data import DataLoader, Subset
from torch import optim
import torch.nn.functional as F
from pytorch_msssim import ssim

from models.cnn_dehaze import CNNDehaze
from utils.dataset import DehazeDataset


# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# Hybrid Loss Function (L1 + SSIM)
def loss_function(pred, target):

    l1 = F.l1_loss(pred, target)

    ssim_loss = 1 - ssim(pred, target, data_range=1, size_average=True)

    return l1 + 0.5 * ssim_loss


def main():

    # Load dataset
    full_dataset = DehazeDataset(
        "data/reside/hazy",
        "data/reside/clean"
    )

    print("Total images:", len(full_dataset))

    # Debug training: only first 20 images
    dataset = Subset(full_dataset, list(range(20)))

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    # Initialize model
    model = CNNDehaze().to(device)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    epochs = 10

    for epoch in range(epochs):

        model.train()
        total_loss = 0.0

        for batch_idx, (hazy, clean) in enumerate(loader):

            hazy = hazy.to(device)
            clean = clean.to(device)

            # Forward pass
            output = model(hazy)

            # Compute loss
            loss = loss_function(output, clean)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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

        scheduler.step()

    # Save trained model
    torch.save(model.state_dict(), "cnn_dehaze.pth")



    print("Model saved as cnn_dehaze.pth")


if __name__ == "__main__":
    main()