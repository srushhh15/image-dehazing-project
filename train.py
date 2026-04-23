import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch import optim
import torch.nn.functional as F
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
import csv
import os

from models.cnn_dehaze import EnhancedCNNDehaze
from utils.dataset import DehazeDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# ------------------------------------------------
# LOSS FUNCTION
# ------------------------------------------------
def loss_function(pred, target):
    l1 = F.l1_loss(pred, target)
    ssim_loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    return l1 + 0.5 * ssim_loss


# ------------------------------------------------
# PSNR
# ------------------------------------------------
def calculate_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()


# ------------------------------------------------
# MAIN
# ------------------------------------------------
def main():

    os.makedirs("samples", exist_ok=True)

    full_dataset = DehazeDataset(
        "data/reside/hazy",
        "data/reside/clean"
    )

    # 50 images for tomorrow submission
    dataset = Subset(full_dataset, list(range(50)))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_data,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_data,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    model = EnhancedCNNDehaze().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    epochs = 15

    best_val = 999

    train_loss_list = []
    val_loss_list = []

    with open("metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch",
            "Train Loss",
            "Val Loss",
            "Train PSNR",
            "Val PSNR"
        ])

    for epoch in range(epochs):

        # ---------------- TRAIN ----------------
        model.train()

        total_train_loss = 0
        total_train_psnr = 0

        for hazy, clean in train_loader:

            hazy = hazy.to(device)
            clean = clean.to(device)

            output = model(hazy)

            loss = loss_function(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr = calculate_psnr(output, clean)

            total_train_loss += loss.item()
            total_train_psnr += psnr

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_psnr = total_train_psnr / len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()

        total_val_loss = 0
        total_val_psnr = 0

        with torch.no_grad():

            for hazy, clean in val_loader:

                hazy = hazy.to(device)
                clean = clean.to(device)

                output = model(hazy)

                loss = loss_function(output, clean)

                psnr = calculate_psnr(output, clean)

                total_val_loss += loss.item()
                total_val_psnr += psnr

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_psnr = total_val_psnr / len(val_loader)

        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Train PSNR: {avg_train_psnr:.2f}")
        print(f"Val PSNR: {avg_val_psnr:.2f}")
        print("-" * 40)

        # Save best model
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")

        # CSV log
        with open("metrics.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1,
                avg_train_loss,
                avg_val_loss,
                avg_train_psnr,
                avg_val_psnr
            ])

    torch.save(model.state_dict(), "enhanced_cnn_dehaze.pth")

    # Save graph
    plt.plot(train_loss_list, label="Train")
    plt.plot(val_loss_list, label="Validation")
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_graph.png")

    print("Training complete.")


if __name__ == "__main__":
    main()