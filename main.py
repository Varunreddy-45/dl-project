"""
Garbage Classification for Smart Waste Management using CNN
Single self-contained file: Dataset + CNN Model + Training + Accuracy Graph + Prediction Demo
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------- 1. DATASET CONFIG & SMART BINS ----------------
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

SMART_BINS = {
    "cardboard": "[Blue Bin] Recyclable Dry Packaging",
    "glass":     "[Green Bin] Recyclable Glass",
    "metal":     "[Yellow Bin] Recyclable Metal",
    "paper":     "[Blue Bin] Recyclable Paper",
    "plastic":   "[Orange Bin] Recyclable Plastic",
    "trash":     "[Grey Bin] General Non-Recyclable"
}

COLORS = {
    "cardboard": (180, 140, 95),
    "glass":     (140, 220, 210),
    "metal":     (190, 190, 195),
    "paper":     (245, 245, 240),
    "plastic":   (240, 80, 40),
    "trash":     (70, 60, 55)
}


def create_dataset_if_missing():
    """Generates a small starter dataset if not present."""
    data_dir = "dataset"
    if os.path.exists(data_dir):
        return data_dir

    print(">> Generating starter dataset in 'dataset/'...")
    for split, count in [("train", 30), ("val", 10)]:
        for cls in CLASSES:
            folder = os.path.join(data_dir, split, cls)
            os.makedirs(folder, exist_ok=True)
            for i in range(count):
                base_color = COLORS[cls]
                img = Image.new("RGB", (128, 128), color=base_color)
                draw = ImageDraw.Draw(img)
                # Draw simple shapes
                for _ in range(4):
                    x0, y0 = random.randint(5, 70), random.randint(5, 70)
                    draw.rectangle([x0, y0, x0 + 40, y0 + 40], outline=(255, 255, 255), width=2)
                img.save(os.path.join(folder, f"{cls}_{i}.jpg"))

    # Also save one test sample for each class
    os.makedirs("test_samples", exist_ok=True)
    for cls in CLASSES:
        img = Image.new("RGB", (128, 128), color=COLORS[cls])
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 100, 100], outline=(255, 255, 255), width=3)
        img.save(os.path.join("test_samples", f"{cls}.jpg"))

    print(">> Dataset generated successfully.\n")
    return data_dir


# ---------------- 2. BASIC CNN MODEL ----------------
class GarbageCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 64x64 -> 32x32
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ---------------- 3. TRAINING & ACCURACY GRAPH ----------------
def train_and_plot(data_dir, epochs=4):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_loader = DataLoader(datasets.ImageFolder(f"{data_dir}/train", transform=transform), batch_size=16, shuffle=True)
    val_loader = DataLoader(datasets.ImageFolder(f"{data_dir}/val", transform=transform), batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GarbageCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_acc_list, val_acc_list = [], []

    print("=" * 60)
    print(f" TRAINING CNN ON {device.type.upper()} ({epochs} EPOCHS)")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        t_correct, t_total, t_loss = 0, 0, 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            t_loss += loss.item() * imgs.size(0)
            t_correct += (outputs.argmax(1) == labels).sum().item()
            t_total += labels.size(0)

        # Validate
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                v_correct += (outputs.argmax(1) == labels).sum().item()
                v_total += labels.size(0)

        train_acc = (t_correct / t_total) * 100
        val_acc = (v_correct / v_total) * 100
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch}/{epochs} | Train Acc: {train_acc:5.1f}% | Val Acc: {val_acc:5.1f}% | Loss: {t_loss/t_total:.4f}")

    print("=" * 60)
    print(f">> FINAL TEST ACCURACY: {val_acc_list[-1]:.2f}% <<")
    print("=" * 60)

    # Save Graph
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, epochs + 1), train_acc_list, "b-o", label="Training Accuracy")
    plt.plot(range(1, epochs + 1), val_acc_list, "g--s", label="Validation Accuracy")
    plt.title("Garbage Classification - Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_graph.png", dpi=150)
    plt.close()
    print("[SUCCESS] Accuracy graph saved to 'accuracy_graph.png'\n")

    return model, transform, device


# ---------------- 4. SMART WASTE CLASSIFIER DEMO ----------------
def predict(image_path, model, transform, device):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        conf, pred_idx = probs.max(0)

    cls_name = CLASSES[pred_idx.item()]
    bin_name = SMART_BINS[cls_name]
    conf_pct = conf.item() * 100

    print("-" * 60)
    print(f"Image File   : {os.path.basename(image_path)}")
    print(f"Prediction   : {cls_name.upper()} ({conf_pct:.2f}% confidence)")
    print(f"Smart Bin    : {bin_name}")
    print("Class Confidence Percentages:")
    for i, c in enumerate(CLASSES):
        pct = probs[i].item() * 100
        bar = "#" * int(pct / 5)
        print(f"  - {c.ljust(10)}: {pct:6.2f}% | {bar}")


# ---------------- 5. MAIN EXECUTION ----------------
if __name__ == "__main__":
    # 1. Dataset
    data_dir = create_dataset_if_missing()

    # 2. Train & Plot
    model, transform, device = train_and_plot(data_dir, epochs=4)

    # 3. Test multiple items
    print("=" * 60)
    print("       LIVE PREDICTIONS & PERCENTAGE BREAKDOWN")
    print("=" * 60)
    for sample in ["plastic.jpg", "glass.jpg", "paper.jpg", "metal.jpg", "cardboard.jpg", "trash.jpg"]:
        sample_path = os.path.join("test_samples", sample)
        if os.path.exists(sample_path):
            predict(sample_path, model, transform, device)
    print("=" * 60)

