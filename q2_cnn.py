import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, roc_auc_score


# -------------------------
# Logging
# -------------------------
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger(__name__)


logger = setup_logger()


# -------------------------
# Config
# -------------------------
DATA_DIR = r"C:\Users\ozsha\Documents\el\data\crops"
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 25
SEED = 42
MODEL_PATH = "tiny_camera_net.pth"
NUM_CLASSES = 2
CLASS_NAMES = ["Cam 1", "Cam 2"]


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Data - tries augmentation but it did not help, so we keep it simple
# -------------------------
def build_transform():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((60, 60)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def build_loaders(data_dir: str, batch_size: int, seed: int):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")

    dataset = datasets.ImageFolder(data_dir, transform=build_transform())

    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# -------------------------
# Model - attempt to use Resnet18 did not work well
# -------------------------
class TinyCameraNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 15 * 15, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# -------------------------
# Train / Eval
# -------------------------
def train(model, loader, criterion, optimizer, device, epochs: int):
    model.train()

    for epoch in range(epochs):
        running = 0.0
        n = 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running += loss.item() * bs
            n += bs

        avg_loss = running / max(n, 1)
        logger.info(f"Epoch {epoch+1:02d}/{epochs} | loss={avg_loss:.4f}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        logits = model(inputs)

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    logger.info("--- Classification Report ---")
    logger.info("\n" + classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    try:
        auc = roc_auc_score(all_labels, all_probs)
        logger.info(f"AUC Score: {auc:.4f}")
    except ValueError as e:
        logger.warning(f"AUC Score unavailable: {e}")


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_loader, val_loader = build_loaders(DATA_DIR, BATCH_SIZE, SEED)

    model = TinyCameraNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train(model, train_loader, criterion, optimizer, device, EPOCHS)
    evaluate(model, val_loader, device)

    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()