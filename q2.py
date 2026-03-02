import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# 1. הגדרות ונתונים
data_dir = r'C:\Users\ozsha\Documents\el\data\crops'
batch_size = 16

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((60, 60)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
train_size = int(0.75 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# 2. בניית המודל המותאם (Tiny Net)
class TinyCameraNet(nn.Module):
    def __init__(self):
        super(TinyCameraNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 15 * 15, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 15 * 15)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = TinyCameraNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. לולאת אימון
for epoch in range(25):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()

# 4. הערכה ומטריקות (AUC + Report)
model.eval()
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1] # הסתברות למצלמה 2
        _, preds = torch.max(outputs, 1)
        all_probs.extend(probs.numpy())
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

print("--- דוח סיווג ---")
print(classification_report(all_labels, all_preds, target_names=['Cam 1', 'Cam 2']))
print(f"AUC Score: {roc_auc_score(all_labels, all_probs):.4f}")

torch.save(model.state_dict(), "tiny_camera_net.pth")