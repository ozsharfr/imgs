import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
from tqdm import tqdm

# הגדרות
data_dir = r'C:\Users\ozsha\Documents\el\data\crops'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. הכנת הנתונים לחילוץ מאפיינים (ללא אוגמנטציה בשלב זה)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset = datasets.ImageFolder(data_dir, transform=transform)

# 2. חילוץ מאפיינים (Embeddings)
model_feature = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model_feature = nn.Sequential(*list(model_feature.children())[:-1]).to(device)
model_feature.eval()

features, labels = [], []
loader = DataLoader(dataset, batch_size=32, shuffle=False)

print("Extracting features for clustering...")
with torch.no_grad():
    for imgs, lbls in tqdm(loader):
        f = model_feature(imgs.to(device)).cpu().numpy().reshape(imgs.size(0), -1)
        features.extend(f)
        labels.extend(lbls.numpy())

# 3. קיבוץ תמונות דומות (Clustering) למניעת זליגה
print("Performing clustering...")
clustering = AgglomerativeClustering(n_clusters=30, metric='cosine', linkage='average')
group_ids = clustering.fit_predict(features)

# 4. חלוקה מבוססת קבוצות (Group Split)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(range(len(dataset)), groups=group_ids))

# 5. אימון המודל (TinyCameraNet)
# הגדרה מחדש של Dataset עם טרנספורמציה מותאמת ל-60x60
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((60, 60)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset.transform = train_transform
train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

class TinyCameraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                                 nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(nn.Linear(32 * 15 * 15, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x): return self.fc(self.cnn(x).view(x.size(0), -1))

model = TinyCameraNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training model...")
for epoch in range(20):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs.to(device)), targets.to(device))
        loss.backward()
        optimizer.step()

# 6. הערכה
model.eval()
all_probs, all_preds, all_labels = [], [], []
with torch.no_grad():
    for inputs, targets in val_loader:
        out = model(inputs.to(device))
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
        all_labels.extend(targets.numpy())

print(classification_report(all_labels, all_preds, target_names=['Cam 1', 'Cam 2']))
print(f"AUC Score: {roc_auc_score(all_labels, all_probs):.4f}")