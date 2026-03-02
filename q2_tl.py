import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, roc_auc_score

# 1. הגדרות
data_dir = r'C:\Users\ozsha\Documents\el\data\crops'
batch_size = 16

# טרנספורמציות (התאמה ל-ResNet)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # ResNet מצפה ל-3 ערוצים
    transforms.Resize((60, 60)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# 2. ResNet18 מותאם ל-60x60
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# שינוי השכבה הראשונה והסרת ה-Maxpool כדי לשמור על רזולוציה
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity() 

# החלפת השכבה האחרונה
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# אימון רק על השכבה האחרונה (Transfer Learning)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 3. לולאת אימון
for epoch in range(20):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()

# 4. הערכה
model.eval()
all_probs, all_preds, all_labels = [], [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        _, preds = torch.max(outputs, 1)
        all_probs.extend(probs.numpy())
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

print("--- ResNet18 Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=['Cam 1', 'Cam 2']))
print(f"AUC Score: {roc_auc_score(all_labels, all_probs):.4f}")