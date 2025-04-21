import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# ===============================
# Configuration
# ===============================
train_dir = "Post-hurricane/train_another"
val_dir = "Post-hurricane/validation_another"
image_size = 128
batch_size = 32
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# Custom Dataset Class
# ===============================
class DamageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# ===============================
# Get Image Paths and Labels
# ===============================
def get_image_paths_and_labels(folder):
    image_paths, labels = [], []
    for label_name in ['damage', 'no_damage']:
        label = 1 if label_name == 'damage' else 0
        folder_path = os.path.join(folder, label_name)
        for img_file in os.listdir(folder_path):
            if img_file.endswith((".jpeg", ".jpg", ".png")):
                image_paths.append(os.path.join(folder_path, img_file))
                labels.append(label)
    return image_paths, labels


train_paths, train_labels = get_image_paths_and_labels(train_dir)
val_paths, val_labels = get_image_paths_and_labels(val_dir)

# ===============================
# Transformations
# ===============================
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# ===============================
# DataLoaders
# ===============================
train_dataset = DamageDataset(train_paths, train_labels, transform)
val_dataset = DamageDataset(val_paths, val_labels, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# ===============================
# CNN Model Definition
# ===============================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Automatically calculate the flattened size after conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            dummy_out = self.conv(dummy)
            self.flattened_size = dummy_out.view(1, -1).shape[1]

        # Fully connected classifier
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# ===============================
# Model Setup
# ===============================
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ===============================
# Training and Validation Loops
# ===============================
for epoch in range(num_epochs):
    model.train()
    total_loss, correct = 0.0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct / len(train_dataset)
    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f}")

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / len(val_dataset)
    print(f"Validation Acc: {val_acc:.4f}")

# ===============================
# Save the Trained Model
# ===============================
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/simple_cnn_model.pt")
print("✅ Best CNN model saved to models/simple_cnn_model.pt")

==================================================================================================================================================

# ===============================
# Resnet50 Model - Baseline
# ===============================
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_CLASSES = 2
EPOCHS = 50

# --- Transforms ---
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Datasets ---
train_dir = "Post-hurricane/train_another"
val_dir = "Post-hurricane/validation_another"
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- ResNet50 Model ---
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

# --- Loss and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- Prepare to Save Best Model ---
best_val_acc = 0.0
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
best_model_path = os.path.join(model_dir, "best_resnet50_model.pt")

# --- Training and Evaluation ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    # --- Validation ---
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()

    val_acc = val_correct / len(val_dataset)

    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model saved at epoch {epoch+1} to: {best_model_path}")

    print(f"\nEpoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {correct/len(train_dataset):.4f}, Val Acc: {val_acc:.4f}\n")

print("🎉 ResNet50 Training Complete")
print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")
print(f"📦 Best model saved at: {best_model_path}")

==================================================================================================================================================

# ===============================
# Resnet50 Model - Final
# ===============================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader

# --- Paths ---
train_dir = "Post-hurricane/train_another"
val_dir = "Post-hurricane/validation_another"
save_path = "models/new/best_resnet50_model.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️ Using device:", device)

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# --- Datasets ---
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# --- Model ---
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: damage / no_damage
model = model.to(device)

# --- Loss & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- Training ---
best_acc = 0.0
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # --- Validation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print("✅ Model improved. Saved!")

print(f"✅ Training complete. Best Acc: {best_acc:.4f}")

==================================================================================================================================================

