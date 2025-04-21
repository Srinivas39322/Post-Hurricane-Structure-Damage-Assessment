# ---------------------------------------------
# Transfer Learning: EfficientNet-B0 (PyTorch)
# ---------------------------------------------

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
EPOCHS = 10

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

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

# --- EfficientNet-B0 Model ---
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

# --- Training and Evaluation ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_val_acc = 0.0
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

    print(f"\nEpoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {correct/len(train_dataset):.4f}, Val Acc: {val_acc:.4f}\n")

    # --- Save Best Model ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/best_efficientnet_b0.pt")
        print("✅ Best EfficientNet-B0 model saved!")

print("✅ EfficientNet-B0 Training Complete. Best Val Accuracy: {:.4f}".format(best_val_acc))

## This code trains an EfficientNet-B0 model for binary image classification using PyTorch. It loads and preprocesses image data from specified training and validation directories, applies data augmentation, fine-tunes a pre-trained EfficientNet-B0 model, and evaluates its performance over multiple epochs—saving the model with the best validation accuracy.
