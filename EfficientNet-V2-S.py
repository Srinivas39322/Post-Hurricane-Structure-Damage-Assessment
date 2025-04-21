# ---------------------------------------------
# EfficientNet-V2-S (Frozen Feature Extractor)
# ---------------------------------------------

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_CLASSES = 2
EPOCHS = 10
BEST_MODEL_PATH = "models/best_efficientnetv2_frozen.pt"

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Datasets & Loaders ---
train_data = datasets.ImageFolder("Post-hurricane/train_another", transform=transform)
val_data = datasets.ImageFolder("Post-hurricane/validation_another", transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# --- Model: EfficientNet-V2-S (Frozen) ---
model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(device)

# --- Loss, Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# --- Training Loop ---
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    loss_sum, correct = 0, 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        correct += (out.argmax(1) == y).sum().item()

    train_acc = correct / len(train_data)

    # --- Validation ---
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            val_correct += (pred == y).sum().item()

    val_acc = val_correct / len(val_data)
    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # --- Save Best Model ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"✅ Best model saved to {BEST_MODEL_PATH}")

print(f"🏁 Training Complete. Best Val Accuracy: {best_val_acc:.4f}")

## This code trains a transfer learning model using EfficientNet-V2-S with frozen feature layers for binary image classification. It loads and preprocesses image data, fine-tunes only the classifier head, evaluates performance over multiple epochs, and saves the model with the highest validation accuracy.