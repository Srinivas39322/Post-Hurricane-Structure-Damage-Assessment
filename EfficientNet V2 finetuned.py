#EfficientNet V2 finetuned
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_CLASSES = 2
EPOCHS = 10
BEST_MODEL_PATH = "models/best_efficientnetv2_finetuned.pt"

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder("Post-hurricane/train_another", transform=transform)
val_dataset = datasets.ImageFolder("Post-hurricane/validation_another", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Load EfficientNet-V2-S with pretrained weights
model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

# Fine-tune only the last two blocks
for name, param in model.named_parameters():
    if "features.6" in name or "features.7" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Replace classifier
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

# Best model tracking
best_val_acc = 0.0

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            val_correct += (preds == labels).sum().item()

    train_acc = correct / len(train_dataset)
    val_acc = val_correct / len(val_dataset)
    print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f} | Val Acc = {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"✅ Best model saved to {BEST_MODEL_PATH}")

print(f"🏁 Training Complete. Best Val Accuracy: {best_val_acc:.4f}")

## This code fine-tunes an EfficientNet-V2-S model for binary image classification by training only the last two feature blocks and the classifier. It preprocesses image data, uses a lower learning rate for selective fine-tuning, evaluates model performance across epochs, and saves the best-performing model based on validation accuracy.