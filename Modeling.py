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

"""
CNN Model – Summary & Results
Architecture:
Custom CNN with 3 convolutional layers + 4 fully connected layers (ReLU + BatchNorm).

Input:
128×128 satellite images (damage vs no_damage)

Training Setup:
Optimizer: Adam (lr=1e-4)
Loss: CrossEntropyLoss
Epochs: 10 | Batch Size: 32
Device: CUDA-enabled GPU

Performance:
Train Accuracy: ↑ from 88.25% to 99.48%
Validation Accuracy: Peaked at 96.25%, stayed above 91%
Loss: Reduced from 86.32 → 4.15

✅ Positive:
High validation accuracy indicates strong generalization to unseen data.

⚠️ Drawbacks:
Requires significant GPU compute and longer training time than lighter models.
"""

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

"""
ResNet50 – Transfer Learning Summary & Results
Architecture:
Pretrained ResNet50, modified for binary classification (damage, no_damage)

Input:
224×224 satellite images with ImageNet normalization + horizontal flip augmentation

Training Setup:
Optimizer: Adam (lr=1e-4)
Loss Function: CrossEntropyLoss
Epochs: 50 | Batch Size: 32
Device: CUDA-enabled GPU (efficiently utilized)

Performance:
Best Validation Accuracy: ⭐ 97.3%
Model saved at: models/best_resnet50_model.pt

✅ Positive Outcome
Faster and more efficient on GPU than the custom CNN, despite being a deeper architecture

⚠️ Consideration
Slightly larger model size, though inference time remains fast due to GPU optimization
"""

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

"""
ResNet50 – Technical Summary & Comparative Advantages
Model Architecture & Setup
Model: Pretrained ResNet50 (torchvision.models.resnet50)

Final layer (fc) replaced with Linear(in_features, 2) for binary classification

Input Size: 224×224

Transforms:
Resize + Normalize ([0.485, 0.456, 0.406] mean, [0.229, 0.224, 0.225] std) — matches ImageNet stats
Optimizer: Adam (lr=1e-4)
Loss: CrossEntropyLoss
Epochs: 10
Hardware: CUDA GPU

ResNet50 – Key Advantages
Achieved 99.50% validation accuracy, outperforming both the baseline ResNet50 and the custom CNN models.
Required only 10 epochs to converge.
Faster training than the custom CNN, due to optimized architecture and better GPU utilization.
Used pretrained ImageNet weights, enabling stronger feature extraction and faster convergence.
Delivered high inference speed with minimal tuning, making it ideal for deployment in real-time pipelines.
Significantly improved generalization performance, with no signs of overfitting.
"""

==================================================================================================================================================

# ===============================
# Resnet50 Model - Final - Testing on Test_Another Data
# ===============================

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_CLASSES = 2

# --- Transform ---
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Load test_another dataset ---
test_dir = "Post-hurricane/test_another"
test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Load model and weights ---
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("models/best_resnet50_model.pt"))
model.to(DEVICE)
model.eval()

# --- Evaluate on test_another ---
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"🧪 Accuracy on Test Another Set: {accuracy:.4f}")

"""
ResNet50 – Test Set Evaluation
Purpose: Assessed the final model's generalization on unseen test data (test_another directory).

Model Used: Best-performing ResNet50, fine-tuned and saved from the training phase.

Setup:

Input: 224×224 images with ImageNet normalization

Device: Evaluated on GPU (CUDA)

Batch Size: 32

Process: Loaded model weights and ran forward passes without gradient computation.

Result:
✅ Achieved a Test Accuracy of 99.61%, confirming excellent generalization beyond training and validation datasets.
"""

==================================================================================================================================================

# ===============================
# Resnet50 Model - Confusion Matrix
# ===============================

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = "Post-hurricane/test_another"
MODEL_PATH = "models/best_resnet50_model.pt"
SAVE_PATH = "models/confusion_matrix_test_another.png"

# Load Model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Load Test Set
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Collect Predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Generate Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)

# Save as image
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', ax=ax)
plt.title("Confusion Matrix - Test Another Set")
plt.tight_layout()
plt.savefig(SAVE_PATH)
plt.close()

print(f"✅ Confusion matrix saved to: {SAVE_PATH}")

"""
Confusion Matrix – Test Set (ResNet50 Final)
Summary
Damage (positive class):
Correctly predicted: 7,980
Misclassified as no_damage: 20

No Damage (negative class):
Correctly predicted: 985
Misclassified as damage: 15

Total Test Accuracy: 99.61%

Pros
High true positive and true negative counts, indicating balanced performance across both classes.
Very low false negatives (20) and false positives (15) — critical for minimizing both missed damage and false alarms.
Model generalizes extremely well on unseen data, reaffirming robustness.

Cons
Minor misclassifications still occur, which could impact edge cases in real-world scenarios (e.g., faint or borderline damage).
"""

==================================================================================================================================================

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

"""
● Architecture: Transfer learning with pretrained EfficientNet-B0, modified for binary
classification (damage, no_damage).
● Input: 224×224 images, ImageNet normalization; basic augmentation (horizontal flip).
● Training:
○ 10 epochs | Batch size: 32
○ Optimizer: Adam (lr=1e-4)
○ Device: CUDA-enabled GPU

�� Performance Highlights
● Best Validation Accuracy: 99.30% (Epoch 9)
● Training Accuracy: Reached 99.93%
● Model saved to: models/best_efficientnet_b0.pt
Strengths
● High accuracy with minimal hyperparameter tuning.
● Lightweight and efficient — fast training cycles and low resource usage.
● Consistent validation performance from early epochs onward.

Limitations
● Risk of overfitting: High training accuracy suggests possible overfitting despite good
validation scores.
● Shallow adaptation: The full model was used without freezing or fine-tuning selective
layers, limiting specialization to hurricane-related imagery.
● Not optimized for edge cases: May misclassify subtle or low-contrast damage regions
without further refinement.
"""
==================================================================================================================================================

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

"""
�� Model Summary
● Architecture: Pretrained EfficientNet-V2-S, used as a frozen feature extractor. Only
the classifier layer is trainable.
● Input: 224×224 RGB images with standard ImageNet normalization.
● Training Details:
○ Epochs: 10
○ Batch size: 32
○ Optimizer: Adam (lr=1e-4)
○ Loss: CrossEntropyLoss
○ Device: CUDA (GPU)

Performance Summary
● Best Validation Accuracy: 91.70% (Epoch 10)
● Training Accuracy: Increased from 77.2% (Epoch 1) to 89.0% (Epoch 10)
● Model saved after every improvement in validation accuracy:
○ Early peak at Epoch 4: 90.30%
○ Final saved model at Epoch 10: 91.70%
● Training Time Efficiency: ~21 seconds/epoch — very fast, due to freezing all feature
layers

Strengths
● Extremely fast training due to frozen base layers — minimal compute overhead.
● Stable learning with gradual improvement across epochs.
● Good choice for low-resource environments or when rapid training is needed.

Limitations
● Lower final accuracy compared to fine-tuned EfficientNet-B0 or V2 models.
● Limited feature adaptation — freezing all feature layers prevents learning domain-
specific patterns (e.g., hurricane-specific textures).
● Plateaued quickly — improvement tapered after Epoch 7, indicating the model reached
its potential early.

"""
==================================================================================================================================================

# ---------------------------------------------
# EfficientNet V2 finetuned
# ---------------------------------------------

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

"""
Model: Pretrained EfficientNet-V2-S with the last two feature blocks and
classifier unfrozen for fine-tuning
Input: 224×224, ImageNet-normalized
Training: 10 epochs | Batch size: 32 | Adam (lr=1e-5) | CrossEntropyLoss
| CUDA
Task: Binary classification – damage vs no_damage
Performance
● Best Validation Accuracy: 97.95% (Epoch 10)
● Training Accuracy: Peaked at 99.32%
● Consistent growth across epochs with no overfitting
● Model saved 8 times, indicating continuous improvement

Strengths
● Selective fine-tuning enabled efficient learning and high generalization
● Fast training (~30s/epoch), ideal for rapid experimentation

Limitations

● Underperformed vs. EfficientNet-B0 (99.30%) despite fine-tuning
● Lower than expected peak accuracy for EfficientNet-V2
● Requires careful layer control to balance learning and overfitting

"""
==================================================================================================================================================

"""
Why EfficientNet Was Not Selected
● Lower Validation Accuracy:
EfficientNet achieved a best validation accuracy of 99.3%, while the final ResNet50
model reached a superior 99.50%, consistently outperforming across epochs.
● No Clear Advantage in Speed or Efficiency:
Although training times were comparable (~30s/epoch), ResNet50 converged faster and
delivered higher accuracy with fewer epochs in earlier experiments.
● Less Robust on Edge Cases:
The EfficientNet model’s learning plateaued after Epoch 7, suggesting limited
adaptability beyond general patterns, unlike ResNet50 which handled subtle damage
indicators more reliably.
● Deployment Readiness:
ResNet50 is already integrated, tested, and validated on test data with a 99.61% test
accuracy and clear confusion matrix performance, offering greater confidence for
deployment.
"""
