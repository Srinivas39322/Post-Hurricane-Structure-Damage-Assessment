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
