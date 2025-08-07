import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score
import numpy as np
import os

from dataset_class import ChestXrayDataset, get_transforms

# Config
train_csv = "/content/drive/MyDrive/final_balanced_225_per_disease.csv"
train_img_dir = "/content/drive/MyDrive/resized_224/final_resized_224x224"

val_csv = "/content/drive/MyDrive/test_600_images.csv"
val_img_dir = "/content/drive/MyDrive/final_test_resizd_224x224"

save_model_path = "/content/drive/MyDrive/efficientnet_chestxray.pth"
num_epochs = 20
batch_size = 32
learning_rate = 1e-4
num_classes = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets
train_dataset = ChestXrayDataset(train_csv, train_img_dir, transform=get_transforms('train'))
val_dataset = ChestXrayDataset(val_csv, val_img_dir, transform=get_transforms('val'))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load model (EfficientNet)
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train():
    best_auc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_labels = []
        val_outputs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                val_labels.append(labels.cpu().numpy())
                val_outputs.append(outputs.cpu().numpy())

        val_labels = np.vstack(val_labels)
        val_outputs = np.vstack(val_outputs)

        auc = roc_auc_score(val_labels, val_outputs, average='macro')

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val AUC: {auc:.4f}")

        # Save the best model
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), save_model_path)
            print(f"Model saved with AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train()
