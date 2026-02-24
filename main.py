import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import classification_report  # Result report ke liye

class SDNETDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        cracked_folders = ["CD", "CP", "CW"]
        for structure in ["D", "P", "W"]:
            structure_path = os.path.join(root_dir, structure)
            if os.path.isdir(structure_path):
                for subfolder in os.listdir(structure_path):
                    subfolder_path = os.path.join(structure_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        label = 1 if subfolder in cracked_folders else 0
                        for img_file in os.listdir(subfolder_path):
                            self.image_paths.append(os.path.join(subfolder_path, img_file))
                            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, self.labels[idx]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_path = r"F:\workplace\visiontransformer\SDNET2018"
dataset = SDNETDataset(dataset_path, transform=train_transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Working on: {device} | Laptop GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

num_uncracked = dataset.labels.count(0)
num_cracked = dataset.labels.count(1)

weights = torch.tensor([1.0, float(num_uncracked / num_cracked)]).to(device)

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Training Start... Total Images: {len(dataset)}")
for epoch in range(3):
    model.train()
    all_preds, all_labels = [], []
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    print(f"\n--- Epoch {epoch + 1} Report ---")
    print(classification_report(all_labels, all_preds, target_names=['Uncracked', 'Cracked']))

torch.save(model.state_dict(), "resnet_sdnet_final.pth")
print("All Done! Model saved as resnet_sdnet_final.pth")