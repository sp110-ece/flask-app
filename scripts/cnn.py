import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import dataset
from torch.utils.data import DataLoader
import plot
import os
import torchvision.models as models

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(BASE_DIR, "data", "aug_images")
CSV_PATH = os.path.join(BASE_DIR, "data", "labels", "keypoints.csv")

TEST_IMG_DIR = os.path.join(BASE_DIR, "data", "test")
TEST_CSV_PATH = os.path.join(BASE_DIR, "data", "test_labels", "standing_quad_pose.csv")


print("Working directory:", os.getcwd())
print("File exists:", os.path.exists("./data/labels/standing_quad_pose.csv"))




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 0
batch_size = 2
learning_rate = 0.00005

train_set = DataLoader(dataset.PoseDataset(img_dir=IMG_DIR, csv_path=CSV_PATH), batch_size = batch_size, shuffle=True)
test_set = DataLoader(dataset.PoseDataset(img_dir=TEST_IMG_DIR, csv_path=TEST_CSV_PATH), batch_size=batch_size, shuffle=False)

resnet = models.resnet18(pretrained=True)
modules = list(resnet.children())[:-1]
backbone = nn.Sequential(*modules)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 12)
        )
        
        
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
        

model = ConvNet().to(device)
criterion = nn.SmoothL1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



if os.path.exists("checkpoint.pth"):
    checkpoint = torch.load("checkpoint.pth")
    model.load_state_dict(checkpoint['weights'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print("loaded checkpoint")


for param in model.backbone.parameters():
    param.requires_grad =False


def unfreeze_last_block(model):
    for name, child in model.backbone.named_children():
        if name == 'layer4':
            for param in child.parameters():
                param.requires_grad =True


n_steps = len(train_set)

train_losses = []
val_losses = []
min_val_loss = 5
for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    print(epoch)
    for images, labels, mask in train_set:
        
        if (epoch > 7):
            unfreeze_last_block(model)
        images = images.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        outputs = model(images)
        # mask_expanded = mask.unsqueeze(-1).repeat(1, 2).view_as(labels)

        loss = criterion(outputs * mask,  labels * mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        running_train_loss += loss.item()
    train_loss = running_train_loss / len(train_set)
    train_losses.append(train_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, labels, mask in test_set:
            images = images.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            outputs = model(images)

            loss = criterion(outputs * mask, labels * mask)
            running_val_loss += loss.item()
    val_loss = running_val_loss / len(test_set)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    if (val_loss < min_val_loss):
        torch.save({'optimizer_state': optimizer.state_dict(), 'weights': model.state_dict()}, "checkpoint.pth")
        min_val_loss = val_loss
        print("model saved")



print('finish training')

model.eval()
total_error = 0
total_keypoints = 0
with torch.no_grad():
    
    for images, labels, mask in test_set:
        print(images.shape)
        image = images[0]
        label = labels[0]
        
        images = images.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        outputs = model(images)
        # print(outputs.shape)
        output = outputs[0]
        plot.plot_image(image, label, output)
        num_keypoints = 6
        outputs = outputs.view(-1, num_keypoints, 2)
        labels = labels.view(-1, num_keypoints, 2)
        # print("labels")
        # print(labels)
        # print("outputs")
        print(outputs)

        

        if torch.isnan(outputs).any() or torch.isnan(labels).any():
            print("NaN detected in outputs or labels!")
            continue
        if torch.isinf(outputs).any() or torch.isinf(labels).any():
            print("Inf detected in outputs or labels!")
            continue
        batch_errors = torch.norm(outputs - labels, dim=2)
        masked_error = (mask.view(-1, num_keypoints, 2).sum(dim=2)==2).float()

        # if torch.isnan(batch_errors) or torch.isinf(batch_errors):
        #     print("NaN or Inf detected in batch_errors!")
        #     continue
        
        error = batch_errors * masked_error
        total_error += error.sum().item()
        total_keypoints += masked_error.sum().item()
    

    if total_keypoints > 0:
        error = total_error / total_keypoints
        print(f'Average keypoint error: {error:.2f}')
    else:
        print("No valid test samples — error computation skipped.")
