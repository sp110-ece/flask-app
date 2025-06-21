import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class PoseDataset(Dataset):
    def __init__(self, img_dir, csv_path, img_size=(256, 256)):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (width, height)

        # Extract keypoints
        keypoints = []
        mask = []
        for i in range(1, len(row), 3):  # skip 'image' column, then read x,y,v triplets
            x = row[i]
            y = row[i+1]
            v = row[i+2]

            
            if v > 0:
                norm_x = (x / original_size[0])
                norm_y = (y / original_size[1])
                mask.append([1, 1])
            else:
                norm_x, norm_y = 0, 0 
                mask.append([0, 0])
                

            keypoints.append([norm_x, norm_y])

        keypoints = torch.tensor(keypoints, dtype=torch.float32).flatten()
        mask = torch.tensor(mask, dtype=torch.float32).flatten()
        image = self.transform(image)

        return image, keypoints, mask
