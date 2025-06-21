import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        
def load_model():
    model = ConvNet().to(device)
    checkpoint = torch.load("checkpoint.pth")
    model.load_state_dict(checkpoint['weights'])
    model.eval()
    return model
