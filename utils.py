from torch.utils.data import Dataset
import pandas as pd
from skimage import io
from torch import from_numpy
from torch import float32
from torch import tensor
from torchvision.transforms import Grayscale
import torch.nn as nn
import torch.nn.functional as F
import torch


class BrainTumorDataset(Dataset):
    def __init__(self, file_paths: list[str], labels: list[int]):
        self.file_paths = file_paths
        self.labels = labels


    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx: int):
        image = io.imread(self.file_paths[idx])
        label = self.labels[idx]



        return Grayscale()(from_numpy(image).to(float32).reshape((3, 240, 240))), tensor(label).to(float32)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 8, 5)
        self.fc1 = nn.Linear(968, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x
    


