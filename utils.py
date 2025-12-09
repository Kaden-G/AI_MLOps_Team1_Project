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
from torchvision.transforms import v2
from copy import deepcopy


class BrainTumorDataset(Dataset):
    def __init__(self, file_paths: list[str], labels: list[int]):
        self.file_paths = file_paths
        self.labels = labels

        self.transforms_each_image = [0 for x in self.file_paths]

        self.original_length = len(self.labels)

        self.transforms = [
            v2.Compose([v2.Grayscale()]),
            v2.Compose([
                v2.Grayscale(),
                # v2.RandomResizedCrop((240, 240), antialias=True,),
                v2.GaussianNoise(sigma=0.05),
                # v2.RandomAdjustSharpness(1.2, p=0.5)
            ]),
        ]

        self.file_paths.extend(deepcopy(self.file_paths))
        self.labels.extend(deepcopy(self.labels))
        self.transforms_each_image.extend([1 for x in range(0, self.original_length)])


    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx: int):
        image = io.imread(self.file_paths[idx])
        label = self.labels[idx]

        image = from_numpy(image).to(float32).reshape((3, 240, 240))

        return self.transforms[self.transforms_each_image[idx]](image), tensor(label).to(float32)


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(240 * 240, 3_600)
        self.linear2 = nn.Linear(3_600, 900)
        self.linear3 = nn.Linear(900, 225)
        self.linear4 = nn.Linear(225, 45)
        self.linear5 = nn.Linear(45, 1)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.sigmoid(self.linear5(x))
        return x
    

class CNN(nn.Module):
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


