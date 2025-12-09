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
import numpy as np
import random
from PIL import Image
from typing import Union

# Deterministic seeding for reproducibility (Partner 2 requirement)
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def preprocess_image(image_input: Union[str, Image.Image], normalize: bool = True) -> torch.Tensor:
    """
    Partner 2's preprocessing function - converts image to model-ready tensor.

    This function applies the same preprocessing pipeline used during training:
    1. Load image from file path OR accept PIL Image directly
    2. Convert to tensor and reshape to (3, 240, 240)
    3. Apply Grayscale transformation -> (1, 240, 240)
    4. Normalize pixel values to [0, 1] range (if normalize=True)
    5. Add batch dimension -> (1, 1, 240, 240)

    Args:
        image_input: Either a file path (str) or a PIL Image object
        normalize: Whether to normalize pixel values to [0, 1] (default: True)

    Returns:
        Preprocessed tensor of shape (1, 1, 240, 240) ready for model input

    Examples:
        >>> # From file path (original usage)
        >>> tensor = preprocess_image("/path/to/image.jpg")

        >>> # From PIL Image (new usage for direct upload)
        >>> from PIL import Image
        >>> img = Image.open("/path/to/image.jpg")
        >>> tensor = preprocess_image(img)
    """
    # Load image (handle both file paths and PIL Images)
    if isinstance(image_input, str):
        # File path provided - load using skimage
        image = io.imread(image_input)
    elif isinstance(image_input, Image.Image):
        # PIL Image provided - convert to numpy array
        image = np.array(image_input)
    else:
        raise TypeError(f"image_input must be a file path (str) or PIL Image, got {type(image_input)}")

    # Convert to tensor and reshape to (3, 240, 240) then apply Grayscale
    # This matches the training preprocessing pipeline
    image_tensor = Grayscale()(from_numpy(image).to(float32).reshape((3, 240, 240)))

    # Normalize pixel values to [0, 1] range (Partner 2 requirement)
    if normalize:
        # Images are typically in [0, 255] range, normalize to [0, 1]
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

    # Add batch dimension: (1, 240, 240) -> (1, 1, 240, 240)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


class BrainTumorDataset(Dataset):
    def __init__(self, file_paths: list[str], labels: list[int], normalize: bool = True):
        """
        Brain Tumor MRI Dataset

        Args:
            file_paths: List of paths to image files
            labels: List of labels (0 = no tumor, 1 = tumor)
            normalize: Whether to normalize pixel values to [0, 1] (default: True)
        """
        self.file_paths = file_paths
        self.labels = labels
        self.normalize = normalize



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        image = io.imread(self.file_paths[idx])
        label = self.labels[idx]

        # Convert to tensor and apply grayscale
        image_tensor = Grayscale()(from_numpy(image).to(float32).reshape((3, 240, 240)))

        # Normalize pixel values to [0, 1] range (Partner 2 requirement)
        if self.normalize and image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        return image_tensor, tensor(label).to(float32)


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


