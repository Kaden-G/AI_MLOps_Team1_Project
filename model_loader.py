"""
Model Loader Module - One-time Model Loading with Caching

This module provides a cached model loading function to avoid reloading
the model on every prediction request. The model is loaded once and cached
in memory for subsequent requests.

Author: Partner 3 - Production API
"""

from functools import lru_cache
import torch
import torch.nn as nn
from utils import CNN, NN
from typing import Literal


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@lru_cache(maxsize=2)
def load_model(model_path: str, model_type: Literal["cnn", "nn"] = "cnn") -> nn.Module:
    """
    Load and cache a trained model. Uses LRU cache to avoid reloading on every request.

    This function is called once per unique (model_path, model_type) combination,
    and subsequent calls with the same parameters return the cached model instance.

    Args:
        model_path: Path to the trained model weights file (.pt or .pth)
        model_type: Type of model architecture ("cnn" or "nn")

    Returns:
        Loaded PyTorch model in evaluation mode

    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails

    Example:
        >>> model = load_model("/path/to/model.pt", "cnn")
        >>> # Subsequent calls with same parameters return cached model:
        >>> model2 = load_model("/path/to/model.pt", "cnn")  # Returns cached instance
    """
    print(f"🔄 Loading model from {model_path} (model_type={model_type})")

    # Initialize the appropriate model architecture
    if model_type == "cnn":
        model = CNN().to(DEVICE)
    elif model_type == "nn":
        model = NN().to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Expected 'cnn' or 'nn'")

    # Load trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights from {model_path}: {str(e)}")

    # Set to evaluation mode (disables dropout, batch norm training mode, etc.)
    model.eval()

    print(f"✅ Model loaded successfully and cached")

    return model


def clear_model_cache():
    """
    Clear the model cache. Useful for freeing memory or forcing model reload.

    Example:
        >>> clear_model_cache()
        >>> # Next load_model() call will reload from disk
    """
    load_model.cache_clear()
    print("🗑️  Model cache cleared")


def get_cache_info():
    """
    Get information about the model cache (hits, misses, size).

    Returns:
        Cache info namedtuple with hits, misses, maxsize, currsize

    Example:
        >>> info = get_cache_info()
        >>> print(f"Cache hits: {info.hits}, misses: {info.misses}")
    """
    return load_model.cache_info()
