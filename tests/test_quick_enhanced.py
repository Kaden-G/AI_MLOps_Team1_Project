"""
Quick test script for enhanced training endpoint - just 2 epochs
"""

import requests
import json
import os

API_URL = "http://127.0.0.1:8000"

# Get the absolute path to the dataset
dataset_path = os.path.abspath("data/initial")
save_path = os.path.abspath("models/quick_test_enhanced")
best_model_path = os.path.abspath("models/quick_test_enhanced_best")

# Training request with new parameters - ONLY 2 EPOCHS FOR QUICK TEST
train_request = {
    "dataset_path": dataset_path,
    "test_size": 0.2,
    "batch_size": 64,  # Larger batch for faster training
    "num_epochs": 2,   # Just 2 epochs for quick verification
    "save_path": save_path,
    "model_type": "cnn",
    "learning_rate": 0.001,
    "momentum": 0.9,
    "early_stopping_patience": 5,
    "random_seed": 42,
    "best_model_path": best_model_path
}

print("Testing Enhanced Training (2 epochs)")
print("=" * 80)

# Send training request with streaming response
response = requests.post(
    f"{API_URL}/train",
    json=train_request,
    stream=True
)

if response.status_code == 200:
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))

    print("\n" + "=" * 80)
    print("SUCCESS! All features working:")
    print("=" * 80)

    # Check if models were created
    if os.path.exists(save_path):
        print(f"✓ Current model: {save_path} ({os.path.getsize(save_path)/(1024*1024):.2f} MB)")
    if os.path.exists(best_model_path):
        print(f"✓ Best model: {best_model_path} ({os.path.getsize(best_model_path)/(1024*1024):.2f} MB)")
else:
    print(f"✗ Training failed: {response.status_code}")
    print(response.text)
