"""
Test script for enhanced training endpoint with comprehensive metrics,
early stopping, and best model checkpointing.
"""

import requests
import json
import os

API_URL = "http://127.0.0.1:8000"

def test_enhanced_training():
    """Test the enhanced training endpoint with all new features"""

    print("=" * 80)
    print("Testing Enhanced Training Endpoint")
    print("=" * 80)

    # Get the absolute path to the dataset
    dataset_path = os.path.abspath("data/initial")
    save_path = os.path.abspath("models/test_enhanced_training")
    best_model_path = os.path.abspath("models/test_enhanced_training_best")

    # Training request with new parameters
    train_request = {
        "dataset_path": dataset_path,
        "test_size": 0.2,
        "batch_size": 32,
        "num_epochs": 5,  # Small number for quick test
        "save_path": save_path,
        "model_type": "cnn",
        "learning_rate": 0.001,
        "momentum": 0.9,
        "early_stopping_patience": 3,  # Will stop if no improvement for 3 epochs
        "random_seed": 42,
        "best_model_path": best_model_path
    }

    print(f"\nTraining Request:")
    print(json.dumps(train_request, indent=2))
    print("\n" + "=" * 80)
    print("Training Progress (streaming):")
    print("=" * 80 + "\n")

    # Send training request with streaming response
    response = requests.post(
        f"{API_URL}/train",
        json=train_request,
        stream=True
    )

    if response.status_code == 200:
        # Stream and print the training progress
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'))

        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)

        # Check if models were created
        if os.path.exists(save_path):
            size_mb = os.path.getsize(save_path) / (1024 * 1024)
            print(f"\n✓ Current model saved: {save_path} ({size_mb:.2f} MB)")
        else:
            print(f"\n✗ Current model NOT found: {save_path}")

        if os.path.exists(best_model_path):
            size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
            print(f"✓ Best model saved: {best_model_path} ({size_mb:.2f} MB)")
        else:
            print(f"✗ Best model NOT found: {best_model_path}")

        return True
    else:
        print(f"\n✗ Training failed with status code: {response.status_code}")
        print(f"Response: {response.text}")
        return False


if __name__ == "__main__":
    # Test health check first
    try:
        health = requests.get(f"{API_URL}/health_check")
        if health.status_code == 200:
            print("✓ Server is running\n")
            test_enhanced_training()
        else:
            print(f"✗ Server health check failed: {health.status_code}")
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Is it running?")
        print("Start server with: uvicorn main:app --reload")
