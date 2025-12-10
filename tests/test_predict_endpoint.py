"""
Test script for the /predict endpoint
Partner 3 deliverable - shows how to use the prediction API
"""

import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"


def test_predict_endpoint():
    """
    Example of how to call the /predict endpoint
    """

    # Example request payload
    payload = {
        "image_path": "/Users/kadengodinez/AI_MLOps_Team1_Project/AI_MLOps_Team1_Project/data/initial/Brain Tumor/Brain Tumor/Image1.jpg",
        "model_path": "/path/to/trained/model.pt",
        "model_type": "cnn"  # or "nn"
    }

    # Make POST request to /predict endpoint
    response = requests.post(f"{BASE_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        print("✓ Prediction successful!")
        print(json.dumps(result, indent=2))
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.json())


def test_health_check():
    """Test the health check endpoint"""
    response = requests.get(f"{BASE_URL}/health_check")
    print(f"Health check: {response.json()}")


if __name__ == "__main__":
    print("=" * 60)
    print("Brain Tumor Classification API - Prediction Test")
    print("=" * 60)

    print("\n1. Testing health check...")
    test_health_check()

    print("\n2. Testing prediction endpoint...")
    test_predict_endpoint()

    print("\n" + "=" * 60)
