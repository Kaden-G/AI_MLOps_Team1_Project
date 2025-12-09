"""
Test the /predict endpoint with the enhanced trained model
"""

import requests
import json
import os

API_URL = "http://127.0.0.1:8000"

def test_predict_endpoint():
    """Test prediction with the enhanced model"""

    # Get absolute paths
    model_path = os.path.abspath("models/quick_test_enhanced_best")
    dataset_path = os.path.abspath("data/initial/Brain Tumor/Brain Tumor")

    # Get some test images
    test_images = [
        os.path.join(dataset_path, "Image1.jpg"),
        os.path.join(dataset_path, "Image100.jpg"),
        os.path.join(dataset_path, "Image500.jpg"),
        os.path.join(dataset_path, "Image1000.jpg"),
        os.path.join(dataset_path, "Image2000.jpg"),
    ]

    print("=" * 80)
    print("Testing /predict Endpoint with Enhanced Model")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    print("=" * 80)

    for i, image_path in enumerate(test_images, 1):
        print(f"\nTest {i}: {os.path.basename(image_path)}")
        print("-" * 80)

        if not os.path.exists(image_path):
            print(f"✗ Image not found: {image_path}")
            continue

        predict_request = {
            "image_path": image_path,
            "model_path": model_path,
            "model_type": "cnn"
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=predict_request)

            if response.status_code == 200:
                result = response.json()
                print(f"✓ Prediction successful!")
                print(f"  Predicted Class: {result['predicted_class']} ({'TUMOR' if result['predicted_class'] == 1 else 'NO TUMOR'})")
                print(f"  Probability: {result['probability']:.4f}")
                print(f"  Confidence: {result['confidence_percentage']:.2f}%")
                print(f"  Interpretation: {result['interpretation']}")
            else:
                print(f"✗ Prediction failed: {response.status_code}")
                print(f"  Error: {response.text}")

        except Exception as e:
            print(f"✗ Exception: {str(e)}")

    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Test health check first
    try:
        health = requests.get(f"{API_URL}/health_check")
        if health.status_code == 200:
            print("✓ Server is running\n")
            test_predict_endpoint()
        else:
            print(f"✗ Server health check failed: {health.status_code}")
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Is it running?")
        print("Start server with: uvicorn main:app --reload")
