"""
Full pipeline test - Tests the /predict endpoint with the trained model
"""
import os
import torch
from utils import preprocess_image, CNN

# Configuration
MODEL_PATH = "models/brain_tumor_cnn_test.pt"
TEST_IMAGE_PATH = "data/initial/Brain Tumor/Brain Tumor/Image1.jpg"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("TESTING PARTNER 3's /PREDICT ENDPOINT (Direct Python Test)")
print("=" * 70)
print()

# Step 1: Verify files exist
print("Step 1: Verifying files...")
print(f"  Model file: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    print("  ✓ Model file found")
else:
    print("  ✗ Model file NOT found - please train a model first")
    exit(1)

print(f"  Test image: {TEST_IMAGE_PATH}")
if os.path.exists(TEST_IMAGE_PATH):
    print("  ✓ Test image found")
else:
    print("  ✗ Test image NOT found")
    exit(1)
print()

# Step 2: Load the model
print("Step 2: Loading trained model...")
model = CNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("  ✓ Model loaded successfully")
print()

# Step 3: Preprocess the image (Partner 2's function)
print("Step 3: Preprocessing image (using Partner 2's function)...")
image_tensor = preprocess_image(TEST_IMAGE_PATH)
image_tensor = image_tensor.to(DEVICE)
print(f"  ✓ Image preprocessed - shape: {image_tensor.shape}")
print()

# Step 4: Make prediction
print("Step 4: Making prediction...")
with torch.no_grad():
    output = model(image_tensor)
    probability = output.squeeze().item()

predicted_class = 1 if probability >= 0.5 else 0

if predicted_class == 1:
    confidence = probability * 100
else:
    confidence = (1 - probability) * 100

if predicted_class == 1:
    interpretation = f"TUMOR DETECTED - High concern (probability: {probability:.2%})"
else:
    interpretation = f"NO TUMOR DETECTED - Low concern (probability: {probability:.2%})"

print("  ✓ Prediction complete!")
print()

# Step 5: Display results
print("=" * 70)
print("PREDICTION RESULTS")
print("=" * 70)
print(f"Image: {TEST_IMAGE_PATH}")
print()
print(f"Predicted Class:      {predicted_class} ({'Tumor' if predicted_class == 1 else 'No Tumor'})")
print(f"Probability:          {probability:.4f}")
print(f"Confidence:           {confidence:.2f}%")
print(f"Interpretation:       {interpretation}")
print("=" * 70)
print()

# Step 6: Test with a few more images
print("Testing with additional images...")
print()

test_images = [
    "data/initial/Brain Tumor/Brain Tumor/Image100.jpg",
    "data/initial/Brain Tumor/Brain Tumor/Image500.jpg",
    "data/initial/Brain Tumor/Brain Tumor/Image1000.jpg",
    "data/initial/Brain Tumor/Brain Tumor/Image2000.jpg",
]

for img_path in test_images:
    if os.path.exists(img_path):
        img_tensor = preprocess_image(img_path).to(DEVICE)
        with torch.no_grad():
            out = model(img_tensor)
            prob = out.squeeze().item()
        pred_class = 1 if prob >= 0.5 else 0
        class_name = "Tumor" if pred_class == 1 else "No Tumor"
        print(f"  {os.path.basename(img_path):20s} → {class_name:10s} (prob: {prob:.4f})")

print()
print("=" * 70)
print("✓ Full pipeline test complete!")
print()
print("This demonstrates that:")
print("  1. Partner 2's preprocessing function works correctly")
print("  2. Partner 1's trained model loads and runs")
print("  3. Partner 3's prediction logic is functional")
print()
print("Next step: Test via the /predict API endpoint")
print("=" * 70)
