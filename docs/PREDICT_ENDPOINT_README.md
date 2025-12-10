# `/predict` Endpoint Documentation
**Partner 3 Deliverable**

## Overview

The `/predict` endpoint is the inference layer of the Brain Tumor Classification API. It loads a trained model and classifies brain MRI images as either containing a tumor (class 1) or not containing a tumor (class 0).

---

## Quick Start

### 1. Start the Server

```bash
uvicorn main:app --reload
```

The server will start at `http://localhost:8000`

### 2. Access Interactive Documentation

Once the server is running:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### 3. Make a Prediction

Using `curl`:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/brain_mri.jpg",
    "model_path": "/path/to/trained_model.pt",
    "model_type": "cnn"
  }'
```

Using Python `requests`:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "image_path": "/path/to/brain_mri.jpg",
        "model_path": "/path/to/trained_model.pt",
        "model_type": "cnn"
    }
)

print(response.json())
```

---

## API Reference

### Endpoint Details

- **URL:** `/predict`
- **Method:** `POST`
- **Content-Type:** `application/json`
- **Response Type:** `application/json`

### Request Schema

```json
{
  "image_path": "string (required)",
  "model_path": "string (required)",
  "model_type": "string (optional, default: 'cnn')"
}
```

#### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_path` | string | Yes | Absolute path to the brain MRI image file (.jpg, .png) |
| `model_path` | string | Yes | Absolute path to the trained model file (.pt, .pth) |
| `model_type` | string | No | Model architecture type: `"cnn"` or `"nn"` (default: `"cnn"`) |

### Response Schema

```json
{
  "predicted_class": 0 or 1,
  "probability": 0.0 to 1.0,
  "confidence_percentage": 0.0 to 100.0,
  "interpretation": "string"
}
```

#### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `predicted_class` | integer | 0 = No Tumor, 1 = Tumor |
| `probability` | float | Raw model output (probability of tumor presence) |
| `confidence_percentage` | float | Confidence level as percentage |
| `interpretation` | string | Human-readable explanation of the prediction |

---

## How It Works

### 1. Input Validation
- Checks if image file exists
- Checks if model file exists
- Returns 404 error if either is missing

### 2. Model Loading
- Instantiates the correct model architecture (CNN or NN)
- Loads trained weights from the specified model file
- Sets model to evaluation mode
- Moves model to available device (GPU if available, else CPU)

### 3. Image Preprocessing
- Uses Partner 2's `preprocess_image()` function
- Loads image and converts to grayscale
- Reshapes to (1, 1, 240, 240) tensor
- Converts to appropriate data type (float32)

### 4. Inference
- Runs forward pass through the model
- Extracts probability from sigmoid output
- Applies 0.5 threshold to determine class
- Calculates confidence percentage

### 5. Response Formatting
- Packages results into structured JSON
- Generates human-readable interpretation
- Returns HTTP 200 with prediction details

---

## Error Handling

### 404 - File Not Found

**Scenario:** Image or model file doesn't exist

```json
{
  "detail": "Image file not found: /path/to/missing/image.jpg"
}
```

### 500 - Prediction Failed

**Scenario:** Model loading error, incompatible dimensions, etc.

```json
{
  "detail": "Prediction failed: [error message]"
}
```

---

## Integration with Other Partners

### From Partner 1 (Model Training)
- **Receives:** Trained model file (.pt)
- **Expects:** Model saved with `torch.save(model.state_dict(), path)`
- **Requires:** Model architecture type (CNN or NN)

### From Partner 2 (Preprocessing)
- **Uses:** `preprocess_image()` function from `utils.py`
- **Expects:** Function returns tensor of shape (1, 1, 240, 240)
- **Requires:** Same preprocessing pipeline as training

### For Partner 4 (Demo/Slides)
- **Provides:** Fully functional prediction endpoint
- **Provides:** JSON response examples (see `PARTNER4_EXAMPLE_OUTPUTS.md`)
- **Provides:** Swagger/OpenAPI documentation

---

## Testing the Endpoint

### Test with Sample Data

```python
# test_predict_endpoint.py
import requests

# Ensure server is running: uvicorn main:app --reload

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "image_path": "data/initial/Brain Tumor/Brain Tumor/Image1.jpg",
        "model_path": "models/trained_model.pt",
        "model_type": "cnn"
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Class: {result['predicted_class']}")
    print(f"Probability: {result['probability']:.2%}")
    print(f"Interpretation: {result['interpretation']}")
else:
    print(f"Error: {response.json()}")
```

### Expected Output

```
Class: 1
Probability: 87.45%
Interpretation: TUMOR DETECTED - High concern (probability: 87.45%)
```

---

## Swagger Documentation

Once the server is running, visit `http://localhost:8000/docs` to access interactive API documentation.

### Features:
- **Try It Out** button for live testing
- Automatic request/response schema display
- Example values for all fields
- Error response documentation

### Taking Screenshots for Partner 4:

1. Navigate to http://localhost:8000/docs
2. Expand the `/predict` endpoint
3. Click "Try it out"
4. Fill in example values
5. Click "Execute"
6. Screenshot the request and response panels

---

## Model Output Interpretation

### Understanding Probability Values

| Probability Range | Interpretation | Recommended Action |
|-------------------|----------------|-------------------|
| 0.90 - 1.00 | Very high tumor likelihood | Immediate clinical review |
| 0.70 - 0.89 | High tumor likelihood | Clinical review recommended |
| 0.50 - 0.69 | Moderate tumor likelihood | Expert review required |
| 0.30 - 0.49 | Low tumor likelihood | May require second opinion |
| 0.10 - 0.29 | Very low tumor likelihood | Likely negative, follow-up as needed |
| 0.00 - 0.09 | Extremely low tumor likelihood | Negative screening result |

**Clinical Note:** In medical imaging, false negatives (missing a tumor) are more dangerous than false positives (flagging a healthy scan). The 0.5 threshold balances sensitivity and specificity, but clinical context should always guide final decisions.

---

## Performance Considerations

### Response Time
- **Image preprocessing:** ~50-200ms
- **Model loading (cached):** ~100-500ms
- **Inference (CNN):** ~10-50ms
- **Total:** ~200-800ms per request

### Optimization Tips
1. **Model Caching:** Load model once at startup instead of per-request
2. **Batch Processing:** Process multiple images in one request
3. **GPU Acceleration:** Use CUDA if available
4. **Model Optimization:** Convert to TorchScript or ONNX for faster inference

---

## Security Considerations

⚠️ **Current Implementation - Development Only**

The current endpoint has NO security measures and should NOT be used in production:

- No authentication/authorization
- Accepts arbitrary file paths (path traversal risk)
- No rate limiting
- No input sanitization beyond existence checks
- No model signature verification

### Recommended for Production:
1. Add API key authentication
2. Validate and sanitize all file paths
3. Implement rate limiting
4. Add request logging
5. Use secure model storage
6. Add HTTPS/TLS
7. Implement CORS policies

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'fastapi'"
**Solution:** Install dependencies
```bash
pip install fastapi uvicorn torch torchvision scikit-image pandas
# OR using uv
uv sync
```

### Issue: "Model file not found"
**Solution:** Ensure you've trained a model first using the `/train` endpoint
```bash
curl -X POST "http://localhost:8000/train" -H "Content-Type: application/json" -d '{...}'
```

### Issue: "Image dimension mismatch"
**Solution:** Ensure images are 240x240 pixels or will be resized correctly by preprocessing

### Issue: "CUDA out of memory"
**Solution:** The model will automatically fall back to CPU if GPU memory is insufficient

---

## Files Modified/Created

### Modified Files
- `main.py` - Added `/predict` endpoint, PredictRequest, PredictResponse models
- `utils.py` - Added `preprocess_image()` function

### Created Files
- `test_predict_endpoint.py` - Test script
- `PARTNER4_EXAMPLE_OUTPUTS.md` - JSON examples for slides
- `PREDICT_ENDPOINT_README.md` - This documentation

---

## Next Steps

1. ✅ Train a model using `/train` endpoint (Partner 1)
2. ✅ Ensure preprocessing is consistent (Partner 2)
3. ✅ Test `/predict` with trained model
4. ⬜ Generate Swagger screenshots (Partner 4)
5. ⬜ Integrate into demo workflow (Partner 4)
6. ⬜ Add to slide deck (Partner 4)

---

## Contact

**Partner 3 - Prediction Endpoint Owner**

For questions about:
- Endpoint functionality
- Request/response formats
- Integration issues
- Error handling

Reach out to Partner 3.
