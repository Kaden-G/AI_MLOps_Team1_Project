# Quick Start Guide - Enhanced Brain Tumor Classification API

## 🚀 Getting Started

### 1. Start the Server

```bash
uvicorn main:app --reload
```

Server will be available at: `http://localhost:8000`
Swagger docs at: `http://localhost:8000/docs`

---

## 🎯 API Endpoints

### **Health Check**
```bash
GET http://localhost:8000/health_check
```

### **Train Model** (Streaming)
```bash
POST http://localhost:8000/train
```

### **Predict** (Get Tumor Classification)
```bash
POST http://localhost:8000/predict
```

---

## 📝 Quick Examples

### **Example 1: Train a Model with All Features**

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "/path/to/data/initial",
    "test_size": 0.2,
    "batch_size": 64,
    "num_epochs": 20,
    "save_path": "/path/to/models/my_model",
    "best_model_path": "/path/to/models/my_model_best",
    "model_type": "cnn",
    "learning_rate": 0.001,
    "momentum": 0.9,
    "early_stopping_patience": 5,
    "random_seed": 42
  }'
```

**What you get:**
- ✅ Real-time streaming of training progress
- ✅ Comprehensive metrics (Loss, Accuracy, Precision, Recall, F1) for each epoch
- ✅ Two saved models: current (`my_model`) and best (`my_model_best`)
- ✅ Early stopping if no improvement for 5 epochs
- ✅ Reproducible results (seed=42)

### **Example 2: Make a Prediction**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.jpg",
    "model_path": "/path/to/models/my_model_best",
    "model_type": "cnn"
  }'
```

**Response:**
```json
{
  "predicted_class": 1,
  "probability": 0.8523,
  "confidence_percentage": 85.23,
  "interpretation": "TUMOR DETECTED - High concern (probability: 85.23%)"
}
```

---

## 🐍 Python Examples

### **Training with Python**

```python
import requests

# Training request
train_request = {
    "dataset_path": "/path/to/data/initial",
    "test_size": 0.2,
    "batch_size": 64,
    "num_epochs": 20,
    "save_path": "/path/to/models/my_model",
    "best_model_path": "/path/to/models/my_model_best",
    "model_type": "cnn",
    "learning_rate": 0.001,
    "momentum": 0.9,
    "early_stopping_patience": 5,
    "random_seed": 42
}

# Send request with streaming
response = requests.post(
    "http://localhost:8000/train",
    json=train_request,
    stream=True
)

# Print training progress in real-time
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

### **Prediction with Python**

```python
import requests

# Prediction request
predict_request = {
    "image_path": "/path/to/image.jpg",
    "model_path": "/path/to/models/my_model_best",
    "model_type": "cnn"
}

# Send request
response = requests.post(
    "http://localhost:8000/predict",
    json=predict_request
)

# Get result
result = response.json()

print(f"Class: {result['predicted_class']}")
print(f"Probability: {result['probability']:.4f}")
print(f"Confidence: {result['confidence_percentage']:.2f}%")
print(f"Interpretation: {result['interpretation']}")
```

---

## ⚙️ Configuration Options

### **Training Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_path` | string | **required** | Path to dataset directory |
| `test_size` | float | 0.2 | Validation split (0.0-1.0) |
| `batch_size` | int | 32 | Training batch size |
| `num_epochs` | int | 10 | Maximum epochs to train |
| `save_path` | string | **required** | Path to save current model |
| `best_model_path` | string | `{save_path}_best` | Path to save best model |
| `model_type` | enum | "cnn" | "cnn" or "nn" |
| `learning_rate` | float | 0.001 | Optimizer learning rate |
| `momentum` | float | 0.9 | SGD momentum |
| `early_stopping_patience` | int | 5 | Epochs to wait before stopping |
| `random_seed` | int | 42 | Random seed for reproducibility |

### **Prediction Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | string | **required** | Path to MRI image |
| `model_path` | string | **required** | Path to trained model (.pt) |
| `model_type` | enum | "cnn" | "cnn" or "nn" |

---

## 📊 Understanding the Output

### **Training Output**

```
Set random seed to 42 for reproducibility
Training starting with 3009 training samples, 753 validation samples
Learning rate: 0.001, Momentum: 0.9
Early stopping patience: 5 epochs

EPOCH 0 ----------------------------
Training Metrics - Loss: 0.0111, Accuracy: 0.4483, Precision: 0.4483, Recall: 1.0000, F1: 0.6191
Validation Metrics - Loss: 0.0111, Accuracy: 0.4436, Precision: 0.4436, Recall: 1.0000, F1: 0.6145
Saved current model to: /path/to/models/my_model
✓ New best model! Saved to: /path/to/models/my_model_best (Val Loss: 0.0111)
```

**Metrics Explained:**
- **Loss**: Lower is better (tracks improvement for early stopping)
- **Accuracy**: % of correct predictions
- **Precision**: Of predicted tumors, % that were actually tumors
- **Recall**: Of actual tumors, % that were detected (MOST IMPORTANT for medical use)
- **F1**: Harmonic mean of precision and recall

### **Prediction Output**

```json
{
  "predicted_class": 1,              // 0 = No Tumor, 1 = Tumor
  "probability": 0.8523,             // Raw model output (0.0-1.0)
  "confidence_percentage": 85.23,    // Confidence as percentage
  "interpretation": "TUMOR DETECTED - High concern (probability: 85.23%)"
}
```

---

## 🧪 Test the API

### **Using Swagger UI**

1. Open browser: `http://localhost:8000/docs`
2. Click "Try it out" on any endpoint
3. Fill in parameters
4. Click "Execute"
5. View results

### **Using Test Scripts**

```bash
# Quick 2-epoch training test
python3 test_quick_enhanced.py

# Test prediction endpoint
python3 test_predict_with_enhanced_model.py

# Full pipeline test
python3 test_full_pipeline.py
```

---

## 🎯 Best Practices

### **For Training**

1. **Use best model for production**: Always use `best_model_path` for predictions
2. **Start with lower epochs**: Test with 2-5 epochs before running 20+
3. **Monitor recall**: For medical use, prioritize high recall (detect tumors)
4. **Use early stopping**: Set patience to 5-10 epochs to avoid overfitting

### **For Prediction**

1. **Check file paths**: Ensure image and model files exist
2. **Use same model type**: Match architecture used during training
3. **Interpret confidence**: Higher confidence = more certain prediction

---

## ⚠️ Troubleshooting

### **Error: "Image file not found"**
- Check that `image_path` is absolute path
- Verify file exists: `ls /path/to/image.jpg`

### **Error: "Model file not found"**
- Check that `model_path` is absolute path
- Verify file exists: `ls /path/to/model.pt`

### **Training not improving**
- Try different `learning_rate` (0.01, 0.001, 0.0001)
- Increase `num_epochs`
- Adjust `batch_size` (32, 64, 128)

### **Server not starting**
- Check port 8000 is available: `lsof -i :8000`
- Kill existing server: `pkill -f uvicorn`
- Restart: `uvicorn main:app --reload`

---

## 📚 More Information

- **Full Documentation**: See `COMPLETE_IMPLEMENTATION_SUMMARY.md`
- **Partner 3 Deliverables**: See `PARTNER3_DELIVERABLES.md`
- **API Reference**: See `PREDICT_ENDPOINT_README.md`
- **Swagger Docs**: `http://localhost:8000/docs` (when server is running)

---

## 🎉 You're Ready!

The enhanced API is now ready for:
- ✅ Training models with comprehensive metrics
- ✅ Early stopping and best model checkpointing
- ✅ Making predictions on new images
- ✅ Reproducible results with seeding
- ✅ Production deployment

**Happy coding!** 🚀
