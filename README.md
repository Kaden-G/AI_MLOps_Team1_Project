# 🧠 Brain Tumor Classification API

**Complete ML Pipeline with Enhanced Training, Preprocessing, and Prediction**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## 📋 Project Overview

Brain tumor classification system using MRI images with:
- **Dataset**: 3,762 brain MRI scans
- **Model**: CNN (4 conv + 4 FC layers)
- **Performance**: 73.7% accuracy, **90.2% recall**
- **API**: FastAPI with `/train` and `/predict` endpoints

**Medical Significance**: 90% recall means only 10% of tumors are missed (critical for screening).

---

## 🚀 Quick Start

### **1. Start Server**
```bash
uvicorn main:app --reload
```

### **2. Train Model**
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "/path/to/data/initial",
    "num_epochs": 20,
    "batch_size": 64,
    "save_path": "/path/to/models/my_model",
    "best_model_path": "/path/to/models/my_model_best"
  }'
```

### **3. Make Prediction**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.jpg",
    "model_path": "/path/to/models/my_model_best",
    "model_type": "cnn"
  }'
```

### **4. View Documentation**
```bash
open http://localhost:8000/docs
```

**For detailed instructions**, see [`docs/QUICK_START_GUIDE.md`](docs/QUICK_START_GUIDE.md)

---

## 📁 Project Structure

```
AI_MLOps_Team1_Project/
├── docs/                    # 📚 All documentation (11 files, 4000+ lines)
│   ├── README.md           # Documentation index
│   ├── COMPLETE_IMPLEMENTATION_SUMMARY.md
│   ├── QUICK_START_GUIDE.md
│   ├── DEMO_WALKTHROUGH.md
│   └── ...
├── notebooks/              # 📓 Jupyter notebooks
│   ├── DEMO_NOTEBOOK.ipynb              # Quick demo (2 epochs)
│   └── TESTING_NOTEBOOK_PROFESSOR.ipynb # Full testing (20 epochs)
├── tests/                  # 🧪 Test scripts
│   ├── test_quick_enhanced.py
│   ├── test_predict_with_enhanced_model.py
│   └── ...
├── models/                 # 🤖 Trained models (.gitignored)
├── data/                   # 📊 Dataset (.gitignored)
├── main.py                 # 🚀 FastAPI application
├── utils.py                # 🛠️ Utility functions
├── Dockerfile              # 🐳 Container configuration
└── README.md               # 📖 This file
```

---

## ✨ Key Features

### **Partner 1: Enhanced Training**
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1 Score
- ✅ **Early Stopping**: Configurable patience (default: 5 epochs)
- ✅ **Best Model Checkpointing**: Saves best model separately
- ✅ **Streaming Progress**: Real-time training updates

### **Partner 2: Robust Preprocessing**
- ✅ **Deterministic Seeding**: Reproducible results (seed=42)
- ✅ **Pixel Normalization**: Scales to [0, 1] range
- ✅ **Reusable Functions**: `set_seed()`, `preprocess_image()`

### **Partner 3: Production API**
- ✅ **`/predict` Endpoint**: Complete inference pipeline
- ✅ **Structured Responses**: Class, probability, confidence, interpretation
- ✅ **Error Handling**: Comprehensive validation
- ✅ **Swagger Docs**: Interactive API documentation

---

## 📊 Model Performance

### **Production Model (20 Epochs)**

| Metric | Value | Significance |
|--------|-------|-------------|
| **Validation Accuracy** | 73.7% | Overall correctness |
| **Validation Recall** | **90.2%** | 90% of tumors detected |
| **Validation Precision** | 80.1% | 80% of predictions correct |
| **Validation F1 Score** | ~84.8% | Balanced metric |

**Medical Context**: High recall (90%) minimizes missed tumors (false negatives) - critical for medical screening.

---

## 🎬 Demo & Testing

### **For Presentation:**
```bash
# Quick demo with Jupyter notebook
jupyter notebook notebooks/DEMO_NOTEBOOK.ipynb
```
- Training: 2 epochs (~2-3 minutes)
- See: [`docs/DEMO_WALKTHROUGH.md`](docs/DEMO_WALKTHROUGH.md)
- Cheat sheet: [`docs/DEMO_CHEAT_SHEET.md`](docs/DEMO_CHEAT_SHEET.md)

### **For Professor Testing:**
```bash
# Full testing notebook
jupyter notebook notebooks/TESTING_NOTEBOOK_PROFESSOR.ipynb
```
- Uses pre-trained 20-epoch model
- Comprehensive evaluation
- Multiple test images

### **Command-Line Testing:**
```bash
# Quick test (2 epochs)
python tests/test_quick_enhanced.py

# Prediction test
python tests/test_predict_with_enhanced_model.py

# Full pipeline test
python tests/test_full_pipeline.py
```

---

## 📚 Documentation

All documentation is in the [`docs/`](docs/) directory:

| Document | Purpose |
|----------|---------|
| [COMPLETE_IMPLEMENTATION_SUMMARY.md](docs/COMPLETE_IMPLEMENTATION_SUMMARY.md) | Full implementation details (800+ lines) |
| [QUICK_START_GUIDE.md](docs/QUICK_START_GUIDE.md) | Quick reference for API usage |
| [DEMO_WALKTHROUGH.md](docs/DEMO_WALKTHROUGH.md) | Step-by-step demo guide |
| [DEMO_CHEAT_SHEET.md](docs/DEMO_CHEAT_SHEET.md) | Quick reference for presentation |
| [PREDICT_ENDPOINT_README.md](docs/PREDICT_ENDPOINT_README.md) | API documentation |
| [WORK_COMPLETE_SUMMARY.md](docs/WORK_COMPLETE_SUMMARY.md) | Project completion status |

**Total**: 4,000+ lines of comprehensive documentation

---

## 🔧 API Endpoints

### **Health Check**
```
GET /health_check
```
Returns: `{"status": "ok"}`

### **Training**
```
POST /train
```
**Request:**
```json
{
  "dataset_path": "/path/to/data",
  "num_epochs": 20,
  "batch_size": 64,
  "save_path": "/path/to/model",
  "best_model_path": "/path/to/model_best",
  "learning_rate": 0.001,
  "momentum": 0.9,
  "early_stopping_patience": 5,
  "random_seed": 42
}
```

**Response**: Streaming training progress with metrics

### **Prediction**
```
POST /predict
```
**Request:**
```json
{
  "image_path": "/path/to/image.jpg",
  "model_path": "/path/to/model",
  "model_type": "cnn"
}
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

**Interactive Docs**: `http://localhost:8000/docs`

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t brain-tumor-api .

# Run
docker run -p 8000:8000 brain-tumor-api
```

See [`Dockerfile`](Dockerfile) for configuration.

---

## 🧪 Testing

All test scripts are in [`tests/`](tests/):

```bash
# Quick training test (2 epochs)
python tests/test_quick_enhanced.py

# Prediction endpoint test
python tests/test_predict_with_enhanced_model.py

# Full pipeline test
python tests/test_full_pipeline.py

# Enhanced training test
python tests/test_enhanced_training.py
```

---

## 🛠️ Technologies

- **Backend**: FastAPI 0.100+
- **ML Framework**: PyTorch 2.0+
- **Image Processing**: scikit-image
- **Data**: pandas, scikit-learn
- **Documentation**: Swagger/OpenAPI (auto-generated)
- **Container**: Docker

---

## 📈 Project Requirements Coverage

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| **Data Ingestion** | ✅ | 3,762 images loaded |
| **Preprocessing** | ✅ | Grayscale + normalization [0,1] |
| **Deterministic Seeding** | ✅ | Fixed seed (42) |
| **Model Training** | ✅ | CNN with comprehensive metrics |
| **Early Stopping** | ✅ | Configurable patience |
| **Best Model Checkpointing** | ✅ | Saves best separately |
| **Comprehensive Metrics** | ✅ | Acc, Prec, Recall, F1 |
| **Prediction API** | ✅ | `/predict` endpoint |
| **Error Handling** | ✅ | File validation, structured errors |
| **Documentation** | ✅ | 4,000+ lines |
| **Testing** | ✅ | 6 test scripts |
| **Deployment** | ✅ | Docker + FastAPI |

**Status**: ✅ **ALL REQUIREMENTS COMPLETED**

---

## 🎓 For Professor

1. **Start server**: `uvicorn main:app --reload`
2. **Open notebook**: `notebooks/TESTING_NOTEBOOK_PROFESSOR.ipynb`
3. **Run all cells**: Tests complete system
4. **View docs**: `http://localhost:8000/docs`

**Expected results:**
- 90% recall on validation set
- Comprehensive metrics displayed
- Multiple predictions successful

---

## 👥 Team Contributions

### **Partner 1 (Training)**
- Enhanced training with comprehensive metrics
- Data augmentation
- Confusion matrices
- Trained 20-epoch models

### **Partner 2 (Preprocessing)**
- Deterministic seeding
- Pixel normalization
- Reusable preprocessing functions

### **Partner 3 (API)**
- `/predict` endpoint implementation
- Error handling and validation
- API documentation
- Testing notebooks

### **Partner 4 (Demo)**
- Presentation materials
- Demo preparation
- Slideshow

---

## 📊 Dataset

- **Source**: Brain Tumor MRI Dataset
- **Total Images**: 3,762
- **Classes**: 2 (Tumor / No Tumor)
- **Split**: 80% train / 20% validation
- **Format**: JPG images (240×240)

---

## 🚀 Future Improvements

- More data augmentation (rotation, zoom, flip)
- Longer training (50+ epochs)
- Hyperparameter tuning (grid search)
- Different architectures (ResNet, VGG)
- Ensemble models
- Explainability (Grad-CAM)
- Cloud deployment (AWS, GCP)

---

## 📞 Support

- **Documentation**: See [`docs/`](docs/) directory
- **Quick Start**: [`docs/QUICK_START_GUIDE.md`](docs/QUICK_START_GUIDE.md)
- **API Reference**: [`docs/PREDICT_ENDPOINT_README.md`](docs/PREDICT_ENDPOINT_README.md)
- **Demo Guide**: [`docs/DEMO_WALKTHROUGH.md`](docs/DEMO_WALKTHROUGH.md)

---

## 📄 License

AI/MLOps Team Project - Johns Hopkins University

---

## 🎉 Project Status

**✅ COMPLETE AND READY FOR DEMO**

- All features implemented
- Comprehensive documentation
- Full testing coverage
- Production-ready API
- Demo materials prepared

**Ready for evaluation!** 🚀
