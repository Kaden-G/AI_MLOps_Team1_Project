# 📋 Notebook Prerequisites

This document lists everything needed to run the Brain Tumor Classification notebooks after cloning the repository.

---

## ✅ Required Setup

### 1. **Clone the Repository**
```bash
git clone https://github.com/Kaden-G/AI_MLOps_Team1_Project.git
cd AI_MLOps_Team1_Project/AI_MLOps_Team1_Project
```

### 2. **Install Python Dependencies**
```bash
pip install requests
# Or install from requirements.txt if available
```

The notebooks will automatically install `requests` if not present.

### 3. **Ensure Dataset is Available**
The dataset should be in:
```
data/initial/Brain Tumor/Brain Tumor/
```

Expected structure:
```
data/
└── initial/
    ├── Brain Tumor/
    │   └── Brain Tumor/
    │       ├── Image1.jpg
    │       ├── Image2.jpg
    │       └── ... (3,762 images total)
    └── Brain Tumor.csv
```

### 4. **Start the API Server**
In a separate terminal, from the project root:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📓 Running the Notebooks

### **Option 1: DEMO_NOTEBOOK.ipynb**
For presentations and quick demos:
```bash
jupyter notebook notebooks/DEMO_NOTEBOOK.ipynb
```

- Trains a 2-epoch model (~2-3 minutes)
- Makes predictions on 5 test images
- Perfect for live demonstrations

### **Option 2: TESTING_NOTEBOOK_PROFESSOR.ipynb**
For comprehensive testing:
```bash
jupyter notebook notebooks/TESTING_NOTEBOOK_PROFESSOR.ipynb
```

- Uses existing 20-epoch model (or trains new one)
- Makes predictions on 7 test images
- Full system validation
- Displays confusion matrix

---

## 🔍 Automatic Checks

Both notebooks include automatic checks for:
- ✅ Python dependencies (auto-installs `requests` if needed)
- ✅ Server connectivity (tests localhost and 127.0.0.1)
- ✅ Model availability (searches for existing models)
- ✅ Dataset presence (verifies image paths)

---

## 🚀 Expected Behavior

### **On First Run:**
1. Notebook checks if server is running
2. Auto-installs `requests` library if needed
3. Searches for existing trained models
4. If no model found, prompts to train one (or skips training section)

### **Path Handling:**
- All paths use `os.path.abspath()` with relative paths
- Works on any operating system (Windows, macOS, Linux)
- No hardcoded user-specific paths

---

## ⚠️ Common Issues

### **Issue 1: "Could not connect to server"**
**Solution:** Make sure the server is running in a separate terminal:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### **Issue 2: "Model file not found"**
**Solution:** Either:
- Run the training section in the notebook (creates a new model)
- Copy a pre-trained model to the `models/` directory

### **Issue 3: "Dataset not found"**
**Solution:** Ensure the dataset is in the correct location:
```
data/initial/Brain Tumor/Brain Tumor/
```

---

## 📊 What You'll See

### **DEMO_NOTEBOOK.ipynb Output:**
- Training progress with comprehensive metrics
- Final validation metrics (Accuracy, Precision, Recall, F1)
- Prediction results for 5 test images
- Links to Swagger UI documentation

### **TESTING_NOTEBOOK_PROFESSOR.ipynb Output:**
- Model search results
- Training metrics (if training is run)
- Confusion matrix
- Prediction results for 7 test images
- Complete system validation summary

---

## 🎯 Success Criteria

You'll know everything is working correctly when:
- ✅ Server health check passes
- ✅ Training completes (or existing model is found)
- ✅ All predictions return valid results
- ✅ No connection errors or file not found errors

---

## 💡 Tips

1. **Use System Python Kernel**: If having connection issues, switch from ipykernel to system Python in Jupyter
2. **Check Server Logs**: Monitor the uvicorn terminal for any API errors
3. **Verify Dataset**: The notebooks expect exactly 3,762 images in the dataset
4. **Model Cache**: The API caches loaded models for faster predictions

---

## ✅ Portable & Ready

Both notebooks are:
- ✅ **Portable**: No hardcoded user-specific paths
- ✅ **Self-Contained**: Auto-installs dependencies
- ✅ **Cross-Platform**: Works on Windows, macOS, Linux
- ✅ **Well-Documented**: Clear instructions and error messages

**Any user can clone the repo and run these notebooks immediately!** 🚀
