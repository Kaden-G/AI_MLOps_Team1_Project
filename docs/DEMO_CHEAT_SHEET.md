# 🎯 Demo Cheat Sheet - Quick Reference

**Keep this open during your demo for quick command lookups!**

---

## 🚀 Start Server (Terminal 1)

```bash
cd /Users/kadengodinez/AI_MLOps_Team1_Project/AI_MLOps_Team1_Project
uvicorn main:app --reload
```

---

## 🎓 Part 1: Training Demo (Terminal 2)

```bash
cd /Users/kadengodinez/AI_MLOps_Team1_Project/AI_MLOps_Team1_Project
python3 test_quick_enhanced.py
```

**What to highlight:**
- ✅ "Set random seed to 42" → **Reproducibility**
- ✅ Comprehensive metrics → **Accuracy, Precision, Recall, F1**
- ✅ "✓ New best model!" → **Best model checkpointing**
- ✅ Two models saved → **Current + Best**

**Key talking point:**
> "Notice we track 5 metrics for both training and validation. Recall is most important for medical use - we achieved 90% on the 20-epoch model."

---

## 🔮 Part 2: Prediction Demo (Terminal 2)

```bash
python3 test_predict_with_enhanced_model.py
```

**What to highlight:**
- ✅ Structured response → **Class, Probability, Confidence, Interpretation**
- ✅ Medical context → **Human-readable results**
- ✅ Preprocessing → **Automatic normalization applied**

**Key talking point:**
> "Each prediction returns a structured response with the predicted class, probability, confidence percentage, and a human-readable interpretation."

---

## 📚 Part 3: Swagger UI

**Browser:** `http://localhost:8000/docs`

**Demonstrate:**
1. ✅ `GET /health_check` → Click "Try it out" → Execute
2. ✅ `POST /train` → Show all configurable parameters
3. ✅ `POST /predict` → Show request/response schema
4. ✅ (OPTIONAL) Make a live prediction

**Live prediction example:**
```json
{
  "image_path": "/Users/kadengodinez/AI_MLOps_Team1_Project/AI_MLOps_Team1_Project/data/initial/Brain Tumor/Brain Tumor/Image1.jpg",
  "model_path": "/Users/kadengodinez/AI_MLOps_Team1_Project/AI_MLOps_Team1_Project/models/quick_test_enhanced_best",
  "model_type": "cnn"
}
```

---

## 💡 Key Talking Points

### **Partner 1 (Training)**
- ✅ Comprehensive metrics (Accuracy, Precision, Recall, F1)
- ✅ Early stopping (stops when not improving)
- ✅ Best model checkpointing (saves best separately)

### **Partner 2 (Preprocessing)**
- ✅ Deterministic seeding (reproducible results)
- ✅ Pixel normalization ([0, 1] range)
- ✅ Reusable functions (set_seed, preprocess_image)

### **Partner 3 (Prediction API)**
- ✅ Production-ready endpoint
- ✅ Error handling
- ✅ Structured responses

### **Medical Impact**
- ✅ 90% recall = only 10% missed tumors
- ✅ Critical for medical screening
- ✅ Production-ready with best practices

---

## 🛠️ Troubleshooting

**Server won't start:**
```bash
pkill -f uvicorn
uvicorn main:app --reload
```

**Check models exist:**
```bash
ls -lh models/
```

**Quick health check:**
```bash
curl http://localhost:8000/health_check
```

---

## 🎬 Demo Flow (12 minutes)

| Time | What |
|------|------|
| 0-1 min | Start server, explain project |
| 1-6 min | Run training, explain metrics |
| 6-9 min | Run predictions, show results |
| 9-12 min | Show Swagger UI, wrap up |

---

## 📊 Stats to Mention

- **Dataset**: 3,762 MRI images (80/20 train/val split)
- **Model**: CNN with 4 conv layers + 4 FC layers
- **Training**: 3,009 samples / Validation: 753 samples
- **Best Performance**: 73.7% accuracy, **90.2% recall** (20-epoch model)
- **Model Size**: ~2.3 MB
- **Documentation**: 1,000+ lines across 6 files

---

## 🎤 Opening Statement

> "Today I'm demonstrating our brain tumor classification API. We've implemented a complete ML pipeline with enhanced training metrics, deterministic preprocessing, and a production-ready prediction endpoint. Let me show you how it works."

---

## 🎤 Closing Statement

> "To summarize: we built a complete system with comprehensive metrics, early stopping, best model checkpointing, pixel normalization, deterministic seeding, and a production-ready API. The 20-epoch model achieves 90% recall, which is excellent for medical use. All code is on GitHub, fully documented, and ready for deployment."

---

## 📝 Commands at a Glance

```bash
# Start
uvicorn main:app --reload

# Train
python3 test_quick_enhanced.py

# Predict
python3 test_predict_with_enhanced_model.py

# Swagger
http://localhost:8000/docs

# Kill server
pkill -f uvicorn
```

---

**Print this and keep it next to you during the demo!** 📄
