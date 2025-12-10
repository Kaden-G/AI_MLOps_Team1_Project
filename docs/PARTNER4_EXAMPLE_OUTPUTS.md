# Partner 3 Deliverable: Prediction API Examples

## JSON Output Samples for Partner 4's Slide Deck

### Example 1: Tumor Detected (High Probability)

**Request:**
```json
{
  "image_path": "/data/initial/Brain Tumor/Brain Tumor/Image1500.jpg",
  "model_path": "/models/brain_tumor_cnn.pt",
  "model_type": "cnn"
}
```

**Response:**
```json
{
  "predicted_class": 1,
  "probability": 0.9234,
  "confidence_percentage": 92.34,
  "interpretation": "TUMOR DETECTED - High concern (probability: 92.34%)"
}
```

---

### Example 2: No Tumor Detected (High Confidence)

**Request:**
```json
{
  "image_path": "/data/initial/Brain Tumor/Brain Tumor/Image50.jpg",
  "model_path": "/models/brain_tumor_cnn.pt",
  "model_type": "cnn"
}
```

**Response:**
```json
{
  "predicted_class": 0,
  "probability": 0.1245,
  "confidence_percentage": 87.55,
  "interpretation": "NO TUMOR DETECTED - Low concern (probability: 12.45%)"
}
```

---

### Example 3: Borderline Case (Low Confidence)

**Request:**
```json
{
  "image_path": "/data/initial/Brain Tumor/Brain Tumor/Image750.jpg",
  "model_path": "/models/brain_tumor_cnn.pt",
  "model_type": "cnn"
}
```

**Response:**
```json
{
  "predicted_class": 1,
  "probability": 0.5678,
  "confidence_percentage": 56.78,
  "interpretation": "TUMOR DETECTED - High concern (probability: 56.78%)"
}
```

---

### Example 4: Using NN Model Instead of CNN

**Request:**
```json
{
  "image_path": "/data/initial/Brain Tumor/Brain Tumor/Image2000.jpg",
  "model_path": "/models/brain_tumor_nn.pt",
  "model_type": "nn"
}
```

**Response:**
```json
{
  "predicted_class": 1,
  "probability": 0.8791,
  "confidence_percentage": 87.91,
  "interpretation": "TUMOR DETECTED - High concern (probability: 87.91%)"
}
```

---

## Error Response Examples

### File Not Found Error

**Request:**
```json
{
  "image_path": "/invalid/path/image.jpg",
  "model_path": "/models/brain_tumor_cnn.pt",
  "model_type": "cnn"
}
```

**Response:**
```json
{
  "detail": "Image file not found: /invalid/path/image.jpg"
}
```
**HTTP Status:** 404

---

### Model Not Found Error

**Request:**
```json
{
  "image_path": "/data/initial/Brain Tumor/Brain Tumor/Image1.jpg",
  "model_path": "/invalid/model.pt",
  "model_type": "cnn"
}
```

**Response:**
```json
{
  "detail": "Model file not found: /invalid/model.pt"
}
```
**HTTP Status:** 404

---

## Understanding the Output

### Field Descriptions

1. **predicted_class** (integer)
   - `0` = No Tumor Detected
   - `1` = Tumor Detected
   - Determined by threshold: probability >= 0.5 → class 1

2. **probability** (float, 0.0 to 1.0)
   - Raw model output (sigmoid activation)
   - Represents probability of tumor presence
   - Higher value = more likely to be tumor

3. **confidence_percentage** (float, 0.0 to 100.0)
   - How confident the model is in its prediction
   - Calculated as distance from decision boundary (0.5)
   - For class 1: `probability * 100`
   - For class 0: `(1 - probability) * 100`

4. **interpretation** (string)
   - Human-readable explanation of the result
   - Includes the class prediction and probability
   - Formatted for clinical context

---

## Clinical Interpretation Guidelines

### High Confidence (>80%)
- Model is very confident in its prediction
- Suitable for automated screening

### Medium Confidence (60-80%)
- Model has reasonable confidence
- Recommend manual review by radiologist

### Low Confidence (<60%)
- Model is uncertain
- **Always requires expert review**

### Important Notes
- **False negatives are more dangerous than false positives in medical contexts**
- All positive predictions should be reviewed by medical professionals
- This is a screening tool, not a diagnostic tool

---

## API Endpoint Information

- **Endpoint:** `POST /predict`
- **Content-Type:** `application/json`
- **Response Type:** `application/json`
- **Authentication:** None (add before production)

---

## For Partner 4's Demo

### Suggested Flow for Slides:

1. **Show the request JSON** (Example 1 - clear tumor case)
2. **Show the response JSON** (highlight predicted_class = 1)
3. **Explain the interpretation** (92.34% confidence)
4. **Show a negative case** (Example 2 - no tumor)
5. **Mention error handling** (file not found example)

### Key Points to Emphasize:

- ✓ Clear, structured JSON responses
- ✓ Multiple output fields for different use cases
- ✓ Human-readable interpretation
- ✓ Proper error handling
- ✓ Works with both CNN and NN models
- ✓ Production-ready API design

---

## Next Steps for Integration

Partner 4 can use these examples to:
1. Create slide deck visualizations
2. Test the complete workflow (train → evaluate → predict)
3. Generate demo screenshots
4. Document the full system architecture
