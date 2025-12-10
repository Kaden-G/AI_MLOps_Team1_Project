# Swagger Documentation Screenshot Guide
**For Partner 4's Slide Deck**

## Prerequisites

1. Install dependencies:
```bash
# Using pip
pip install fastapi uvicorn torch torchvision scikit-image pandas scikit-learn

# OR using uv (if available)
uv sync
```

2. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The server should start at `http://localhost:8000`

---

## Accessing Swagger UI

Once the server is running, open your browser and navigate to:

**http://localhost:8000/docs**

This will display the interactive Swagger/OpenAPI documentation.

---

## Screenshots to Capture for Slides

### 1. API Overview Page

**What to capture:**
- Full Swagger UI showing all endpoints
- Shows: GET `/`, GET `/health_check`, POST `/train`, POST `/predict`

**How to capture:**
1. Navigate to http://localhost:8000/docs
2. Make sure all endpoints are visible
3. Take a full-page screenshot
4. Save as: `swagger_overview.png`

**What it shows:**
- Complete API structure
- All available endpoints
- API title and description

---

### 2. `/predict` Endpoint - Collapsed View

**What to capture:**
- The `/predict` endpoint in its collapsed state
- Shows the endpoint summary and description

**How to capture:**
1. Locate the `POST /predict` section
2. Ensure it's collapsed (not expanded)
3. Screenshot just that section
4. Save as: `predict_endpoint_collapsed.png`

**What it shows:**
- Endpoint name and method
- Brief description
- Response model indicator

---

### 3. `/predict` Endpoint - Expanded Schema

**What to capture:**
- Request schema (PredictRequest)
- Response schema (PredictResponse)

**How to capture:**
1. Click on `POST /predict` to expand
2. Scroll to show both "Request body" and "Responses" sections
3. Screenshot the schemas
4. Save as: `predict_schemas.png`

**What it shows:**
- Required fields (image_path, model_path)
- Optional fields (model_type with default)
- Response structure with all fields
- Field descriptions and types

---

### 4. `/predict` Endpoint - Example Request

**What to capture:**
- The "Try it out" interface with example values filled in

**How to capture:**
1. Click "Try it out" button
2. Fill in example values:
   ```json
   {
     "image_path": "/Users/kadengodinez/AI_MLOps_Team1_Project/AI_MLOps_Team1_Project/data/initial/Brain Tumor/Brain Tumor/Image1.jpg",
     "model_path": "/path/to/trained_model.pt",
     "model_type": "cnn"
   }
   ```
3. Screenshot the request body editor
4. Save as: `predict_example_request.png`

**What it shows:**
- How to format a prediction request
- Example file paths
- Model type selection

---

### 5. `/predict` Endpoint - Example Response (Success)

**What to capture:**
- Successful prediction response

**How to capture:**
1. After clicking "Execute" (requires trained model)
2. Scroll to the "Response" section
3. Screenshot showing:
   - HTTP 200 status
   - Response body with prediction results
4. Save as: `predict_example_response_success.png`

**What it shows:**
- Successful prediction
- predicted_class (0 or 1)
- probability value
- confidence_percentage
- interpretation text

**Note:** If you don't have a trained model, you can show the example response from the schema instead.

---

### 6. `/predict` Endpoint - Error Response

**What to capture:**
- Error response (404 - File Not Found)

**How to capture:**
1. Try with an invalid path:
   ```json
   {
     "image_path": "/invalid/path/image.jpg",
     "model_path": "/invalid/model.pt",
     "model_type": "cnn"
   }
   ```
2. Click "Execute"
3. Screenshot the error response
4. Save as: `predict_example_response_error.png`

**What it shows:**
- Error handling
- HTTP 404 status
- Descriptive error message

---

### 7. Response Schema Detail

**What to capture:**
- Detailed view of PredictResponse schema

**How to capture:**
1. In the `/predict` section, scroll to "Responses"
2. Expand "200 Successful Response"
3. Click on "Schema" tab if not already visible
4. Screenshot the detailed schema
5. Save as: `predict_response_schema.png`

**What it shows:**
- All response fields with types
- Field descriptions
- Example values

---

### 8. Full API Documentation (Alternative View - ReDoc)

**What to capture:**
- ReDoc alternative documentation view

**How to capture:**
1. Navigate to http://localhost:8000/redoc
2. Scroll to `/predict` section
3. Take screenshot
4. Save as: `redoc_predict_endpoint.png`

**What it shows:**
- Alternative documentation style
- More detailed, printable format
- Better for comprehensive documentation slides

---

## Example Screenshot Captions for Slides

### Slide 1: API Overview
> "Brain Tumor Classification API - Four main endpoints including our prediction service"

### Slide 2: Predict Endpoint
> "POST /predict - Inference endpoint that loads trained models and classifies brain MRI images"

### Slide 3: Request Schema
> "Simple JSON request requiring only image path and model path"

### Slide 4: Response Schema
> "Structured response with predicted class, probability, confidence, and human-readable interpretation"

### Slide 5: Live Demo
> "Interactive API testing via Swagger UI - no additional tools needed"

### Slide 6: Error Handling
> "Robust error handling with descriptive messages for debugging"

---

## Tips for Better Screenshots

1. **Zoom Level:** Set browser zoom to 100% for clear text
2. **Window Size:** Use a consistent window size (e.g., 1920x1080)
3. **Dark Mode:** Swagger supports dark mode - consider if it looks better
4. **Annotations:** Add arrows or highlights in your slide software after capture
5. **Consistency:** Use the same browser and settings for all screenshots

---

## Creating a Demo Video (Optional Bonus)

If you want to create a video demo:

1. Use screen recording software (QuickTime, OBS, etc.)
2. Start recording
3. Navigate through:
   - Opening Swagger UI
   - Expanding `/predict` endpoint
   - Filling in example request
   - Clicking "Execute"
   - Showing successful response
4. Edit video to highlight key points
5. Include in presentation as embedded video

---

## What to Highlight in Slides

### Key Points for Partner 4:

1. **Easy to Use:**
   - Interactive documentation
   - No coding required to test
   - Built-in "Try it out" feature

2. **Well-Documented:**
   - Clear field descriptions
   - Example values provided
   - Error responses documented

3. **Production-Ready Features:**
   - Proper HTTP status codes
   - Structured error messages
   - Type validation
   - Clear response schema

4. **Integration-Friendly:**
   - RESTful API design
   - JSON request/response
   - Standard authentication headers (to be added)
   - OpenAPI/Swagger spec for code generation

---

## Troubleshooting Screenshot Issues

### Issue: Server won't start
**Solution:**
```bash
# Install missing dependencies
pip install -r requirements.txt

# Or manually
pip install fastapi uvicorn
```

### Issue: "localhost:8000 refused to connect"
**Solution:**
- Check server is running in terminal
- Look for message: "Uvicorn running on http://127.0.0.1:8000"
- Try http://127.0.0.1:8000/docs instead

### Issue: Swagger page is blank
**Solution:**
- Clear browser cache
- Try different browser
- Check browser console for JavaScript errors

### Issue: Can't execute requests (gets errors)
**Solution:**
- This is expected if model file doesn't exist yet
- Use the example responses from schema instead
- Or train a model first using `/train` endpoint

---

## Files Checklist

After completing screenshots, you should have:

- [ ] `swagger_overview.png` - Full API overview
- [ ] `predict_endpoint_collapsed.png` - Endpoint summary
- [ ] `predict_schemas.png` - Request/response schemas
- [ ] `predict_example_request.png` - Example request
- [ ] `predict_example_response_success.png` - Success response
- [ ] `predict_example_response_error.png` - Error response
- [ ] `predict_response_schema.png` - Detailed response schema
- [ ] `redoc_predict_endpoint.png` - ReDoc view (optional)

---

## Next Steps for Partner 4

1. Capture screenshots using this guide
2. Insert screenshots into slide deck
3. Add captions and annotations
4. Prepare talking points for demo
5. Test live demo flow before presentation
6. Have backup screenshots in case live demo fails

---

## Support

If you encounter issues capturing screenshots or running the server:
- Check `PREDICT_ENDPOINT_README.md` for troubleshooting
- Review the main `README.md` for setup instructions
- Contact Partner 3 for endpoint-specific questions
