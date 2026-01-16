# ML Visual Comparison Test (TEST ONLY)

**⚠️ WARNING: This feature is for testing and debugging only. DO NOT release to production.**

## Purpose

This test endpoint generates side-by-side comparison images to visually verify that ML outlier detection is working correctly. It shows:
- **Left side**: Human annotations (green boxes)
- **Right side**: Model predictions (red boxes = high confidence, orange boxes = low confidence)

## Usage

### Method 1: Via Fast Validation UI (Easiest)

1. **Run Fast Validation** on a job from the Queue page
2. **Look for ML-detected outliers** section in the results
3. **Click the "Visual" link** next to any frame number
4. **View side-by-side comparison** in a new tab

The UI shows:
```
Frame 42    2 outlier(s)  [ML icon]  [👁 Visual]  ← Click this link!
  ├─ Box ID 12345    🔴 No matching prediction
  │  └─ Model predicted: car (conf: 15.2%)
  │     IoU: 12.3%
```

### Method 2: Direct API Access

First, run Fast Validation on a job to identify frames with ML outliers:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/job/{job_id}/fast-validation"
```

This will return frames with outliers, including the frame numbers.

Then generate visual comparison for any frame:

```bash
# Via browser (will display image):
http://localhost:8000/api/test/ml-comparison/{job_id}/{frame_num}

# Via curl (save to file):
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/test/ml-comparison/{job_id}/{frame_num}" \
  -o comparison.png
```

**Example:**
```bash
# Visual comparison for job 175, frame 42
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/test/ml-comparison/175/42" \
  -o frame_42_comparison.png
```

### 3. Interpret the Results

The generated image shows:

#### Left Side - Human Annotations (Green)
- Green bounding boxes show what the annotator labeled
- Box ID is displayed above each box

#### Right Side - Model Predictions (Red/Orange)
- **Red boxes**: High confidence predictions (>50%)
- **Orange boxes**: Low confidence predictions (<50%)
- Class name and confidence score shown above each box

#### What to Look For

**Good Annotations** (should match):
- Green boxes on left match red boxes on right
- Similar sizes and positions
- Model has high confidence

**Potential Outliers** (ML flagged these):
- Green box with no matching red/orange box → Model didn't detect it
- Green box matches orange box → Model has low confidence
- Red box with no green box → Missing annotation (not flagged as outlier)

## Examples

### Example 1: Good Annotation
```
Left: [Green box around car]
Right: [Red box around car] "car: 0.95"
Result: Good match, high confidence
```

### Example 2: Outlier - No Match
```
Left: [Green box labeled "person"]
Right: No matching box
Result: ML flagged as outlier (no model prediction)
```

### Example 3: Outlier - Low Confidence
```
Left: [Green box around blurry object]
Right: [Orange box] "person: 0.35"
Result: ML flagged as outlier (low confidence match)
```

## Testing Workflow

1. **Run Fast Validation** on a job
2. **Note frames with ML outliers** from the response
3. **Generate visual comparisons** for those frames
4. **Verify the ML detection** is working correctly:
   - Are the flagged outliers actually problematic?
   - Is the model detecting objects correctly?
   - Are there obvious mismatches?

## API Response Format

**Success** (200 OK):
- Content-Type: `image/png`
- Returns PNG image bytes

**Error Responses**:
- 404: Job or frame not found
- 503: ML detector not available (no model loaded)
- 500: Internal error

## Cleanup

**Before Production Release:**
- [ ] Remove the `/api/test/ml-comparison` endpoint from routes.py
- [ ] Remove the `create_comparison_image()` method from ml_outlier_detection.py
- [ ] Delete this TEST_ML_VISUAL.md file

## Technical Details

### Image Generation
- Uses PIL (Pillow) for image drawing
- Side-by-side layout with 20px gap
- 40px top margin for titles
- Box labels positioned 15px above bounding boxes

### Color Coding
- **Green**: Human annotations
- **Red**: High confidence predictions (≥50%)
- **Orange**: Low confidence predictions (<50%)
- **White**: Background

### Performance
- Processes one frame at a time
- Image generation is fast (~100-500ms per frame)
- No caching (generates fresh each time)

## Troubleshooting

### "ML detector not available"
- Ensure YOLO model is installed at `./models/default/model.pt`
- Or set up project-specific model
- See [models_README.md](models_README.md) for setup

### "Frame not found"
- Verify frame number exists in the job
- Frame numbers are 0-indexed

### Image looks wrong
- Check that annotations have valid bounding boxes
- Verify model is running inference correctly
- Check server console logs for errors

## Security Note

This endpoint requires authentication (user must be logged in). However, since it's for testing only, it should be removed before production deployment.
