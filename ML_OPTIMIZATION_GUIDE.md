# ML Inference Optimization Guide

This document explains the optimizations applied to YOLO model inference for faster Fast Validation performance.

## Applied Optimizations

### 1. GPU Acceleration (CUDA)
**What it does**: Uses GPU instead of CPU for inference
**Speed improvement**: 10-50x faster depending on GPU
**How to enable**:
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Status indicators**:
```
✅ YOLO model loaded from ./models/default/model.pt
   Device: CUDA                    # ← GPU enabled
   GPU: NVIDIA GeForce RTX 3080
   Using FP16 (half precision) for faster inference
```

```
✅ YOLO model loaded from ./models/default/model.pt
   Device: CPU                     # ← CPU only
   ⚠️  Running on CPU - inference will be slower
   💡 Install CUDA-enabled PyTorch for GPU acceleration
```

### 2. FP16 Half Precision
**What it does**: Uses 16-bit floats instead of 32-bit on GPU
**Speed improvement**: 1.5-2x faster on modern GPUs
**Accuracy impact**: Minimal (~0.5% difference)
**Automatically enabled**: Only on GPU (CUDA)

### 3. Reduced Image Size
**What it does**: Resizes images to 416x416 instead of 640x640 before inference
**Speed improvement**: ~2x faster
**Accuracy impact**: Small (5-10% less accurate on small objects)
**Configuration**:
```python
# Current setting (fast)
predictions = ml_detector.run_inference(image_bytes, imgsz=416)

# Options:
# - 320: Very fast, lower accuracy
# - 416: Fast, good accuracy (CURRENT)
# - 640: Standard, best accuracy (default YOLO)
# - 1280: Slow, best for small objects
```

### 4. Optimized NMS and Detection Limits
**What it does**:
- `agnostic_nms=False`: Faster non-maximum suppression
- `max_det=300`: Limits detections per image
**Speed improvement**: 10-20% faster
**Impact**: Minimal for typical annotation workloads

### 5. Frame Limit
**What it does**: Limits ML detection to first 2 frames per job
**Why**: Prevents timeout while testing performance
**Configuration**: See `routes.py` line 2070
```python
frames_to_check = dict(list(frame_annotations_for_ml.items())[:2])
```

## Performance Expectations

### CPU Only (No GPU)
- Per-frame inference: **2-5 seconds**
- 2 frames: **4-10 seconds** total
- 5 frames: **10-25 seconds** total

### With GPU (CUDA)
- Per-frame inference: **0.1-0.5 seconds**
- 2 frames: **0.2-1 second** total
- 5 frames: **0.5-2.5 seconds** total
- 50 frames: **5-25 seconds** total

### Network Overhead
- Fetching frame images: **0.3-0.8 seconds per frame**
- This is unavoidable (depends on CVAT server speed)

## Checking Current Performance

When running Fast Validation, check the server console logs:

```
🤖 ML Detection: Processing 2 frames from task 9
  Frame 0: 15 predictions, 3 outliers (fetch: 0.45s, inference: 0.18s, compare: 0.01s, total: 0.64s)
  Frame 1: 12 predictions, 1 outliers (fetch: 0.42s, inference: 0.15s, compare: 0.01s, total: 0.58s)
🤖 ML Detection: Complete - Found 4 outliers across 2 frames
⏱️  ML Detection completed in 1.22 seconds
```

**Key metrics**:
- **fetch**: Time to download image from CVAT (network)
- **inference**: Time for YOLO model to process image (GPU/CPU)
- **compare**: Time to compare predictions with annotations (always fast)
- **total**: Total time per frame

## Troubleshooting

### Still Too Slow?

#### 1. Verify GPU is being used
```bash
# In server console, look for:
✅ YOLO model loaded from ./models/default/model.pt
   Device: CUDA  # ← Should say CUDA, not CPU
```

If it says CPU, install CUDA PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Use smaller image size
Edit `ml_outlier_detection.py` line 466:
```python
imgsz=320  # Even faster (was 416)
```

#### 3. Use smaller/faster YOLO model
- YOLOv8n (nano): Fastest, less accurate
- YOLOv8s (small): Fast, good accuracy
- YOLOv8m (medium): Balanced
- YOLOv8l (large): Slow, better accuracy
- YOLOv8x (extra): Slowest, best accuracy

Download smaller model:
```bash
# Replace your model with YOLOv8n (nano)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
cp ~/.cache/ultralytics/yolov8n.pt ./models/default/model.pt
```

#### 4. Increase frame limit cautiously
Only after confirming single-frame inference is fast (<0.5s):

Edit `routes.py` line 2070:
```python
frames_to_check = dict(list(frame_annotations_for_ml.items())[:5])  # Increase from 2 to 5
```

Then eventually:
```python
frames_to_check = dict(list(frame_annotations_for_ml.items())[:50])  # Production setting
```

#### 5. Consider batch processing (future enhancement)
Currently processes frames one-by-one. Could batch multiple frames together for better GPU utilization.

## Hardware Recommendations

### Minimum (CPU Only)
- Modern CPU (4+ cores)
- 8GB RAM
- Expected: 2-5 seconds per frame

### Recommended (GPU)
- NVIDIA GPU with CUDA support (GTX 1060 or better)
- 4GB+ VRAM
- CUDA 11.8 or newer
- Expected: 0.1-0.5 seconds per frame

### Optimal (Production)
- NVIDIA RTX 3060 or better
- 8GB+ VRAM
- CUDA 11.8 or newer
- Expected: 0.05-0.2 seconds per frame
- Can handle 50+ frames per job without timeout

## Testing Your Setup

### 1. Check GPU availability:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 2. Test inference speed:
```python
from ultralytics import YOLO
import time

# Load model
model = YOLO('./models/default/model.pt')

# Load test image
from PIL import Image
img = Image.open('test_image.jpg')

# Warmup (first inference is always slower)
model(img, device='cuda', half=True, imgsz=416, verbose=False)

# Benchmark
start = time.time()
for i in range(10):
    results = model(img, device='cuda', half=True, imgsz=416, verbose=False)
elapsed = time.time() - start

print(f"Average inference time: {elapsed/10:.3f} seconds")
```

**Target times**:
- GPU: <0.2 seconds
- CPU: <3 seconds

### 3. Run Fast Validation:
Monitor the server console logs for actual performance in production.

## Configuration Summary

| Setting | Location | Current Value | Options |
|---------|----------|---------------|---------|
| Device | Auto-detected | CUDA or CPU | - |
| FP16 | Auto (GPU only) | True | - |
| Image size | `ml_outlier_detection.py:466` | 416 | 320, 416, 640, 1280 |
| Frame limit | `routes.py:2070` | 2 | 1-50+ |
| Confidence | `routes.py:2087` | 0.25 | 0.1-0.9 |
| IoU threshold | `routes.py:2088` | 0.3 | 0.1-0.9 |

## Future Enhancements

1. **Batch processing**: Process multiple frames in one GPU call
2. **Async inference**: Run inference in background thread
3. **Model caching**: Keep model warm in memory
4. **Progressive processing**: Return results as they're ready
5. **TensorRT optimization**: 2-5x faster on NVIDIA GPUs
6. **ONNX export**: Better CPU performance

## Monitoring in Production

Add these metrics to track performance:
- Average inference time per frame
- GPU utilization %
- Frames processed per minute
- Timeout rate
- Frame limit usage (how many jobs hit the limit)

This will help identify when to increase frame limits or add more GPU resources.
