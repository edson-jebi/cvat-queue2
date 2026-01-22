"""
Machine Learning-based outlier detection service using YOLO models.
Compares YOLO predictions with human annotations to identify outliers.
Supports per-project models for better accuracy.
Includes bucket validation to check if annotations are within detected regions.
"""

import os
import math
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import httpx


class MLOutlierDetector:
    """Detects annotation outliers by comparing human annotations with YOLO model predictions."""

    def __init__(self, model_path: str = None, project_id: int = None, cvat_host: str = None):
        """
        Initialize the ML outlier detector.

        Args:
            model_path: Path to the YOLO .pt model file. If None, uses instance/project-specific or default.
            project_id: Project ID for loading project-specific model.
            cvat_host: CVAT instance host URL for instance-specific models.
        """
        self.project_id = project_id
        self.cvat_host = cvat_host
        self.model_path = self._resolve_model_path(model_path, project_id)
        self.model = None
        self._ensure_model_loaded()

    def _resolve_model_path(self, model_path: str = None, project_id: int = None) -> str:
        """
        Resolve the model path based on CVAT instance and project ID.

        Directory structure:
        ./models/
        ├── {cvat_host_sanitized}/
        │   ├── {project_id}/
        │   │   └── model.pt
        │   └── default/
        │       └── model.pt
        └── default/
            └── model.pt

        Priority:
        1. Explicit model_path parameter
        2. Instance + project-specific: ./models/{host}/{project_id}/model.pt
        3. Instance default: ./models/{host}/default/model.pt
        4. Environment variable: $YOLO_MODEL_PATH
        5. Global default: ./models/default/model.pt

        Args:
            model_path: Explicit model path
            project_id: Project ID for project-specific model

        Returns:
            Resolved model path
        """
        # If explicit path provided, use it
        if model_path:
            return model_path

        # Get CVAT host from instance (stored during init)
        cvat_host = getattr(self, 'cvat_host', None)

        if cvat_host:
            # Sanitize host for filesystem (remove protocol, replace special chars)
            host_sanitized = cvat_host.replace('http://', '').replace('https://', '').replace('/', '_').replace(':', '_')

            # Try instance + project-specific model
            if project_id is not None:
                instance_project_model = f"./models/{host_sanitized}/{project_id}/model.pt"
                if os.path.exists(instance_project_model):
                    print(f"Using model for instance {cvat_host}, project {project_id}: {instance_project_model}")
                    return instance_project_model

            # Try instance default model
            instance_default_model = f"./models/{host_sanitized}/default/model.pt"
            if os.path.exists(instance_default_model):
                print(f"Using default model for instance {cvat_host}: {instance_default_model}")
                return instance_default_model

        # Try environment variable
        env_model = os.getenv("YOLO_MODEL_PATH")
        if env_model and os.path.exists(env_model):
            return env_model

        # Global default fallback
        return "./models/default/model.pt"

    def _ensure_model_loaded(self):
        """Load the YOLO model with optimizations if not already loaded."""
        if self.model is None:
            try:
                # Lazy import to avoid requiring ultralytics if not used
                from ultralytics import YOLO
                import torch

                if not os.path.exists(self.model_path):
                    print(f"Warning: YOLO model not found at {self.model_path}")
                    print("ML-based outlier detection will be disabled.")
                    return

                # Load model
                self.model = YOLO(self.model_path)

                # Check for GPU availability
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

                # Use FP16 (half precision) on GPU for faster inference
                self.use_half = self.device == 'cuda'

                print(f"✅ YOLO model loaded from {self.model_path}")
                print(f"   Device: {self.device.upper()}")
                if self.device == 'cuda':
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"   GPU: {gpu_name}")
                    print(f"   Using FP16 (half precision) for faster inference")
                else:
                    print(f"   ⚠️  Running on CPU - inference will be slower")
                    print(f"   💡 Install CUDA-enabled PyTorch for GPU acceleration")

            except ImportError as e:
                print(f"Warning: Required package not installed: {e}")
                print("ML-based outlier detection disabled.")
                print("Install with: pip install ultralytics torch")
            except Exception as e:
                print(f"Warning: Failed to load YOLO model: {e}")

    def is_available(self) -> bool:
        """Check if ML detection is available."""
        return self.model is not None

    async def get_frame_image(self, cvat_client, task_id: int, frame_num: int) -> Optional[bytes]:
        """
        Fetch frame image from CVAT using CVATClient.

        Args:
            cvat_client: CVATClient instance
            task_id: Task ID
            frame_num: Frame number

        Returns:
            Image bytes or None if failed
        """
        try:
            return await cvat_client.get_frame_image(task_id, frame_num)
        except Exception as e:
            print(f"Error fetching frame image: {e}")
            return None

    def run_inference_batch(
        self,
        images_bytes_list: List[bytes],
        conf_threshold: float = 0.25,
        imgsz: int = 416
    ) -> List[List[Dict]]:
        """
        Run optimized YOLO inference on multiple images at once (batch processing).
        Much faster than processing one by one.

        Args:
            images_bytes_list: List of image data
            conf_threshold: Confidence threshold for predictions
            imgsz: Image size for inference

        Returns:
            List of predictions for each image
        """
        if not self.is_available():
            return [[] for _ in images_bytes_list]

        try:
            import io
            from PIL import Image

            # Load all images
            images = []
            for image_bytes in images_bytes_list:
                try:
                    images.append(Image.open(io.BytesIO(image_bytes)))
                except:
                    images.append(None)

            # Run batch inference (much faster than one-by-one)
            results_list = self.model(
                images,
                conf=conf_threshold,
                device=self.device,
                half=self.use_half,
                imgsz=imgsz,
                verbose=False,
                agnostic_nms=False,
                max_det=300
            )

            # Extract predictions for each image
            all_predictions = []
            for result in results_list:
                predictions = []
                if result is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        cls_name = self.model.names[cls_id]

                        predictions.append({
                            "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                            "confidence": conf,
                            "class_id": cls_id,
                            "class_name": cls_name
                        })
                all_predictions.append(predictions)

            return all_predictions

        except Exception as e:
            print(f"Error during batch YOLO inference: {e}")
            import traceback
            traceback.print_exc()
            return [[] for _ in images_bytes_list]

    def run_inference(
        self,
        image_bytes: bytes,
        conf_threshold: float = 0.25,
        imgsz: int = 640
    ) -> List[Dict]:
        """
        Run optimized YOLO inference on image.

        Args:
            image_bytes: Image data
            conf_threshold: Confidence threshold for predictions
            imgsz: Image size for inference (smaller = faster, less accurate)
                   Default 640. Use 320-416 for faster inference.

        Returns:
            List of predicted boxes with format:
            [{"bbox": [x1, y1, x2, y2], "confidence": float, "class_id": int, "class_name": str}, ...]
        """
        if not self.is_available():
            return []

        try:
            # Import here to handle missing dependencies gracefully
            import io
            from PIL import Image

            # Load image from bytes
            image = Image.open(io.BytesIO(image_bytes))

            # Run optimized inference
            # - device: Use GPU if available
            # - half: Use FP16 for faster inference on GPU
            # - imgsz: Resize image for faster processing
            # - verbose: Disable logging for cleaner output
            results = self.model(
                image,
                conf=conf_threshold,
                device=self.device,
                half=self.use_half,
                imgsz=imgsz,
                verbose=False,
                # Additional optimizations
                agnostic_nms=False,  # Faster NMS
                max_det=300  # Limit detections per image
            )

            # Extract predictions
            predictions = []
            for result in results:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = self.model.names[cls_id]

                    predictions.append({
                        "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name
                    })

            return predictions

        except Exception as e:
            print(f"Error during YOLO inference: {e}")
            import traceback
            traceback.print_exc()
            return []

    def create_comparison_image(
        self,
        image_bytes: bytes,
        annotations: List[Dict],
        predictions: List[Dict]
    ) -> Optional[bytes]:
        """
        Create side-by-side comparison image showing annotations vs predictions.
        TEST ONLY - Not for production use.

        Args:
            image_bytes: Original image data
            annotations: List of human annotations with "id", "bbox", "label_id"
            predictions: List of YOLO predictions with "bbox", "confidence", "class_id"

        Returns:
            PNG image bytes with side-by-side comparison or None if failed
        """
        try:
            import io
            from PIL import Image, ImageDraw, ImageFont

            # Load image
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size

            # Create two copies for side-by-side comparison
            img_annotations = image.copy()
            img_predictions = image.copy()

            # Draw annotations on left image (green boxes)
            draw_ann = ImageDraw.Draw(img_annotations)
            for ann in annotations:
                bbox = ann["bbox"]
                draw_ann.rectangle(bbox, outline="green", width=3)
                # Add label
                label_text = f"ID: {ann['id']}"
                draw_ann.text((bbox[0], bbox[1] - 15), label_text, fill="green")

            # Draw predictions on right image (red boxes)
            draw_pred = ImageDraw.Draw(img_predictions)
            for pred in predictions:
                bbox = pred["bbox"]
                color = "red" if pred["confidence"] > 0.5 else "orange"
                draw_pred.rectangle(bbox, outline=color, width=3)
                # Add label with confidence
                label_text = f"{pred['class_name']}: {pred['confidence']:.2f}"
                draw_pred.text((bbox[0], bbox[1] - 15), label_text, fill=color)

            # Create side-by-side image
            combined_width = width * 2 + 20  # 20px gap
            combined = Image.new('RGB', (combined_width, height + 40), color='white')

            # Add title text
            draw_combined = ImageDraw.Draw(combined)
            draw_combined.text((10, 5), "Human Annotations (Green)", fill="green")
            draw_combined.text((width + 30, 5), "Model Predictions (Red=high conf, Orange=low conf)", fill="red")

            # Paste images
            combined.paste(img_annotations, (0, 40))
            combined.paste(img_predictions, (width + 20, 40))

            # Convert to bytes
            output = io.BytesIO()
            combined.save(output, format='PNG')
            return output.getvalue()

        except Exception as e:
            print(f"Error creating comparison image: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes [x1, y1, x2, y2]."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def compare_annotations_with_predictions(
        self,
        annotations: List[Dict],
        predictions: List[Dict],
        iou_threshold: float = 0.3,
        min_confidence: float = 0.5
    ) -> List[Dict]:
        """
        Compare human annotations with YOLO predictions to find outliers.

        An annotation is considered an outlier if:
        1. It has no matching prediction (IoU < threshold with all predictions)
        2. Or the best matching prediction has low confidence
        3. Or there are high-confidence predictions with no matching annotations

        Args:
            annotations: List of human annotations with "id", "bbox", "label_id"
            predictions: List of YOLO predictions with "bbox", "confidence", "class_id"
            iou_threshold: Minimum IoU to consider a match
            min_confidence: Minimum confidence for a prediction to be trusted

        Returns:
            List of detailed outlier information with format:
            [{
                "annotation_id": int,
                "annotation_bbox": [x1, y1, x2, y2],
                "annotation_label_id": int,
                "reason": str,  # "no_match" or "low_confidence_match"
                "best_prediction": {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                    "class_id": int,
                    "class_name": str
                } or None,
                "iou": float
            }, ...]
        """
        if not predictions:
            # No predictions available, can't determine outliers
            return []

        outlier_details = []
        matched_annotations = set()
        matched_predictions = set()

        # Find best matches between annotations and predictions
        for i, ann in enumerate(annotations):
            best_iou = 0.0
            best_pred_idx = -1

            for j, pred in enumerate(predictions):
                iou = self.calculate_iou(ann["bbox"], pred["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = j

            # Check if annotation has a good match
            if best_iou >= iou_threshold:
                matched_annotations.add(i)
                matched_predictions.add(best_pred_idx)

                # Check if matched prediction has low confidence (suspicious annotation)
                if predictions[best_pred_idx]["confidence"] < min_confidence:
                    outlier_details.append({
                        "annotation_id": ann["id"],
                        "annotation_bbox": ann["bbox"],
                        "annotation_label_id": ann.get("label_id"),
                        "reason": "low_confidence_match",
                        "best_prediction": predictions[best_pred_idx],
                        "iou": best_iou
                    })
            else:
                # Annotation has no matching prediction - likely outlier
                best_pred = predictions[best_pred_idx] if best_pred_idx >= 0 else None
                outlier_details.append({
                    "annotation_id": ann["id"],
                    "annotation_bbox": ann["bbox"],
                    "annotation_label_id": ann.get("label_id"),
                    "reason": "no_match",
                    "best_prediction": best_pred,
                    "iou": best_iou if best_pred_idx >= 0 else 0.0
                })

        # Optional: Check for high-confidence predictions without matches
        # (could indicate missing annotations, but we focus on bad annotations here)

        return outlier_details

    async def detect_ml_outliers(
        self,
        cvat_client,
        task_id: int,
        frame_annotations: Dict[int, List[Dict]],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.3,
        min_confidence: float = 0.5
    ) -> Dict[int, List[Dict]]:
        """
        Detect outliers across multiple frames using ML inference.

        Args:
            cvat_client: CVATClient instance
            task_id: Task ID
            frame_annotations: Dict mapping frame_num -> list of annotations
            conf_threshold: YOLO confidence threshold
            iou_threshold: IoU threshold for matching
            min_confidence: Minimum confidence for predictions

        Returns:
            Dict mapping frame_num -> list of detailed outlier information
        """
        if not self.is_available():
            print("ML outlier detection not available (model not loaded)")
            return {}

        import time

        print(f"🤖 ML Detection: Starting inference using model: {self.model_path}")
        print(f"🤖 ML Detection: Processing {len(frame_annotations)} frames from task {task_id}")

        outliers_by_frame = {}
        frame_count = 0

        for frame_num, annotations in frame_annotations.items():
            frame_count += 1
            frame_start = time.time()

            # Skip frames with no annotations
            if not annotations:
                continue

            # Fetch frame image
            fetch_start = time.time()
            image_bytes = await self.get_frame_image(cvat_client, task_id, frame_num)
            fetch_time = time.time() - fetch_start

            if not image_bytes:
                print(f"  Frame {frame_num}: Failed to fetch image")
                continue

            # Run YOLO inference with optimized image size
            # Use 416 instead of 640 for ~2x faster inference with minimal accuracy loss
            inference_start = time.time()
            predictions = self.run_inference(
                image_bytes,
                conf_threshold=conf_threshold,
                imgsz=416  # Smaller size for faster inference
            )
            inference_time = time.time() - inference_start

            if not predictions:
                print(f"  Frame {frame_num}: No predictions (fetch: {fetch_time:.2f}s, inference: {inference_time:.2f}s)")
                continue

            # Compare annotations with predictions - now returns detailed info
            compare_start = time.time()
            outlier_details = self.compare_annotations_with_predictions(
                annotations,
                predictions,
                iou_threshold,
                min_confidence
            )
            compare_time = time.time() - compare_start

            frame_total = time.time() - frame_start
            print(f"  Frame {frame_num}: {len(predictions)} predictions, {len(outlier_details)} outliers (fetch: {fetch_time:.2f}s, inference: {inference_time:.2f}s, compare: {compare_time:.2f}s, total: {frame_total:.2f}s)")

            if outlier_details:
                outliers_by_frame[frame_num] = outlier_details

        # Summary
        total_outliers = sum(len(details) for details in outliers_by_frame.values())
        print(f"🤖 ML Detection: Complete - Found {total_outliers} outliers across {len(outliers_by_frame)} frames")

        return outliers_by_frame

    # ==================== BUCKET VALIDATION ====================

    @staticmethod
    def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.

        Args:
            point: (x, y) coordinate
            polygon: List of (x, y) vertices

        Returns:
            True if point is inside polygon
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    @staticmethod
    def bbox_intersects_polygon(bbox: List[float], polygon: List[Tuple[float, float]]) -> Tuple[bool, float]:
        """
        Check if a bounding box intersects with a polygon.
        Returns intersection status and approximate overlap percentage.

        Args:
            bbox: [x1, y1, x2, y2] bounding box
            polygon: List of (x, y) vertices

        Returns:
            (intersects: bool, overlap_percentage: float)
        """
        x1, y1, x2, y2 = bbox

        # Check if any corner of bbox is inside polygon
        corners = [
            (x1, y1),  # top-left
            (x2, y1),  # top-right
            (x1, y2),  # bottom-left
            (x2, y2),  # bottom-right
        ]

        corners_inside = sum(1 for corner in corners if MLOutlierDetector.point_in_polygon(corner, polygon))

        # Check center point
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        center_inside = MLOutlierDetector.point_in_polygon(center, polygon)

        # Calculate approximate overlap percentage
        if corners_inside == 4:
            overlap_pct = 100.0
        elif corners_inside > 0 or center_inside:
            # Partial overlap - estimate based on corners inside
            overlap_pct = (corners_inside / 4) * 100
            if center_inside and corners_inside < 4:
                overlap_pct = max(overlap_pct, 50.0)  # At least 50% if center is inside
        else:
            overlap_pct = 0.0

        intersects = corners_inside > 0 or center_inside
        return intersects, overlap_pct

    @staticmethod
    def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def distance_to_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> float:
        """
        Calculate minimum distance from point to polygon edges.

        Args:
            point: (x, y) coordinate
            polygon: List of (x, y) vertices

        Returns:
            Minimum distance to polygon boundary
        """
        px, py = point
        min_dist = float('inf')

        for i in range(len(polygon)):
            # Get edge vertices
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % len(polygon)]

            # Calculate distance to line segment
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0:
                dist = math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
            else:
                t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
                proj_x, proj_y = x1 + t * dx, y1 + t * dy
                dist = math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

            min_dist = min(min_dist, dist)

        return min_dist

    def detect_bucket_from_predictions(
        self,
        predictions: List[Dict],
        bucket_class_names: List[str] = None,
        min_bucket_confidence: float = 0.3
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Detect THE SINGLE bucket/region polygon from YOLO predictions.
        Only ONE bucket should exist per frame.

        Args:
            predictions: List of YOLO predictions
            bucket_class_names: List of class names that represent buckets
                               Default: ["bucket", "container", "bin", "region", "area", "box"]
            min_bucket_confidence: Minimum confidence to consider a bucket detection valid

        Returns:
            Polygon vertices as list of (x, y) tuples, or None if no bucket found
        """
        if bucket_class_names is None:
            bucket_class_names = ["bucket", "container", "bin", "region", "area", "box", "crate", "tray"]

        # Find all bucket predictions
        bucket_candidates = []
        for pred in predictions:
            class_name = pred.get("class_name", "").lower()
            if any(bucket_name in class_name for bucket_name in bucket_class_names):
                if pred["confidence"] >= min_bucket_confidence:
                    # Calculate bucket area
                    bbox = pred["bbox"]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    bucket_candidates.append({
                        "pred": pred,
                        "area": area,
                        "confidence": pred["confidence"]
                    })

        if not bucket_candidates:
            return None

        # Strategy: Take the LARGEST bucket with good confidence
        # (main bucket should be the biggest in the frame)
        bucket_candidates.sort(key=lambda x: x["area"], reverse=True)

        # Take the largest bucket
        selected_bucket = bucket_candidates[0]["pred"]

        # Log if multiple buckets detected (should only be one)
        if len(bucket_candidates) > 1:
            others = ", ".join([f"{c['pred']['class_name']}({c['confidence']:.2f})"
                               for c in bucket_candidates[1:]])
            print(f"      ⚠️  Multiple buckets detected: {len(bucket_candidates)} - "
                  f"Selected LARGEST: {selected_bucket['class_name']} "
                  f"(conf: {selected_bucket['confidence']:.2f}, area: {bucket_candidates[0]['area']:.0f}px²). "
                  f"Ignored: {others}")

        # Convert bbox to polygon (rectangle)
        bbox = selected_bucket["bbox"]
        x1, y1, x2, y2 = bbox
        polygon = [
            (x1, y1),  # top-left
            (x2, y1),  # top-right
            (x2, y2),  # bottom-right
            (x1, y2),  # bottom-left
        ]

        return polygon

    def validate_annotations_against_bucket(
        self,
        annotations: List[Dict],
        bucket_polygon: List[Tuple[float, float]],
        min_overlap_pct: float = 25.0
    ) -> List[Dict]:
        """
        Validate annotations against bucket polygon.

        Args:
            annotations: List of annotations with "id", "bbox"
            bucket_polygon: Polygon vertices of bucket region
            min_overlap_pct: Minimum overlap percentage to consider valid

        Returns:
            List of invalid annotations with details
        """
        invalid_annotations = []

        for ann in annotations:
            bbox = ann.get("bbox")
            if not bbox:
                continue

            intersects, overlap_pct = self.bbox_intersects_polygon(bbox, bucket_polygon)
            center = self.get_bbox_center(bbox)

            if not intersects or overlap_pct < min_overlap_pct:
                # Calculate distance from center to bucket
                distance = self.distance_to_polygon(center, bucket_polygon)

                invalid_annotations.append({
                    "annotation_id": ann["id"],
                    "annotation_bbox": bbox,
                    "annotation_label_id": ann.get("label_id"),
                    "reason": "outside_bucket",
                    "intersects_bucket": intersects,
                    "overlap_percentage": round(overlap_pct, 1),
                    "distance_to_bucket": round(distance, 1),
                    "center": center
                })

        return invalid_annotations

    def process_frame_bucket_validation(
        self,
        frame_num: int,
        image_bytes: bytes,
        annotations: List[Dict],
        conf_threshold: float = 0.25,
        bucket_class_names: List[str] = None,
        min_overlap_pct: float = 25.0
    ) -> Dict:
        """
        Process a single frame for bucket validation (synchronous for thread pool).

        Args:
            frame_num: Frame number
            image_bytes: Image data
            annotations: List of annotations for this frame
            conf_threshold: YOLO confidence threshold
            bucket_class_names: List of class names for bucket detection
            min_overlap_pct: Minimum overlap to consider annotation valid

        Returns:
            Dict with frame results
        """
        import time
        start_time = time.time()

        # Run inference
        predictions = self.run_inference(image_bytes, conf_threshold, imgsz=416)
        inference_time = time.time() - start_time

        # Detect bucket
        bucket_polygon = self.detect_bucket_from_predictions(predictions, bucket_class_names)

        if bucket_polygon is None:
            return {
                "frame": frame_num,
                "status": "no_bucket",
                "message": "No bucket detected in frame",
                "inference_time": inference_time,
                "predictions_count": len(predictions)
            }

        # Validate annotations against bucket
        invalid_annotations = self.validate_annotations_against_bucket(
            annotations, bucket_polygon, min_overlap_pct
        )

        total_time = time.time() - start_time

        return {
            "frame": frame_num,
            "status": "validated",
            "bucket_polygon": bucket_polygon,
            "total_annotations": len(annotations),
            "invalid_count": len(invalid_annotations),
            "invalid_annotations": invalid_annotations,
            "inference_time": inference_time,
            "total_time": total_time,
            "predictions_count": len(predictions)
        }

    async def detect_bucket_outliers_parallel(
        self,
        cvat_client,
        task_id: int,
        frame_annotations: Dict[int, List[Dict]],
        conf_threshold: float = 0.25,
        bucket_class_names: List[str] = None,
        min_overlap_pct: float = 25.0,
        chunk_size: int = 10,
        max_workers: int = 4
    ) -> Dict[int, Dict]:
        """
        Detect bucket outliers across multiple frames using parallel processing.

        Args:
            cvat_client: CVATClient instance
            task_id: Task ID
            frame_annotations: Dict mapping frame_num -> list of annotations
            conf_threshold: YOLO confidence threshold
            bucket_class_names: Class names for bucket detection
            min_overlap_pct: Minimum overlap to consider valid
            chunk_size: Number of frames to process in each batch
            max_workers: Maximum parallel workers for inference

        Returns:
            Dict mapping frame_num -> validation results
        """
        if not self.is_available():
            print("ML bucket validation not available (model not loaded)")
            return {}

        import time
        total_start = time.time()

        print(f"🪣 Bucket Validation: Starting parallel processing")
        print(f"   Model: {self.model_path}")
        print(f"   Frames: {len(frame_annotations)}")
        print(f"   Chunk size: {chunk_size}, Workers: {max_workers}")

        results_by_frame = {}
        frame_list = list(frame_annotations.items())
        total_frames = len(frame_list)

        # Fetch ALL images in parallel first (much faster)
        print(f"   Fetching all {total_frames} images in parallel...")
        fetch_start = time.time()
        fetch_tasks = [cvat_client.get_frame_image(task_id, fn) for fn, _ in frame_list]
        all_images = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        fetch_time = time.time() - fetch_start
        print(f"   ✅ Fetched all images in {fetch_time:.2f}s ({fetch_time/total_frames:.3f}s per image)")

        # Process in chunks for batch inference
        print(f"   Running batch inference in chunks of {chunk_size}...")
        inference_start = time.time()

        for chunk_start in range(0, total_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_frames)
            chunk = frame_list[chunk_start:chunk_end]
            chunk_images = all_images[chunk_start:chunk_end]

            # Filter out failed images
            valid_indices = []
            valid_images = []
            for i, img in enumerate(chunk_images):
                if not isinstance(img, Exception) and img is not None:
                    valid_indices.append(i)
                    valid_images.append(img)
                else:
                    frame_num = chunk[i][0]
                    results_by_frame[frame_num] = {
                        "frame": frame_num,
                        "status": "error",
                        "message": f"Failed to fetch image: {img}"
                    }

            if not valid_images:
                continue

            # Batch inference (process multiple images at once - MUCH faster)
            batch_predictions = self.run_inference_batch(
                valid_images,
                conf_threshold,
                imgsz=416
            )

            # Validate each frame in the chunk
            for i, pred_idx in enumerate(valid_indices):
                frame_num, annotations = chunk[pred_idx]
                predictions = batch_predictions[i]

                # Detect bucket (only ONE per frame)
                bucket_polygon = self.detect_bucket_from_predictions(
                    predictions,
                    bucket_class_names,
                    min_bucket_confidence=0.4  # Higher threshold to avoid false positives
                )

                if bucket_polygon is None:
                    results_by_frame[frame_num] = {
                        "frame": frame_num,
                        "status": "no_bucket",
                        "message": "No bucket detected",
                        "predictions_count": len(predictions)
                    }
                    continue

                # Validate annotations
                invalid_annotations = self.validate_annotations_against_bucket(
                    annotations, bucket_polygon, min_overlap_pct
                )

                results_by_frame[frame_num] = {
                    "frame": frame_num,
                    "status": "validated",
                    "bucket_polygon": bucket_polygon,
                    "total_annotations": len(annotations),
                    "invalid_count": len(invalid_annotations),
                    "invalid_annotations": invalid_annotations,
                    "predictions_count": len(predictions)
                }

        inference_time = time.time() - inference_start
        print(f"   ✅ Inference completed in {inference_time:.2f}s ({inference_time/total_frames:.3f}s per frame)")

        # Summary
        total_time = time.time() - total_start
        validated = sum(1 for r in results_by_frame.values() if r.get("status") == "validated")
        no_bucket = sum(1 for r in results_by_frame.values() if r.get("status") == "no_bucket")
        errors = sum(1 for r in results_by_frame.values() if r.get("status") == "error")
        total_invalid = sum(r.get("invalid_count", 0) for r in results_by_frame.values())

        print(f"🪣 Bucket Validation Complete:")
        print(f"   Total time: {total_time:.2f}s ({total_time/max(1,len(frame_annotations)):.3f}s per frame)")
        print(f"   Validated: {validated}, No bucket: {no_bucket}, Errors: {errors}")
        print(f"   Invalid annotations found: {total_invalid}")

        return results_by_frame


# Cache for detector instances per (cvat_host, project_id)
_detector_cache: Dict[Tuple[Optional[str], Optional[int]], MLOutlierDetector] = {}


def get_ml_detector(model_path: str = None, project_id: int = None, cvat_host: str = None) -> MLOutlierDetector:
    """
    Get or create an ML detector instance.
    Caches detector instances per (CVAT instance, project) for efficiency.

    Args:
        model_path: Optional explicit model path
        project_id: Optional project ID for project-specific model
        cvat_host: Optional CVAT instance host URL

    Returns:
        MLOutlierDetector instance
    """
    # Use (cvat_host, project_id) tuple as cache key
    cache_key = (cvat_host, project_id)

    # Return cached instance if available
    if cache_key in _detector_cache:
        detector = _detector_cache[cache_key]
        # Verify model path matches if explicit path provided
        if model_path and detector.model_path != model_path:
            # Path changed, create new instance
            detector = MLOutlierDetector(model_path, project_id, cvat_host)
            _detector_cache[cache_key] = detector
        return detector

    # Create new detector instance
    detector = MLOutlierDetector(model_path, project_id, cvat_host)
    _detector_cache[cache_key] = detector
    return detector


def clear_detector_cache(cvat_host: str = None, project_id: int = None):
    """
    Clear cached detector instance(s).
    Useful when model has been updated.

    Args:
        cvat_host: Clear specific instance's detectors, or None for all
        project_id: Clear specific project's detector, or None for all in instance
    """
    global _detector_cache

    if cvat_host is not None and project_id is not None:
        # Clear specific instance + project
        _detector_cache.pop((cvat_host, project_id), None)
    elif cvat_host is not None:
        # Clear all projects for this instance
        keys_to_remove = [k for k in _detector_cache.keys() if k[0] == cvat_host]
        for key in keys_to_remove:
            _detector_cache.pop(key)
    else:
        # Clear all
        _detector_cache.clear()
