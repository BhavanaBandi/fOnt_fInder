#!/usr/bin/env python3
"""
Text detection module for font recognition pipeline.
Supports CRAFT-like detection via docTR or custom implementations.
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
import time

logger = logging.getLogger(__name__)


class TextDetector:
    """Base text detector interface."""
    
    def __init__(self, min_confidence: float = 0.5, device: str = "cuda"):
        self.min_confidence = min_confidence
        self.device = device
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect text regions in image. Returns list of bounding boxes."""
        raise NotImplementedError
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """Batch detection on multiple images."""
        return [self.detect(img) for img in images]


class DocTRDetector(TextDetector):
    """Text detector using docTR library (DBNet/CRAFT-like)."""
    
    def __init__(
        self,
        model_name: str = "db_resnet50",
        min_confidence: float = 0.5,
        device: str = "cuda"
    ):
        super().__init__(min_confidence, device)
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """Load docTR detection model."""
        try:
            from doctr.models import detection_predictor
            import torch
            
            self.model = detection_predictor(
                arch=self.model_name,
                pretrained=True
            )
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
            
            logger.info(f"Loaded docTR detector: {self.model_name}")
        except ImportError:
            logger.warning("docTR not available, using fallback detector")
            self.model = None
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect text regions using docTR."""
        if self.model is None:
            return self._fallback_detect(image)
        
        start_time = time.time()
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        result = self.model([image])
        
        detections = []
        h, w = image.shape[:2]
        
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    geometry = line.geometry
                    
                    x1 = int(geometry[0][0] * w)
                    y1 = int(geometry[0][1] * h)
                    x2 = int(geometry[1][0] * w)
                    y2 = int(geometry[1][1] * h)
                    
                    confidence = getattr(line, 'confidence', 0.9)
                    
                    if confidence >= self.min_confidence:
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence),
                            'polygon': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        })
        
        elapsed = time.time() - start_time
        logger.debug(f"Detection took {elapsed:.3f}s, found {len(detections)} regions")
        
        return detections
    
    def _fallback_detect(self, image: np.ndarray) -> List[Dict]:
        """Fallback detection using OpenCV contours."""
        return ContourDetector().detect(image)


class ContourDetector(TextDetector):
    """Simple contour-based text detector (fallback)."""
    
    def __init__(self, min_confidence: float = 0.5, device: str = "cpu"):
        super().__init__(min_confidence, device)
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect text using contour analysis."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        detections = []
        h, w = image.shape[:2]
        min_area = (h * w) * 0.001
        
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            
            if area > min_area and bw > bh * 0.5:
                detections.append({
                    'bbox': [x, y, x + bw, y + bh],
                    'confidence': 0.7,
                    'polygon': [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]
                })
        
        detections = self._merge_overlapping(detections)
        
        return detections
    
    def _merge_overlapping(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """Merge overlapping detections."""
        if len(detections) <= 1:
            return detections
        
        def iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            
            return inter / (area1 + area2 - inter + 1e-6)
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            current_box = det1['bbox'].copy()
            used.add(i)
            
            for j, det2 in enumerate(detections):
                if j in used:
                    continue
                
                if iou(current_box, det2['bbox']) > iou_threshold:
                    current_box[0] = min(current_box[0], det2['bbox'][0])
                    current_box[1] = min(current_box[1], det2['bbox'][1])
                    current_box[2] = max(current_box[2], det2['bbox'][2])
                    current_box[3] = max(current_box[3], det2['bbox'][3])
                    used.add(j)
            
            merged.append({
                'bbox': current_box,
                'confidence': det1['confidence'],
                'polygon': [
                    [current_box[0], current_box[1]],
                    [current_box[2], current_box[1]],
                    [current_box[2], current_box[3]],
                    [current_box[0], current_box[3]]
                ]
            })
        
        return merged


class CropManager:
    """Manages extraction and saving of text region crops."""
    
    def __init__(self, output_dir: str = "./outputs/crops", padding: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.padding = padding
    
    def extract_crops(
        self,
        image: np.ndarray,
        detections: List[Dict],
        image_id: str
    ) -> List[Tuple[np.ndarray, Dict]]:
        """Extract crops from detected regions."""
        crops = []
        h, w = image.shape[:2]
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            
            x1 = max(0, bbox[0] - self.padding)
            y1 = max(0, bbox[1] - self.padding)
            x2 = min(w, bbox[2] + self.padding)
            y2 = min(h, bbox[3] + self.padding)
            
            crop = image[y1:y2, x1:x2]
            
            if crop.size > 0:
                crops.append((crop, det))
        
        return crops
    
    def save_crops(
        self,
        image: np.ndarray,
        detections: List[Dict],
        image_id: str
    ) -> List[str]:
        """Save crops to disk and return paths."""
        crops = self.extract_crops(image, detections, image_id)
        
        image_dir = self.output_dir / image_id
        image_dir.mkdir(exist_ok=True)
        
        paths = []
        for i, (crop, det) in enumerate(crops):
            crop_path = image_dir / f"crop_{i:04d}.png"
            
            if len(crop.shape) == 3 and crop.shape[2] == 3:
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            else:
                crop_bgr = crop
            
            cv2.imwrite(str(crop_path), crop_bgr)
            paths.append(str(crop_path))
        
        return paths


def get_detector(
    detector_type: str = "doctr",
    model_name: str = "db_resnet50",
    min_confidence: float = 0.5,
    device: str = "cuda"
) -> TextDetector:
    """Factory function to get text detector."""
    if detector_type == "doctr":
        return DocTRDetector(model_name, min_confidence, device)
    elif detector_type == "contour":
        return ContourDetector(min_confidence, device)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
