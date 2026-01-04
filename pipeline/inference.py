#!/usr/bin/env python3
"""
Unified inference engine for font recognition.
Supports multiple models and detection modes.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
import torch
from PIL import Image

from .detection import get_detector, CropManager
from .preprocessing import FontImagePreprocessor, get_val_transforms

logger = logging.getLogger(__name__)


class FontInferenceEngine:
    """Unified inference engine for font recognition."""
    
    def __init__(
        self,
        model_registry_path: str = "./models/registry.json",
        default_model: str = "hybrid",
        device: str = "cuda",
        detection_model: str = "db_resnet50",
        min_detection_confidence: float = 0.5
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.registry_path = Path(model_registry_path)
        self.default_model = default_model
        
        self.models = {}
        self.current_model_name = None
        self.current_model = None
        self.class_names = None
        
        self.detector = None
        self.detection_model = detection_model
        self.min_detection_confidence = min_detection_confidence
        
        self.preprocessor = FontImagePreprocessor()
        self.crop_manager = CropManager()
        
        self._load_registry()
    
    def _load_registry(self):
        """Load model registry."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
            logger.info(f"Loaded registry with {len(self.registry)} models")
        else:
            self.registry = {}
            logger.warning("No model registry found")
    
    def _load_class_names(self, class_names_path: Optional[str] = None):
        """Load class names from file or dataset."""
        if class_names_path and Path(class_names_path).exists():
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        else:
            train_dir = Path("./font_project_dataset/data/splits/train")
            if train_dir.exists():
                self.class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
            else:
                self.class_names = None
    
    def load_model(self, model_name: str) -> bool:
        """Load a specific model from registry."""
        if model_name in self.models:
            self.current_model = self.models[model_name]
            self.current_model_name = model_name
            return True
        
        if model_name not in self.registry:
            logger.error(f"Model '{model_name}' not in registry")
            return False
        
        model_info = self.registry[model_name]
        checkpoint_path = model_info.get('checkpoint')
        
        if not checkpoint_path or not Path(checkpoint_path).exists():
            logger.error(f"Checkpoint not found for '{model_name}'")
            return False
        
        try:
            model_type = model_info.get('type', 'hybrid')
            model = self._create_model(model_type, model_info)
            
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
            
            model = model.to(self.device)
            model.eval()
            
            self.models[model_name] = model
            self.current_model = model
            self.current_model_name = model_name
            
            logger.info(f"Loaded model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            return False
    
    def _create_model(self, model_type: str, model_info: Dict):
        """Create model instance based on type."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))
        
        num_classes = model_info.get('num_classes', 3812)
        model_config = model_info.get('config', {}) or {}
        
        if model_type == 'vit':
            from models.vit_model import ViTFontModel, DeiTFontModel
            model_name = model_config.get('name')
            model_variant = model_config.get('variant')
            model_kind = model_config.get('model_type', 'vit')
            if model_name is None:
                if model_kind == 'deit' and model_variant:
                    model_name = f"deit_{model_variant}_patch16_224"
                elif model_variant:
                    model_name = f"vit_{model_variant}_patch16_224"
                else:
                    model_name = "vit_base_patch16_224"

            if model_kind == 'deit':
                return DeiTFontModel(num_classes=num_classes, model_name=model_name, pretrained=False)
            return ViTFontModel(num_classes=num_classes, model_name=model_name, pretrained=False)
        elif model_type == 'hybrid':
            from models.hybrid_model import HybridFontModel
            backbone = model_config.get('backbone', 'convnext_tiny')
            transformer_dim = model_config.get('transformer_dim', 384)
            num_heads = model_config.get('num_heads', 6)
            num_layers = model_config.get('num_layers', 4)
            drop_rate = model_config.get('drop_rate', 0.1)
            return HybridFontModel(
                num_classes=num_classes,
                backbone_name=backbone,
                transformer_dim=transformer_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                pretrained_backbone=False,
                drop_rate=drop_rate,
            )
        elif model_type == 'fastervit':
            from models.fastervit_model import FasterViTFontModel
            return FasterViTFontModel(
                num_classes=num_classes,
                embed_dim=model_config.get('embed_dim', 96),
                depths=model_config.get('depths', [2, 2, 6, 2]),
                num_heads=model_config.get('num_heads', [3, 6, 12, 24]),
                window_size=model_config.get('window_size', 7),
                drop_rate=model_config.get('drop_rate', 0.0),
                use_carrier=model_config.get('use_carrier', True),
            )
        elif model_type == 'fontnext':
            from models.fontnext_model import FontNeXtFontModel
            return FontNeXtFontModel(
                num_classes=num_classes,
                backbone_name=model_config.get('backbone', 'convnext_tiny'),
                embed_dim=model_config.get('embed_dim', 384),
                depth=model_config.get('depth', 4),
                num_heads=model_config.get('num_heads', 6),
                mlp_ratio=model_config.get('mlp_ratio', 4.0),
                pool_sizes=model_config.get('pool_sizes', [8, 8, 14, 7]),
                pretrained_backbone=False,
                drop_rate=model_config.get('drop_rate', 0.0),
                attn_drop_rate=model_config.get('attn_drop_rate', 0.0),
                drop_path_rate=model_config.get('drop_path_rate', 0.0),
            )
        elif model_type == 'lightweight':
            import timm

            # Registry may store either a timm model name (preferred) or a shorthand.
            timm_name = model_config.get('name')
            if timm_name is None:
                model_key = model_config.get('model')
                variant = model_config.get('variant')
                if model_key == 'mobilenetv3' and variant == 'large':
                    timm_name = 'mobilenetv3_large_100'
                elif model_key == 'mobilenetv3' and variant == 'small':
                    timm_name = 'mobilenetv3_small_100'
                elif model_key == 'tinyvit':
                    timm_name = 'tiny_vit_5m_224'
                elif model_key == 'convnext_tiny':
                    timm_name = 'convnext_tiny'
                else:
                    timm_name = model_key

            if timm_name is None:
                raise ValueError("Missing timm model name for lightweight model")

            return timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _init_detector(self):
        """Initialize text detector lazily."""
        if self.detector is None:
            self.detector = get_detector(
                detector_type="doctr",
                model_name=self.detection_model,
                min_confidence=self.min_detection_confidence,
                device=self.device
            )
    
    def _classify_crop(self, crop: np.ndarray) -> Dict:
        """Classify a single crop image."""
        if self.current_model is None:
            raise RuntimeError("No model loaded")
        
        tensor = self.preprocessor(crop, training=False)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.current_model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = probs.max(dim=1)
        
        pred_idx = pred_idx.item()
        confidence = confidence.item()
        
        if self.class_names and pred_idx < len(self.class_names):
            class_name = self.class_names[pred_idx]
            parts = class_name.rsplit('_', 1)
            if len(parts) == 2:
                font_name = parts[0].replace('_', ' ').title()
                style = parts[1].title()
            else:
                font_name = class_name.replace('_', ' ').title()
                style = "Regular"
        else:
            font_name = f"class_{pred_idx}"
            style = "Unknown"
        
        return {
            'font': font_name,
            'style': style,
            'confidence': confidence,
            'class_idx': pred_idx
        }
    
    def run_inference(
        self,
        image_path: str,
        model_name: Optional[str] = None,
        detect_text: bool = True,
        single_text_mode: bool = False
    ) -> List[Dict]:
        """
        Run font recognition inference on an image.
        
        Args:
            image_path: Path to input image
            model_name: Name of model to use (from registry)
            detect_text: Whether to run text detection
            single_text_mode: If True, skip detection and classify full image
            
        Returns:
            List of predictions with bbox, font, style, confidence, model_used
        """
        timings = {}
        start_total = time.time()
        
        model_to_use = model_name or self.default_model
        if self.current_model_name != model_to_use:
            if not self.load_model(model_to_use):
                raise RuntimeError(f"Failed to load model: {model_to_use}")
        
        if self.class_names is None:
            self._load_class_names()
        
        start_load = time.time()
        image = np.array(Image.open(image_path).convert('RGB'))
        timings['image_load'] = time.time() - start_load
        
        h, w = image.shape[:2]
        
        results = []
        
        if single_text_mode or not detect_text:
            start_classify = time.time()
            pred = self._classify_crop(image)
            timings['classification'] = time.time() - start_classify
            
            results.append({
                'bbox': [0, 0, w, h],
                'font': pred['font'],
                'style': pred['style'],
                'confidence': pred['confidence'],
                'model_used': model_to_use
            })
        else:
            self._init_detector()
            
            start_detect = time.time()
            detections = self.detector.detect(image)
            timings['detection'] = time.time() - start_detect
            
            if not detections:
                start_classify = time.time()
                pred = self._classify_crop(image)
                timings['classification'] = time.time() - start_classify
                
                results.append({
                    'bbox': [0, 0, w, h],
                    'font': pred['font'],
                    'style': pred['style'],
                    'confidence': pred['confidence'],
                    'model_used': model_to_use,
                    'note': 'no_text_detected_used_full_image'
                })
            else:
                crops = self.crop_manager.extract_crops(
                    image, detections, Path(image_path).stem
                )
                
                start_classify = time.time()
                for crop_img, det in crops:
                    pred = self._classify_crop(crop_img)
                    
                    results.append({
                        'bbox': det['bbox'],
                        'font': pred['font'],
                        'style': pred['style'],
                        'confidence': pred['confidence'],
                        'model_used': model_to_use
                    })
                timings['classification'] = time.time() - start_classify
        
        timings['total'] = time.time() - start_total
        
        logger.info(f"Inference timing: {timings}")
        
        return results
    
    def run_batch_inference(
        self,
        image_paths: List[str],
        model_name: Optional[str] = None,
        detect_text: bool = True
    ) -> Dict[str, List[Dict]]:
        """Run inference on multiple images."""
        results = {}
        for path in image_paths:
            try:
                results[path] = self.run_inference(path, model_name, detect_text)
            except Exception as e:
                logger.error(f"Failed on {path}: {e}")
                results[path] = []
        return results
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from registry."""
        return list(self.registry.keys())


def run_inference(
    image_path: str,
    model_name: str = "hybrid",
    detect_text: bool = True
) -> List[Dict]:
    """
    Convenience function for running inference.
    
    Args:
        image_path: Path to input image
        model_name: Name of model to use
        detect_text: Whether to run text detection
        
    Returns:
        List of predictions with bbox, font, style, confidence, model_used
    """
    engine = FontInferenceEngine(default_model=model_name)
    return engine.run_inference(image_path, model_name, detect_text)
