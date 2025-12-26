#!/usr/bin/env python3
"""
Image preprocessing module for font recognition pipeline.
Handles resizing, normalization, contrast enhancement, and background suppression.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class FontImagePreprocessor:
    """Preprocessor for font recognition images."""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        grayscale: bool = False,
        normalize_contrast: bool = True,
        background_suppression: bool = False
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.grayscale = grayscale
        self.normalize_contrast = normalize_contrast
        self.background_suppression = background_suppression
        
        self._build_transforms()
    
    def _build_transforms(self):
        """Build torchvision transforms."""
        transform_list = []
        
        if self.grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=3))
        
        transform_list.extend([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        self.transform = transforms.Compose(transform_list)
        
        self.train_transform = transforms.Compose([
            transforms.Resize((int(self.image_size[0] * 1.1), int(self.image_size[1] * 1.1))),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement."""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def suppress_background(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for background suppression."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        if len(image.shape) == 3:
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        return thresh
    
    def preprocess_numpy(self, image: np.ndarray) -> np.ndarray:
        """Preprocess numpy array image."""
        if self.normalize_contrast:
            image = self.enhance_contrast(image)
        
        if self.background_suppression:
            image = self.suppress_background(image)
        
        return image
    
    def __call__(
        self,
        image: Union[np.ndarray, Image.Image, str],
        training: bool = False
    ) -> torch.Tensor:
        """Process image and return tensor."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if self.normalize_contrast or self.background_suppression:
                image = self.preprocess_numpy(image)
            image = Image.fromarray(image)
        
        if training:
            return self.train_transform(image)
        return self.transform(image)
    
    def batch_preprocess(
        self,
        images: list,
        training: bool = False
    ) -> torch.Tensor:
        """Preprocess a batch of images."""
        tensors = [self(img, training) for img in images]
        return torch.stack(tensors)


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """Resize image preserving aspect ratio with padding."""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas


def get_train_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Get training augmentation transforms."""
    return transforms.Compose([
        transforms.Resize((int(image_size[0] * 1.15), int(image_size[1] * 1.15))),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.05),
        transforms.RandomRotation(degrees=3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)
    ])


def get_val_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Get validation/inference transforms."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
