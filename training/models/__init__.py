"""Font Recognition Model Implementations."""

from .vit_model import ViTFontModel, DeiTFontModel
from .hybrid_model import HybridFontModel
from .fastervit_model import FasterViTFontModel
from .fontnext_model import FontNeXtFontModel
from .Model_M_75M import HybridCNNViTFontDetector, create_font_detector
from .Model_P_27_8M import HybridCNNViT, get_model
from .Model_O_23_9M import FontClassifier

__all__ = [
    'ViTFontModel',
    'DeiTFontModel', 
    'HybridFontModel',
    'FasterViTFontModel',
    'FontNeXtFontModel',
    'HybridCNNViTFontDetector',
    'create_font_detector',
    'HybridCNNViT',
    'get_model',
    'FontClassifier',
]
