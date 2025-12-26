#!/usr/bin/env python3
"""
Streamlit Frontend UI for Font Recognition System.
Allows users to upload images, select models, and run font detection.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.inference import FontInferenceEngine
from pipeline.detection import get_detector, CropManager
from pipeline.preprocessing import FontImagePreprocessor


st.set_page_config(
    page_title="Font Recognition",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
]


@st.cache_resource
def load_inference_engine():
    """Load inference engine (cached)."""
    registry_path = Path(__file__).parent.parent / "models" / "registry.json"
    return FontInferenceEngine(
        model_registry_path=str(registry_path),
        default_model="hybrid"
    )


def get_available_models() -> List[str]:
    """Get list of available models from registry."""
    registry_path = Path(__file__).parent.parent / "models" / "registry.json"
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        return list(registry.keys())
    return ["vit_base", "hybrid_convnext_tiny", "mobilenetv3", "fastervit_base"]


def draw_predictions(
    image: np.ndarray,
    predictions: List[Dict],
    show_confidence: bool = True
) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    img = image.copy()
    
    for i, pred in enumerate(predictions):
        color = COLORS[i % len(COLORS)]
        bbox = pred['bbox']
        
        cv2.rectangle(
            img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color,
            2
        )
        
        label = f"{pred['font']}"
        if pred.get('style') and pred['style'] != 'Unknown':
            label += f" ({pred['style']})"
        if show_confidence:
            label += f" {pred['confidence']:.2f}"
        
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_bg_end = (int(bbox[0]) + label_size[0] + 4, int(bbox[1]) - label_size[1] - 8)
        
        cv2.rectangle(
            img,
            (int(bbox[0]), int(bbox[1]) - label_size[1] - 10),
            label_bg_end,
            color,
            -1
        )
        
        cv2.putText(
            img,
            label,
            (int(bbox[0]) + 2, int(bbox[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return img


def main():
    st.title("🔤 Font Recognition System")
    st.markdown("Upload an image to detect and identify fonts")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        available_models = get_available_models()
        selected_model = st.selectbox(
            "Select Model",
            options=available_models,
            index=0,
            help="Choose the model for font classification"
        )
        
        st.markdown("---")
        
        detection_enabled = st.checkbox(
            "Enable Text Detection",
            value=True,
            help="Detect text regions before classification"
        )
        
        single_text_mode = st.checkbox(
            "Single Text Mode",
            value=False,
            help="Treat entire image as single text (skip detection)"
        )
        
        if single_text_mode:
            detection_enabled = False
        
        st.markdown("---")
        
        show_confidence = st.checkbox("Show Confidence", value=True)
        show_crops = st.checkbox("Show Detected Crops", value=False)
        
        st.markdown("---")
        st.markdown("### Model Info")
        
        registry_path = Path(__file__).parent.parent / "models" / "registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            
            if selected_model in registry:
                model_info = registry[selected_model]
                st.write(f"**Type:** {model_info.get('type', 'N/A')}")
                st.write(f"**Accuracy:** {model_info.get('accuracy', 'N/A'):.2f}%")
                st.write(f"**Top-5:** {model_info.get('top5_accuracy', 'N/A'):.2f}%")
                params = model_info.get('num_parameters', 0)
                st.write(f"**Parameters:** {params:,}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
            help="Upload an image containing text"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image)
            
            st.image(image, caption="Original Image", use_container_width=True)
            
            st.write(f"**Size:** {image.width} x {image.height}")
    
    with col2:
        st.subheader("🎯 Results")
        
        if uploaded_file is not None:
            if st.button("🚀 Run Font Recognition", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        image.save(tmp.name)
                        tmp_path = tmp.name
                    
                    try:
                        engine = load_inference_engine()
                        
                        if selected_model not in engine.registry:
                            st.warning(f"Model '{selected_model}' not found. Using demo mode.")
                            predictions = [{
                                'bbox': [10, 10, image.width - 10, image.height - 10],
                                'font': 'Demo Font',
                                'style': 'Regular',
                                'confidence': 0.95,
                                'model_used': 'demo'
                            }]
                        else:
                            predictions = engine.run_inference(
                                image_path=tmp_path,
                                model_name=selected_model,
                                detect_text=detection_enabled,
                                single_text_mode=single_text_mode
                            )
                        
                        result_image = draw_predictions(
                            image_np, predictions, show_confidence
                        )
                        
                        st.image(
                            result_image,
                            caption="Detection Results",
                            use_container_width=True
                        )
                        
                        st.markdown("### 📊 Detected Fonts")
                        
                        for i, pred in enumerate(predictions):
                            with st.expander(
                                f"Region {i+1}: {pred['font']}",
                                expanded=(i == 0)
                            ):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.write(f"**Font:** {pred['font']}")
                                    st.write(f"**Style:** {pred.get('style', 'Unknown')}")
                                with col_b:
                                    st.write(f"**Confidence:** {pred['confidence']:.2%}")
                                    st.write(f"**Model:** {pred.get('model_used', selected_model)}")
                                
                                bbox = pred['bbox']
                                st.write(f"**BBox:** [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
                                
                                if show_crops:
                                    x1, y1, x2, y2 = [int(b) for b in bbox]
                                    crop = image_np[y1:y2, x1:x2]
                                    if crop.size > 0:
                                        st.image(crop, caption=f"Crop {i+1}", width=200)
                        
                        results_json = json.dumps(predictions, indent=2)
                        st.download_button(
                            "📥 Download Results (JSON)",
                            data=results_json,
                            file_name="font_predictions.json",
                            mime="application/json"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during inference: {str(e)}")
                        st.exception(e)
                    
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
        else:
            st.info("👆 Upload an image to get started")
            
            st.markdown("### 📖 How to Use")
            st.markdown("""
            1. **Upload** an image containing text
            2. **Select** a model from the sidebar
            3. **Configure** detection settings
            4. **Click** 'Run Font Recognition'
            5. **View** results with bounding boxes and font predictions
            """)
            
            st.markdown("### 🎯 Supported Models")
            st.markdown("""
            - **ViT/DeiT**: Vision Transformer models
            - **Hybrid**: CNN backbone + Transformer head
            - **FasterViT**: Efficient hierarchical attention
            - **MobileNetV3**: Lightweight mobile model
            - **ConvNeXt**: Modern CNN architecture
            """)
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Font Recognition System | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
