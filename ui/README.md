# Font Recognition UI

Streamlit-based frontend for the font recognition system.

## Quick Start

```bash
# Install dependencies
pip install streamlit

# Run the UI
cd /path/to/Font_Detector
streamlit run ui/app.py
```

## Features

- **Image Upload**: Support for PNG, JPG, JPEG, BMP, WebP formats
- **Model Selection**: Choose from trained models (ViT, Hybrid, MobileNet, etc.)
- **Text Detection**: Optional text region detection before classification
- **Single Text Mode**: Classify entire image as single text region
- **Visual Results**: Bounding boxes with font labels and confidence scores
- **Export**: Download results as JSON

## UI Components

```
ui/
├── app.py           # Main Streamlit application
├── components/      # Reusable UI components
├── assets/          # Static assets (images, icons)
└── README.md        # This file
```

## Configuration

Settings are controlled via the sidebar:

- **Model Selection**: Choose classification model
- **Text Detection**: Enable/disable region detection
- **Single Text Mode**: Skip detection, use full image
- **Show Confidence**: Display confidence scores
- **Show Crops**: Display detected text crops

## Requirements

- streamlit>=1.28.0
- numpy
- Pillow
- opencv-python

## Usage Tips

1. For **screenshots or cropped text**, use Single Text Mode for best results
2. For **posters or multi-font images**, enable Text Detection
3. Higher confidence scores indicate more reliable predictions
4. Use the JSON export for programmatic access to results
