"""Project information and documentation."""
import os
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

st.set_page_config(page_title="About Project", page_icon="📖")
st.title("📖 About the Project")

st.markdown(
    """
## AI-Based Cloud Removal and Spectral Reconstruction of Satellite Imagery

This project removes clouds from optical satellite imagery and reconstructs the
underlying spectral information using deep learning. It fuses **Sentinel-1 SAR** (weather
independent) with **Sentinel-2 optical** data through an **Attention U-Net** architecture,
trained with a combined L1 + SSIM + spectral-angle loss on the **SEN12MS-CR** dataset.

### Highlights
- **Data pipeline** - SEN12MS-CR parsing, alignment, cloud-mask generation, patching
- **Model** - U-Net with attention gates and optional SAR/mask conditioning
- **Training** - mask-weighted multi-objective loss, mixed precision, checkpointing
- **Evaluation** - PSNR, SSIM, RMSE, SAM, NDVI, NDWI
- **App** - this Streamlit interface for interactive demos and analysis

### Tech stack
PyTorch · NumPy · rasterio · scikit-image · Streamlit · Plotly · Matplotlib

### Getting started
```bash
pip install -r requirements.txt
python scripts/download_dataset.py      # download SEN12MS-CR (see docs/DATASETS.md)
python scripts/preprocess_dataset.py    # build npz patches
python scripts/train.py                 # train the model
python scripts/evaluate.py              # evaluate on test split
streamlit run app/streamlit_app.py      # launch this app
```

### Documentation
Full documentation lives in `docs/` (problem statement, data pipeline, model
architecture, training, evaluation, and more).
"""
)
