# Streamlit Interface

The project interface is implemented entirely with Streamlit.

## Entry Point

```bash
streamlit run app/streamlit_app.py
```

## Responsibilities

The application connects the trained PyTorch model to a user-friendly interface for:

- satellite sample selection
- cloud-removal inference
- reconstructed-image visualization
- spectral analysis
- NDVI/NDWI analysis
- model metric visualization
- experiment comparison

## Architecture

```text
Streamlit
   ↓
Inference Module
   ↓
Preprocessing
   ↓
Attention U-Net
   ↓
Post-processing
   ↓
Evaluation
   ↓
Visualization
```

No separate React frontend or FastAPI backend is used.
