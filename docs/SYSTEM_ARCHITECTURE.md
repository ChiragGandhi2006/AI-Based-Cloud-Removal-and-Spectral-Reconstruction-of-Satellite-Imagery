# System Architecture

## High-Level Architecture

```text
                  ┌───────────────────┐
                  │ Satellite Dataset│
                  └─────────┬─────────┘
                            ▼
                  ┌───────────────────┐
                  │ Data Preprocessor │
                  └─────────┬─────────┘
                            ▼
                  ┌───────────────────┐
                  │ Feature Generator │
                  └─────────┬─────────┘
                            ▼
              ┌───────────────────────────┐
              │ Multimodal AI Model       │
              │ Attention U-Net           │
              └─────────────┬─────────────┘
                            ▼
                  ┌───────────────────┐
                  │ Reconstruction    │
                  └─────────┬─────────┘
                            ▼
                  ┌───────────────────┐
                  │ Evaluation Engine  │
                  └─────────┬─────────┘
                            ▼
                  ┌───────────────────┐
                  │ Streamlit App      │
                  └───────────────────┘
```

## Application Responsibilities

The Streamlit application handles:

- selecting/uploading input data
- running preprocessing
- loading the trained model
- running inference
- displaying reconstructed imagery
- calculating/displaying metrics
- displaying spectral plots
- displaying NDVI/NDWI comparisons
- displaying experiment results

## Python Module Flow

```text
app/streamlit_app.py
        │
        ├── src/inference/
        │       └── predict.py
        │
        ├── src/preprocessing/
        │
        ├── src/models/
        │
        └── src/evaluation/
```

## Deployment Scope

For the semester project, the application will initially be run locally using Streamlit.

No Docker, Kubernetes, or separate backend/frontend deployment is required.
