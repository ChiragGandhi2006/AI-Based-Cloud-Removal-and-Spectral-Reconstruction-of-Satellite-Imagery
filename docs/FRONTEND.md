# Streamlit Interface

## Why Streamlit?

Streamlit is the final interface technology for this project.

We will use Streamlit because it allows us to build the complete demonstration interface directly in Python without maintaining a separate React frontend or FastAPI backend.

## Main Application

Run the application with:

```bash
streamlit run app/streamlit_app.py
```

## Application Structure

```text
app/
├── streamlit_app.py
├── pages/
│   ├── 1_Cloud_Removal.py
│   ├── 2_Spectral_Analysis.py
│   ├── 3_Model_Performance.py
│   └── 4_About_Project.py
└── components/
    ├── image_viewer.py
    ├── metrics_panel.py
    ├── spectral_plot.py
    ├── index_plot.py
    └── comparison.py
```

## Main Pages

### 1. Cloud Removal

The user can:

- select/upload satellite data
- view the cloudy input
- run the trained model
- view the reconstructed multispectral output
- compare input and output

### 2. Spectral Analysis

Display:

- input spectrum
- reconstructed spectrum
- ground-truth spectrum when available
- spectral error
- band-wise comparison

### 3. Model Performance

Display:

- PSNR
- SSIM
- RMSE
- SAM
- experiment/ablation comparison

### 4. About Project

Explain:

- problem
- solution
- dataset
- model
- technology stack
- project team

## Result Layout

```text
┌─────────────────────────────────────────────┐
│       AI Satellite Cloud Removal            │
├─────────────────────────────────────────────┤
│ Upload / Select Satellite Sample             │
│                                             │
│              [ Reconstruct ]                │
├─────────────────────────────────────────────┤
│ Cloudy Input        AI Reconstruction       │
│ ┌─────────────┐     ┌─────────────┐         │
│ │             │     │             │         │
│ │   IMAGE     │     │   IMAGE     │         │
│ │             │     │             │         │
│ └─────────────┘     └─────────────┘         │
├─────────────────────────────────────────────┤
│ PSNR | SSIM | RMSE | SAM                    │
├─────────────────────────────────────────────┤
│ Spectral Comparison                         │
└─────────────────────────────────────────────┘
```

## Important Principle

The application should clearly distinguish:

- cloudy input
- AI prediction
- ground truth/reference
- reconstruction error

The prediction must not be presented as guaranteed ground truth.

## No Separate Frontend/Backend

The project does **not** use:

- React
- FastAPI
- Docker
- separate frontend/backend services

The Streamlit application directly invokes the Python preprocessing and inference modules.
