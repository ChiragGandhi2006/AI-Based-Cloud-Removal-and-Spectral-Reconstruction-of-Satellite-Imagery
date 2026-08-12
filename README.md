# AI-Based Cloud Removal and Spectral Reconstruction of Satellite Imagery

## Project Overview

This project focuses on removing clouds from satellite imagery and reconstructing the spectral information using deep learning techniques. The system leverages Sentinel-1 (SAR) and Sentinel-2 (optical) data to produce cloud-free, spectrally consistent images.

## Key Components

- **Data Pipeline**: SEN12MS-CR dataset handling, preprocessing, and augmentation
- **Model Architecture**: U-Net with attention mechanisms and SAR-optical fusion
- **Training Pipeline**: Multi-loss optimization with spectral and structural consistency
- **Evaluation**: Comprehensive metrics including PSNR, SSIM, SAM, NDVI, NDWI
- **Streamlit App**: Interactive interface for cloud removal and spectral analysis

## Directory Structure

```
AI-Cloud-Removal/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/                    # Project documentation
│   ├── PROJECT_OVERVIEW.md
│   ├── PROBLEM_STATEMENT.md
│   ├── OBJECTIVES.md
│   ├── DATASETS.md
│   ├── DATA_PIPELINE.md
│   ├── FEATURE_ENGINEERING.md
│   ├── MODEL_ARCHITECTURE.md
│   ├── TRAINING_PIPELINE.md
│   ├── LOSS_FUNCTIONS.md
│   ├── EVALUATION.md
│   ├── TECH_STACK.md
│   ├── SYSTEM_ARCHITECTURE.md
│   ├── NOVELTY.md
│   ├── IMPLEMENTATION_ROADMAP.md
│   ├── EXPERIMENT_PLAN.md
│   ├── FUTURE_SCOPE.md
│   ├── REFERENCES.md
│   └── PROJECT_DECISIONS.md
├── data/                    # Dataset directories
│   ├── raw/SEN12MS-CR/
│   ├── interim/aligned/
│   ├── interim/normalized/
│   ├── interim/masks/
│   ├── interim/patches/
│   └── processed/train/val/test/
├── configs/                 # YAML configuration files
│   ├── config.yaml
│   ├── dataset.yaml
│   ├── model.yaml
│   └── training.yaml
├── notebooks/               # Jupyter notebooks
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_sentinel2_visualization.ipynb
│   ├── 03_sar_visualization.ipynb
│   ├── 04_cloud_mask_analysis.ipynb
│   ├── 05_feature_analysis.ipynb
│   └── 06_results_analysis.ipynb
├── src/                     # Source code
│   ├── data/                # Data loading and processing
│   ├── preprocessing/       # Image preprocessing pipelines
│   ├── features/            # Feature extraction modules
│   ├── models/              # Neural network architectures
│   ├── losses/              # Custom loss functions
│   ├── training/            # Training loops and utilities
│   ├── evaluation/          # Metrics and evaluation scripts
│   ├── inference/           # Prediction and postprocessing
│   └── utils/               # Utility functions
├── experiments/             # Experiment directories
│   ├── optical_baseline/
│   ├── cloud_aware/
│   ├── sar_fusion/
│   └── attention_unet/
├── checkpoints/             # Model checkpoints
│   ├── baseline/
│   ├── sar_fusion/
│   └── attention_unet/
├── outputs/                 # Generated outputs
│   ├── predictions/
│   ├── visualizations/
│   ├── error_maps/
│   └── spectral_plots/
├── app/                     # Streamlit application
│   ├── streamlit_app.py
│   ├── pages/
│   └── components/
├── tests/                   # Unit tests
│   ├── test_dataset.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_model.py
│   ├── test_losses.py
│   └── test_metrics.py
└── scripts/                 # Utility scripts
    ├── download_dataset.py
    ├── preprocess_dataset.py
    ├── train.py
    ├── evaluate.py
    └── predict.py
```

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Download the SEN12MS-CR dataset using `python scripts/download_dataset.py`
3. Preprocess the data: `python scripts/preprocess_dataset.py`
4. Train a model: `python scripts/train.py`
5. Run the Streamlit app: `streamlit run app/streamlit_app.py`

## Documentation

All project documentation is available in the `docs/` folder, covering problem statements, objectives, datasets, data pipelines, model architectures, training procedures, loss functions, evaluation metrics, and more.