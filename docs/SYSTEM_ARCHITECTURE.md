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
                  │ FastAPI Backend    │
                  └─────────┬─────────┘
                            ▼
                  ┌───────────────────┐
                  │ React Frontend    │
                  └───────────────────┘
```

## Backend Responsibilities

- upload/receive imagery
- validate input
- preprocess
- run inference
- calculate metrics
- return result metadata
- expose reconstructed imagery

## Frontend Responsibilities

- upload image
- show processing state
- display cloudy input
- display reconstruction
- display target when available
- show metrics
- show spectral plots
- show index comparison

## Deployment

First run locally.

Only deploy to a cloud GPU/server after the pipeline is stable.
