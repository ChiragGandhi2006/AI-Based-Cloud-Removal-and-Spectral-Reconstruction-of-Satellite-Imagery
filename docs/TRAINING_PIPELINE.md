# Training Pipeline

## Training Stages

### Stage 1 — Baseline

Train a basic optical reconstruction model.

### Stage 2 — Cloud-Aware Model

Add cloud/cloud-shadow masks.

### Stage 3 — Multimodal Model

Add Sentinel-1 VV/VH.

### Stage 4 — Attention Model

Add attention modules.

This staged approach provides useful ablation experiments.

## Training Procedure

```text
Load batch
   ↓
Normalize
   ↓
Forward pass
   ↓
Prediction
   ↓
Calculate reconstruction + spectral losses
   ↓
Backpropagation
   ↓
Optimizer step
   ↓
Validation
```

## Recommended Training Practices

- PyTorch mixed precision where supported
- AdamW optimizer
- learning-rate scheduler
- early stopping based on validation performance
- checkpoint best validation model
- deterministic seeds where practical

## Data Split

Use geographically aware splits where dataset metadata permits.

## Experiment Tracking

Record:

- experiment name
- dataset version
- input channels
- model configuration
- loss weights
- learning rate
- batch size
- number of epochs
- validation metrics
- checkpoint path
