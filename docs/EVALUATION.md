# Evaluation

## Core Metrics

### MAE

Measures average absolute reconstruction error.

Lower is better.

### RMSE

Measures root mean squared error.

Lower is better.

### PSNR

Measures reconstruction fidelity on an image-quality scale.

Higher is better.

### SSIM

Measures structural similarity.

Higher is better.

### SAM

Measures spectral-angle difference.

Lower is better.

## Evaluate in Two Regions

This is important.

### Clear pixels

Measure whether the model preserves already-visible information.

### Obscured pixels

Measure whether the model reconstructs cloud/cloud-shadow regions.

The obscured-region evaluation is the most important for the project objective.

## Visual Evaluation

Display:

```text
Cloudy Input | Ground Truth | Prediction | Error Map
```

## Spectral Evaluation

Select representative pixels or regions and plot:

```text
Band / Wavelength
       ↓
Ground-truth reflectance
Predicted reflectance
Input reflectance
```

## Index Evaluation

Compare:

- NDVI target vs prediction
- NDWI target vs prediction

## Ablation Evaluation

Compare:

| Experiment | Optical | Mask | SAR | Attention |
|---|---:|---:|---:|---:|
| A | ✓ | | | |
| B | ✓ | ✓ | | |
| C | ✓ | ✓ | ✓ | |
| D | ✓ | ✓ | ✓ | ✓ |

This provides evidence for the value of each feature group.
