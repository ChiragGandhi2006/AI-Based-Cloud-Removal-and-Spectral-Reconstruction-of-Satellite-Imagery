# Frontend

## Main Screen

```text
┌─────────────────────────────────────────────┐
│ Satellite Cloud Removal                    │
├─────────────────────────────────────────────┤
│ Upload Satellite Image                     │
│ [ Choose File ] [ Run Reconstruction ]      │
├─────────────────────────────────────────────┤
│                                             │
│ Cloudy Input      Reconstructed Output      │
│                                             │
├─────────────────────────────────────────────┤
│ Metrics                                     │
│ PSNR | SSIM | RMSE | SAM                   │
├─────────────────────────────────────────────┤
│ Spectral Comparison                         │
│                                             │
├─────────────────────────────────────────────┤
│ NDVI / NDWI Comparison                      │
└─────────────────────────────────────────────┘
```

## Important UI Principle

Do not present the reconstruction as guaranteed ground truth.

Use wording such as:

- Predicted reconstruction
- Reference/ground truth
- Reconstruction error
- Confidence (if uncertainty modeling is added)

## Future Map View

An interactive map can show:

- input footprint
- cloudy image
- reconstructed image
- selected spectral band
- index layers
