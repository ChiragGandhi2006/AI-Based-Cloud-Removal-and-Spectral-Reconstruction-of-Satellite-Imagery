# Data Pipeline

## Step 1 — Download

Acquire SEN12MS-CR from its official distribution source.

## Step 2 — Inspect

For representative samples:

- inspect raster dimensions
- inspect band names
- inspect data ranges
- inspect CRS/georeferencing where present
- verify SAR channels
- verify cloudy and cloud-free pairing

## Step 3 — Resample

Bring selected bands to a common spatial resolution for the model.

A practical initial target is 10 m, but memory and information content must be considered before finalizing the resampling strategy.

## Step 4 — Normalize

Use training-set statistics or a documented reflectance normalization scheme.

Never compute normalization statistics using the test set.

## Step 5 — Align

Ensure:

- cloudy optical image
- cloud-free target
- SAR
- masks

refer to the same spatial patch.

## Step 6 — Generate Masks

Create:

- cloud mask
- cloud-shadow mask
- combined obscured-pixel mask

## Step 7 — Build Model Tensor

Initial conceptual tensor:

```text
Cloudy Sentinel-2 bands
+ Cloud mask
+ Cloud-shadow mask
+ Sentinel-1 VV
+ Sentinel-1 VH
```

Optional derived indices should be used carefully and documented separately.

## Step 8 — Split

Prefer geographic/ROI-aware splitting to reduce spatial leakage.

## Step 9 — Augment

Potential augmentations:

- horizontal flip
- vertical flip
- 90-degree rotation

Avoid transformations that change spectral meaning.
