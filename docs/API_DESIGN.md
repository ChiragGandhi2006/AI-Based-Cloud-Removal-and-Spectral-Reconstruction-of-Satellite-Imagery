# API Design

## Proposed Endpoints

### POST /predict

Accept an input sample and return reconstruction metadata and output location/reference.

### POST /evaluate

Run evaluation when ground truth is available.

### GET /health

Return service health.

### GET /model-info

Return model version and supported input configuration.

## Example Response

```json
{
  "model_version": "attention-unet-v1",
  "metrics": {
    "psnr": 0.0,
    "ssim": 0.0,
    "rmse": 0.0,
    "sam": 0.0
  },
  "bands": ["B01", "B02", "B03"],
  "status": "success"
}
```

Exact schema should be finalized after the model input/output contract is fixed.

## Security

If deployed:

- validate file types
- limit file sizes
- sanitize filenames
- avoid arbitrary filesystem access
- rate-limit inference endpoints where appropriate
