# Project Decisions

## Current Recommended Decisions

| Decision | Recommendation | Status |
|---|---|---|
| Primary dataset | SEN12MS-CR | Recommended |
| SAR | Sentinel-1 VV/VH | Recommended |
| Temporal data | Phase 2 | Recommended |
| Model | Attention U-Net | Recommended |
| Output | Multispectral Sentinel-2 | Recommended |
| Cloud detector | Existing mask initially | Recommended |
| Indices | NDVI + NDWI for analysis | Recommended |
| Frontend | React + FastAPI | Recommended |
| Metrics | PSNR, SSIM, RMSE, SAM | Recommended |
| Advanced uncertainty | Phase 2 | Optional |

## Decisions Still Requiring Team Confirmation

- exact Sentinel-2 bands to reconstruct
- final common spatial resolution
- GPU/VRAM constraints
- batch size
- geographic split strategy based on available metadata
- exact cloud-mask source available in the downloaded dataset
- frontend/deployment priority
