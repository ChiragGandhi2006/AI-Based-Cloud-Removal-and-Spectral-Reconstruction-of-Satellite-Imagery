# AI-Based Cloud Removal and Spectral Reconstruction of Satellite Imagery

## Project Summary

An AI-based remote-sensing system that reconstructs cloud-obscured Sentinel-2 multispectral imagery using cloudy optical observations, cloud/cloud-shadow information, and Sentinel-1 SAR information.

The primary training dataset is **SEN12MS-CR**, which provides paired Sentinel-1 SAR, cloudy Sentinel-2, and cloud-free Sentinel-2 observations. Real Sentinel-2 imagery may be used for final out-of-distribution testing.

## Core Pipeline

```text
Satellite Data
    ↓
Preprocessing & Alignment
    ↓
Cloud / Shadow Mask
    ↓
Multispectral + SAR Feature Preparation
    ↓
Cloud-Aware Attention U-Net
    ↓
Multispectral Spectral Reconstruction
    ↓
Post-processing
    ↓
PSNR / SSIM / RMSE / SAM + Visual Analysis
    ↓
Web Application
```

## Recommended Initial Scope

- SEN12MS-CR as the main supervised dataset
- Sentinel-1 VV/VH SAR fusion
- Sentinel-2 multispectral reconstruction
- Existing cloud/cloud-shadow masks initially
- Attention U-Net as the baseline architecture
- NDVI and NDWI for validation/analysis
- PSNR, SSIM, RMSE and SAM for evaluation
- React + FastAPI interface after the model pipeline is stable

## Important Scope Rule

Do not start with diffusion models, temporal modeling, a custom cloud detector, or uncertainty estimation. These are optional Phase-2 extensions after the baseline works.
