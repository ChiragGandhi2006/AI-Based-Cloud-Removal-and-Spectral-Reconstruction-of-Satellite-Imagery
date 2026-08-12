# Implementation Roadmap

## Phase 0 — Environment

- [ ] Create Python environment
- [ ] Install PyTorch
- [ ] Install Rasterio/GDAL
- [ ] Verify GPU
- [ ] Create Git repository

## Phase 1 — Dataset

- [ ] Download SEN12MS-CR
- [ ] Inspect directory structure
- [ ] Read representative samples
- [ ] Verify bands and shapes
- [ ] Build Dataset class
- [ ] Build DataLoader

## Phase 2 — Preprocessing

- [ ] Normalize
- [ ] Resample
- [ ] Align S1/S2
- [ ] Prepare cloud masks
- [ ] Implement augmentations
- [ ] Implement geographic split

## Phase 3 — Baseline

- [ ] Optical-only U-Net
- [ ] Train
- [ ] Validate
- [ ] Save checkpoint
- [ ] Calculate metrics

## Phase 4 — Multimodal

- [ ] Add masks
- [ ] Add VV/VH
- [ ] Implement feature fusion
- [ ] Train
- [ ] Compare with baseline

## Phase 5 — Attention

- [ ] Add attention blocks
- [ ] Tune hyperparameters
- [ ] Run ablation study
- [ ] Select final model

## Phase 6 — Evaluation

- [ ] MAE
- [ ] RMSE
- [ ] PSNR
- [ ] SSIM
- [ ] SAM
- [ ] NDVI
- [ ] NDWI
- [ ] Error maps
- [ ] Spectral plots

## Phase 7 — Application

- [ ] FastAPI inference service
- [ ] React interface
- [ ] Result visualization
- [ ] Metrics display
- [ ] Export reconstructed result

## Phase 8 — Optional Research Extensions

- [ ] Temporal information
- [ ] Confidence map
- [ ] Custom cloud detector
- [ ] Transformer comparison
- [ ] Diffusion comparison
