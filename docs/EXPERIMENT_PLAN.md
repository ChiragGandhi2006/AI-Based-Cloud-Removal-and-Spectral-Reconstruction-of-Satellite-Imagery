# Experiment Plan

## Experiment 1 — Optical Baseline

Input:

- cloudy Sentinel-2

Model:

- U-Net

Purpose:

Establish baseline performance.

## Experiment 2 — Cloud-Aware Baseline

Input:

- cloudy Sentinel-2
- cloud mask
- cloud-shadow mask

Purpose:

Measure whether explicit cloud awareness helps.

## Experiment 3 — SAR Fusion

Input:

- cloudy Sentinel-2
- masks
- Sentinel-1 VV/VH

Purpose:

Measure contribution of radar information.

## Experiment 4 — Attention

Input:

- cloudy Sentinel-2
- masks
- SAR

Model:

- Attention U-Net

Purpose:

Evaluate attention-based fusion.

## Experiment 5 — Loss Ablation

Compare:

- L1
- L1 + SSIM
- L1 + SSIM + spectral

## Experiment 6 — Obscured-Region Evaluation

Report metrics separately on:

- all pixels
- clear pixels
- cloud pixels
- cloud-shadow pixels

## Experiment 7 — Real-World Generalization

Use unseen Sentinel-2 scenes where appropriate.

## Results Table Template

| Experiment | PSNR ↑ | SSIM ↑ | RMSE ↓ | SAM ↓ |
|---|---:|---:|---:|---:|
| Optical U-Net | TBD | TBD | TBD | TBD |
| + Masks | TBD | TBD | TBD | TBD |
| + SAR | TBD | TBD | TBD | TBD |
| + Attention | TBD | TBD | TBD | TBD |
