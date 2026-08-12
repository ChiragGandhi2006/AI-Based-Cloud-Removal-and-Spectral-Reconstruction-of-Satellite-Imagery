# Objectives

## Primary Objectives

1. Build a preprocessing pipeline for paired Sentinel-1/Sentinel-2 remote-sensing data.
2. Prepare cloudy Sentinel-2 multispectral inputs.
3. Integrate cloud/cloud-shadow information.
4. Integrate Sentinel-1 VV and VH SAR information.
5. Train an Attention U-Net style reconstruction model.
6. Reconstruct the full selected Sentinel-2 multispectral output.
7. Evaluate spatial reconstruction quality.
8. Evaluate spectral consistency.
9. Build a visualization interface for input, target, output and metrics.

## Secondary Objectives

- Calculate NDVI and NDWI for analysis.
- Compare optical-only and optical+SAR variants.
- Investigate attention visualization.
- Test the trained model on unseen real Sentinel-2 scenes if feasible.

## Phase-2 Objectives

- Temporal observations
- Custom cloud detection
- Uncertainty/confidence maps
- Transformer or diffusion-based comparison
