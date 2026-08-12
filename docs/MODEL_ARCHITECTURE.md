# Model Architecture

## Baseline

**Cloud-Aware Multimodal Attention U-Net**

## Inputs

Conceptually:

```text
X_optical = cloudy Sentinel-2
X_mask = cloud + shadow information
X_sar = Sentinel-1 VV/VH
```

## Architecture

```text
             Cloudy S2
                 │
          Optical Encoder
                 │
                 ├──────────────┐
                 │              │
             SAR Input          │
                 │              │
             SAR Encoder        │
                 │              │
                 └──────┬───────┘
                        ▼
                  Feature Fusion
                        ▼
                 Attention Blocks
                        ▼
                  U-Net Decoder
                        ▼
              Reconstructed S2
```

## Why U-Net?

- Strong spatial reconstruction capability
- Skip connections preserve fine details
- Easier to train than a very large generative architecture
- Suitable for 256×256 image patches

## Why Attention?

Attention can help the network prioritize:

- obscured regions
- relevant spatial context
- SAR features
- useful spectral relationships

## Output

The network predicts the selected Sentinel-2 spectral channels.

The final output should retain the same spatial layout and band ordering as the target.

## Optional Ablations

1. Optical-only U-Net
2. Optical + mask U-Net
3. Optical + SAR U-Net
4. Optical + SAR + attention

These comparisons help demonstrate the contribution of each component.
