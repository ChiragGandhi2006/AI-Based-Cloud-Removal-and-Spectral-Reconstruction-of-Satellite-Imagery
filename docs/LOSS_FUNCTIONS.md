# Loss Functions

## 1. Pixel Reconstruction Loss

L1 loss is a good initial choice because it is less sensitive to extreme outliers than squared error and often produces sharper reconstruction than pure MSE.

```text
L1 = mean(abs(target - prediction))
```

## 2. Structural Loss

SSIM-based loss can encourage preservation of local spatial structure.

## 3. Spectral Loss

A spectral consistency component should encourage predicted spectral vectors to point in a similar direction to the target vectors.

Spectral Angle Mapper (SAM) is particularly useful for evaluation and can also be explored as a training loss after numerical stability is verified.

## Total Loss

```text
L_total =
    λ1 * L1
  + λ2 * L_SSIM
  + λ3 * L_spectral
```

The initial lambda values should be treated as hyperparameters and tuned on the validation set.

## Mask-Aware Loss

An additional improvement is to weight cloud-obscured pixels more strongly:

```text
L = L_clear + α * L_obscured
```

This prevents the large clear portion of an image from dominating the reconstruction objective.

The exact weighting must be validated experimentally.
