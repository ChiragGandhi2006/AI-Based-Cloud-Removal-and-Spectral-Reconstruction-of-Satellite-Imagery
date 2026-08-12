# Novelty and Contribution

## Important Claim

Cloud removal itself is not a new research problem.

The project should not claim that it invented cloud removal.

## Proposed Contribution

The project combines:

1. Multispectral Sentinel-2 information
2. Cloud/cloud-shadow awareness
3. Sentinel-1 SAR fusion
4. Attention-based feature fusion
5. Spectral reconstruction objective
6. Separate obscured-region evaluation
7. Spectral/index consistency analysis

## Proposed Research Contribution Statement

> A practical cloud-aware multimodal deep-learning pipeline for reconstructing cloud-obscured Sentinel-2 multispectral information using complementary Sentinel-1 SAR observations, with explicit evaluation of both spatial reconstruction quality and spectral consistency.

## Demonstrating the Contribution

The ablation study should compare:

- optical only
- optical + masks
- optical + SAR
- optical + SAR + attention

If the multimodal model improves obscured-region and spectral metrics, that provides empirical evidence for the design.
