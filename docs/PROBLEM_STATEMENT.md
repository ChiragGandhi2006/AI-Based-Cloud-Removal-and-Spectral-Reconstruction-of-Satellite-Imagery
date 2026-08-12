# Problem Statement

## Background

Satellite optical imagery is essential for monitoring the Earth's surface, but cloud cover and cloud shadows can obscure significant portions of an image.

## Problem

When cloud-covered pixels are discarded, useful observations are lost. Simple interpolation or generic image inpainting can produce visually plausible pixels while failing to preserve physically meaningful spectral information.

## Project Problem Statement

> Design and implement an AI-based multimodal reconstruction system that estimates cloud-obscured Sentinel-2 multispectral information by jointly learning from cloudy optical imagery, cloud/cloud-shadow information and Sentinel-1 SAR observations, and evaluate the reconstruction using spatial and spectral metrics.

## Why AI?

The relationship between SAR backscatter, neighboring optical context and hidden multispectral reflectance is nonlinear. A deep-learning model can learn this relationship from paired cloudy and cloud-free observations.

## Constraints

- The problem is inherently underdetermined: the exact hidden surface cannot always be recovered.
- Seasonal or land-cover changes may exist between observations.
- SAR and optical sensors have different physical characteristics.
- Bands have different native spatial resolutions.
- A visually good reconstruction is not necessarily spectrally accurate.

Therefore, evaluation must include both image-quality and spectral metrics.
