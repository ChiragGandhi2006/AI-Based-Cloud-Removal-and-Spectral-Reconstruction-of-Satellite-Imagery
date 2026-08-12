# Feature Engineering

## A. Sentinel-2 Spectral Features

The primary optical information consists of Sentinel-2 spectral bands.

Important groups include:

- visible bands
- red-edge bands
- NIR
- SWIR
- atmospheric/water-vapour-related bands

The final selected bands should be determined by dataset availability, spatial resolution and model-memory constraints.

## B. Cloud Features

Inputs:

- binary cloud mask
- cloud probability if available

Purpose:

Tell the network which pixels are observed reliably and which pixels require reconstruction.

## C. Cloud-Shadow Features

A separate shadow mask helps distinguish dark cloud-shadow regions from naturally dark surfaces such as water or dense vegetation.

## D. SAR Features

Use:

- VV
- VH

SAR provides information from a sensing modality that is not affected by optical cloud opacity in the same way.

## E. Spectral Indices

Initial analysis:

### NDVI

NDVI = (NIR - Red) / (NIR + Red)

Used to assess vegetation consistency.

### NDWI

Use an appropriate Sentinel-2 band formulation and document it consistently.

NDVI/NDWI should initially be treated mainly as validation/analysis features rather than automatically adding every index to the network.

## F. Spatial Features

The convolutional encoder learns:

- edges
- texture
- local context
- shapes
- spatial relationships

## G. Temporal Features

Not part of the initial baseline.

If Phase 2 is implemented, previous/current/future observations can be added through a temporal fusion module.
