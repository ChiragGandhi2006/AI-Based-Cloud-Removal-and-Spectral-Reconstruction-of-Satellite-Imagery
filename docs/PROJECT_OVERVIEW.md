# Project Overview

## Title

**AI-Based Cloud Removal and Spectral Reconstruction of Satellite Imagery**

## Domain

**Satellite / Geospatial AI, Computer Vision, Deep Learning, Remote Sensing**

## Problem

Optical satellite imagery is frequently contaminated by clouds and cloud shadows. These regions hide the Earth's surface and reduce the usefulness of imagery for agriculture, forestry, water monitoring, urban analysis and environmental applications.

Traditional cloud masking can identify unusable pixels but does not recover the underlying surface information.

## Proposed Solution

Develop a multimodal deep-learning model that uses:

1. Cloudy Sentinel-2 multispectral observations
2. Cloud/cloud-shadow information
3. Sentinel-1 SAR observations

to estimate the cloud-obscured multispectral surface reflectance.

The target is not merely a visually pleasing RGB image. The output should preserve the spectral structure of the Sentinel-2 observation.

## Main Research Question

Can multimodal deep learning use optical context, cloud information and SAR observations to reconstruct cloud-obscured multispectral satellite imagery while maintaining both spatial quality and spectral consistency?

## Expected Outcome

A system that accepts a cloudy satellite patch and produces:

- reconstructed multispectral imagery
- cloud-free visualization
- quality metrics
- spectral comparison
- optional vegetation/water index comparison
