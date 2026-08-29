# AGENTS.md

> Project: AI-Based Cloud Removal and Spectral Reconstruction of Satellite Imagery Version: 1.0 Architecture: Multi-Agent Geospatial AI Pipeline (ISRO Inspired)

# Table of Contents

1. Introduction

2. Why Multi-Agent Architecture?

3. Complete Agent Pipeline

4. Agent Communication Flow

5. Agent Responsibilities

6. Agent 1 – Data Retrieval Agent

7. Agent 2 – Preprocessing Agent

8. Agent 3 – Cloud & Shadow Detection Agent

9. Agent 4 – Change Detection Agent

10. Agent 5 – Adaptive Decision Agent

11. Agent 6 – Cross-Attention Fusion Agent

12. Agent 7 – Multi-Hypothesis Reconstruction Agent

13. Agent 8 – Quality Assessment Network Agent

14. Agent 9 – Confidence Estimation Agent

15. Agent 10 – Delivery & Integration Agent

16. Agent Memory & Data Exchange

17. Error Handling

18. Future Agent Enhancements

# 1. Introduction

This project follows a Multi-Agent Artificial Intelligence Architecture instead of a single monolithic pipeline.

Each stage of satellite image reconstruction is handled by an independent intelligent software agent. Every agent performs a specialized task and passes structured outputs to the next agent.

The architecture is inspired by modern geospatial AI systems used in remote sensing applications.

### Overall Goal

Convert a cloud-covered LISS-IV satellite image into an analysis-ready cloud-free GeoTIFF while preserving spectral consistency and temporal accuracy.

# 2. Why Multi-Agent Architecture?

Traditional ML pipelines perform every task inside one script. That approach becomes difficult to maintain, scale, and debug.

Our architecture separates responsibilities into intelligent agents.

## Advantages

|
Feature

|

Benefit

|
| --- | --- |
|

Modular

|

Easy maintenance

|
|

Scalable

|

Independent upgrades

|
|

Explainable

|

Every decision is traceable

|
|

Parallel

|

Multiple agents can run simultaneously

|
|

Reusable

|

Individual agents can be reused

|

# 3. Complete Agent Pipeline

```
Cloudy LISS-IV Image
        │
        ▼
Data Retrieval Agent
        │
        ▼
Preprocessing Agent
        │
        ▼
Cloud Detection Agent
        │
        ▼
Change Detection Agent
        │
        ▼
Adaptive Decision Agent
        │
        ▼
Cross-Attention Fusion Agent
        │
        ▼
MRR Reconstruction Agent
        │
        ▼
Quality Assessment Agent
        │
        ▼
Confidence Estimation Agent
        │
        ▼
Delivery Agent
        │
        ▼
Cloud-Free GeoTIFF
```

# 4. Agent Communication Flow

Each agent exchanges structured tensors and metadata instead of raw images.

|
Sender

|

Receiver

|

Data

|
| --- | --- | --- |
|

Data Agent

|

Preprocessing

|

TIFF + Metadata

|
|

Preprocessing

|

Cloud Detection

|

Normalized Tensor

|
|

Cloud Detection

|

Change Detection

|

Cloud Mask

|
|

Change Detection

|

Decision Agent

|

Change Map

|
|

Decision Agent

|

Fusion Agent

|

Fusion Strategy

|
|

Fusion Agent

|

Reconstruction

|

Multi-modal Features

|
|

Reconstruction

|

Quality Agent

|

Candidate Images

|
|

Quality Agent

|

Confidence

|

Best Candidate

|
|

Confidence

|

Delivery

|

Final Image

|

# 5. Agent Responsibilities

|
Agent

|

Primary Role

|
| --- | --- |
|

Data Retrieval

|

Fetch historical imagery

|
|

Preprocessing

|

Image alignment

|
|

Cloud Detection

|

Generate cloud mask

|
|

Change Detection

|

Detect temporal changes

|
|

Decision Engine

|

Select reconstruction strategy

|
|

Fusion

|

Combine SAR + Optical

|
|

Reconstruction

|

Generate cloud-free image

|
|

QAN

|

Evaluate image quality

|
|

Confidence

|

Estimate pixel reliability

|
|

Delivery

|

Export & dashboard

|

# 6. Agent 1 – Data Retrieval Agent

## Purpose

Retrieve all required geospatial datasets before AI processing begins.

### Inputs

* Cloudy LISS-IV Image

* Metadata

* Acquisition Date

### External Sources

* Historical LISS-IV

* Sentinel-1 SAR

* Orbit Metadata

### Tasks

1. Read metadata

2. Find historical images

3. Match acquisition dates

4. Retrieve SAR image

5. Select best temporal match

### Output

JSON

```
{
  "liss":"historical_liss.tif",
  "sar":"sentinel1.tif",
  "date":"2026-02-18"
}
```

### Technologies

* Rasterio

* GDAL

* GeoPandas

# 7. Agent 2 – Preprocessing Agent

## Purpose

Prepare images for deep learning.

### Operations

* Co-registration

* Radiometric correction

* Resampling

* Normalization

### Workflow

```
TIFF
 │
 ▼
Read Bands
 │
 ▼
Align Images
 │
 ▼
Normalize
 │
 ▼
Tensor
```

### Input Bands

|
Band

|

Description

|
| --- | --- |
|

B2

|

Blue

|
|

B3

|

Green

|
|

B4

|

Red

|
|

B8

|

NIR

|

### Output

Normalized 4-channel tensor.

# 8. Agent 3 – Cloud & Shadow Detection Agent

## Objective

Identify cloud and shadow pixels.

### AI Model

U-Net Segmentation Network

### Inputs

* RGB

* NIR

* Texture Features

### Outputs

* Cloud Probability Map

* Shadow Probability Map

### Threshold

|
Probability

|

Label

|
| --- | --- |
|

0–0.3

|

Clear

|
|

0.3–0.6

|

Uncertain

|
|

0.6–1.0

|

Cloud

|

### Result

Binary cloud mask.

# 9. Agent 4 – Change Detection Agent

## Purpose

Determine whether the landscape has changed since historical imagery.

### Inputs

* Current Image

* Historical Image

* SAR Image

### Processing

1. Temporal Difference

2. SAR Coherence

3. Pixel Comparison

### Output

Change Probability Map

```
Green → Stable
Red → Changed
```

### Importance

Prevents using outdated historical pixels.

# 10. Agent 5 – Adaptive Decision Agent

## Purpose

Select the best reconstruction strategy.

### Inputs

* Cloud Density

* Change Map

* SAR Availability

* Historical Reliability

### Decision Logic

|
Condition

|

Strategy

|
| --- | --- |
|

Stable

|

Historical dominant

|
|

Moderate

|

Adaptive fusion

|
|

Changed

|

SAR dominant

|

### Output

JSON

```
{
 "strategy":"adaptive_fusion"
}
```

# 11. Agent 6 – Cross-Attention Fusion Agent

## Purpose

Fuse optical and radar features.

### Inputs

* Current LISS-IV

* Historical LISS-IV

* Sentinel-1 SAR

* Cloud Mask

### Deep Learning Block

Cross-Attention Transformer

### Output Features

* Spatial Features

* Spectral Features

* Radar Features

These are passed to reconstruction.

# 12. Agent 7 – Multi-Hypothesis Reconstruction Agent

## Purpose

Generate multiple cloud-free candidates.

### Candidate 1

Historical Dominant

Best for stable regions.

### Candidate 2

SAR Dominant

Best for changed landscapes.

### Candidate 3

Adaptive Fusion

Balanced reconstruction.

### Output

Three reconstructed images.

# 13. Agent 8 – Quality Assessment Network (QAN)

## Purpose

Rank reconstruction candidates.

### Evaluation Metrics

* Spectral Consistency

* Structural Similarity

* SAR Consistency

* Temporal Consistency

### Scoring Formula

Score=0.35Sspec+0.30Sssim+0.20Ssar+0.15StempScore=0.35S_{spec}+0.30S_{ssim}+0.20S_{sar}+0.15S_{temp}Score=0.35Sspec+0.30Sssim+0.20Ssar+0.15Stemp

### Output

Best candidate index.

# 14. Agent 9 – Confidence Estimation Agent

## Purpose

Estimate reliability of every reconstructed pixel.

### Confidence Levels

|
Score

|

Confidence

|
| --- | --- |
|

0.8–1.0

|

High

|
|

0.5–0.8

|

Medium

|
|

0–0.5

|

Low

|

### Output

Confidence heatmap.

### Use Cases

* Scientific analysis

* Decision support

* Quality reporting

# 15. Agent 10 – Delivery & Integration Agent

## Purpose

Provide analysis-ready outputs.

### Outputs

* Cloud-Free GeoTIFF

* Confidence Map

* Change Map

* Quality Report

* Metadata

### Delivery Methods

* Streamlit Dashboard

* REST API

* GeoTIFF Download

# 16. Agent Memory & Data Exchange

Agents communicate through structured objects.

Python

Run

```
ImagePacket
{
    image,
    metadata,
    cloud_mask,
    change_map,
    confidence,
    timestamp
}
```

This enables independent execution.

# 17. Error Handling

|
Agent

|

Possible Error

|

Solution

|
| --- | --- | --- |
|

Data

|

Missing TIFF

|

Request new image

|
|

Preprocessing

|

CRS mismatch

|

Reproject

|
|

Cloud

|

Empty mask

|

Retry threshold

|
|

Change

|

No historical image

|

Use SAR only

|
|

Fusion

|

Missing SAR

|

Optical fallback

|
|

Reconstruction

|

Low confidence

|

Generate alternative

|
|

Delivery

|

Export failed

|

Recreate TIFF

|

# 18. Future Agent Enhancements

## Planned Improvements

### LangGraph Integration

Each agent will become an autonomous LangGraph node.

### LLM Decision Agent

Explain reconstruction decisions in natural language.

### Distributed Processing

Parallel execution across GPU clusters.

### Active Learning Agent

Automatically collect difficult training samples.

### Human-in-the-Loop

Scientists can approve or reject reconstructed regions.

# Agent Summary

|
Agent

|

AI Type

|

Output

|
| --- | --- | --- |
|

Data Retrieval

|

Rule-Based

|

Historical datasets

|
|

Preprocessing

|

Image Processing

|

Normalized tensor

|
|

Cloud Detection

|

U-Net

|

Cloud mask

|
|

Change Detection

|

CNN + SAR

|

Change map

|
|

Decision Engine

|

Rule + ML

|

Fusion strategy

|
|

Cross-Attention

|

Transformer

|

Feature maps

|
|

Reconstruction

|

U-Net

|

3 candidates

|
|

QAN

|

CNN

|

Ranked image

|
|

Confidence

|

Probabilistic

|

Reliability map

|
|

Delivery

|

Service Layer

|

GeoTIFF & Reports

|

## Final Workflow

```
Input Satellite Image
        │
        ▼
Data Retrieval Agent
        │
        ▼
Preprocessing Agent
        │
        ▼
Cloud Detection Agent
        │
        ▼
Change Detection Agent
        │
        ▼
Adaptive Decision Agent
        │
        ▼
Cross-Attention Fusion Agent
        │
        ▼
Multi-Hypothesis Reconstruction
        │
        ▼
Quality Assessment Network
        │
        ▼
Confidence Estimation
        │
        ▼
Delivery Dashboard + API + GeoTIFF
```

This `AGENTS.md` serves as the complete technical specification for the multi-agent AI architecture of the project.
