# API_SPEC.md

> Project: AI-Based Cloud Removal and Spectral Reconstruction of Satellite Imagery Version: 1.0 Architecture: RESTful Backend API Specification Protocol: HTTPS + JSON + GeoTIFF

# Table of Contents

1. API Overview

2. Architecture

3. Authentication

4. Base URL

5. Request Standards

6. Response Standards

7. Error Codes

8. Data Models

9. Endpoints

10. Upload API

11. Prediction API

12. Cloud Detection API

13. Change Detection API

14. Reconstruction API

15. Quality Assessment API

16. Confidence API

17. Reports API

18. Download API

19. Admin APIs

20. Security Specification

# 1. API Overview

The backend exposes REST APIs for processing cloud-covered satellite imagery.

The APIs allow:

* Upload GeoTIFF images

* Generate cloud masks

* Perform change detection

* Reconstruct cloud-free imagery

* Generate confidence maps

* Download analysis-ready GeoTIFF

* Produce quality reports

All APIs return JSON except download endpoints.

# 2. System Architecture

```
Client (Dashboard)
        │
 HTTPS/REST
        │
FastAPI Backend
        │
────────────────────────────
│ Data Service
│ ML Service
│ QAN Service
│ Report Service
────────────────────────────
        │
Model Server (TensorFlow)
        │
GeoTIFF Storage
```

# 3. Authentication

Every secured request contains a Bearer token.

Header

http

```
Authorization: Bearer <JWT_TOKEN>
```

### Roles

|
Role

|

Permission

|
| --- | --- |
|

Analyst

|

Upload & Predict

|
|

Scientist

|

Reports + Download

|
|

Admin

|

Manage Pipeline

|

# 4. Base URL

## Development

```
http://localhost:8000/api/v1
```

## Production

```
https://isro-cloud-ai.gov/api/v1
```

# 5. Request Standards

### Content Types

|
Type

|

Usage

|
| --- | --- |
|

application/json

|

Metadata

|
|

multipart/form-data

|

Image Upload

|
|

image/tiff

|

GeoTIFF

|
|

application/pdf

|

Reports

|

### Image Requirements

|
Property

|

Value

|
| --- | --- |
|

Format

|

GeoTIFF

|
|

CRS

|

EPSG:4326

|
|

Bands

|

B2,B3,B4,B8

|
|

Size

|

Max 2 GB

|

# 6. Response Standards

Every response follows the same structure.

JSON

```
{
  "success": true,
  "message": "Prediction completed",
  "data": {},
  "timestamp": "2026-08-22T12:30:10Z"
}
```

# 7. Error Codes

|
Code

|

Meaning

|
| --- | --- |
|

200

|

Success

|
|

201

|

Created

|
|

400

|

Invalid Request

|
|

401

|

Unauthorized

|
|

403

|

Forbidden

|
|

404

|

Not Found

|
|

413

|

File Too Large

|
|

422

|

Invalid GeoTIFF

|
|

500

|

Internal Error

|

Example:

JSON

```
{
 "success":false,
 "message":"Invalid CRS"
}
```

# 8. Data Models

## Image Metadata

JSON

```
{
 "filename":"liss4.tif",
 "width":10240,
 "height":10240,
 "bands":4,
 "crs":"EPSG:4326",
 "resolution":5.8
}
```

## Prediction Result

JSON

```
{
 "prediction_id":"PRD10021",
 "status":"Completed",
 "ssim":0.93,
 "psnr":32.6
}
```

# 9. Endpoint Summary

|
Method

|

Endpoint

|

Purpose

|
| --- | --- | --- |
|

POST

|

/upload

|

Upload TIFF

|
|

POST

|

/predict

|

Complete pipeline

|
|

POST

|

/cloud-mask

|

Cloud detection

|
|

POST

|

/change-map

|

Change detection

|
|

POST

|

/reconstruct

|

Image reconstruction

|
|

GET

|

/quality

|

Quality metrics

|
|

GET

|

/confidence

|

Confidence map

|
|

GET

|

/report

|

PDF report

|
|

GET

|

/download

|

Download TIFF

|

# 10. Upload API

## POST /upload

Uploads a cloud-covered LISS-IV image.

### Request

http

```
POST /upload
Content-Type: multipart/form-data
```

### Body

|
Field

|

Type

|
| --- | --- |
|

image

|

TIFF

|
|

project

|

String

|

### Success

JSON

```
{
 "success":true,
 "image_id":"IMG1021"
}
```

# 11. Prediction API

## POST /predict

Runs the complete AI pipeline.

### Request

JSON

```
{
 "image_id":"IMG1021",
 "historical":true,
 "sar":true
}
```

### Processing

1. Retrieve history

2. Detect cloud

3. Change analysis

4. Fusion

5. Reconstruction

6. QAN

7. Confidence

### Response

JSON

```
{
 "prediction_id":"P1001",
 "status":"completed",
 "output":"cloudfree.tif",
 "ssim":0.92,
 "psnr":31.8
}
```

# 12. Cloud Detection API

## POST /cloud-mask

Generates cloud and shadow masks.

### Request

JSON

```
{
 "image_id":"IMG1021"
}
```

### Response

JSON

```
{
 "cloud_pixels":245102,
 "shadow_pixels":38120,
 "cloud_percentage":23.8
}
```

### Outputs

* Cloud Mask

* Shadow Mask

* Probability Map

# 13. Change Detection API

## POST /change-map

Creates temporal change probability map.

### Input

* Current Image

* Historical Image

* Sentinel-1 SAR

### Response

JSON

```
{
 "stable":78.4,
 "changed":21.6
}
```

### Returned Files

* change_map.tif

* probability_map.png

# 14. Reconstruction API

## POST /reconstruct

Generates cloud-free candidates.

### Request

JSON

```
{
 "prediction_id":"P1001",
 "strategy":"adaptive"
}
```

### Candidates

|
Candidate

|

Description

|
| --- | --- |
|

1

|

Historical

|
|

2

|

SAR

|
|

3

|

Adaptive Fusion

|

### Response

JSON

```
{
 "best_candidate":3,
 "quality_score":96.2
}
```

# 15. Quality Assessment API

## GET /quality/

Returns QAN metrics.

### Response

JSON

```
{
 "spectral":95.4,
 "structural":93.8,
 "temporal":91.6,
 "overall":94.1
}
```

### Metrics

|
Metric

|

Description

|
| --- | --- |
|

SSIM

|

Structural Similarity

|
|

PSNR

|

Peak Signal Ratio

|
|

SAM

|

Spectral Angle Mapper

|
|

IoU

|

Cloud Accuracy

|

# 16. Confidence API

## GET /confidence/

Returns pixel reliability.

### Response

JSON

```
{
 "high":81.3,
 "medium":13.2,
 "low":5.5
}
```

### Confidence Levels

|
Range

|

Label

|
| --- | --- |
|

0.8–1

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

# 17. Reports API

## GET /report/

Generates analysis report.

### Output

PDF containing:

* Original Image

* Cloud Mask

* Change Map

* Reconstructed Image

* Confidence Map

* SSIM

* PSNR

* Metadata

### Response

JSON

```
{
 "report":"report.pdf"
}
```

# 18. Download API

## GET /download/

Downloads analysis-ready GeoTIFF.

### Response Type

http

```
Content-Type: image/tiff
```

Example

```
GET /download/cloudfree_P1001.tif
```

# 19. Admin APIs

## GET /admin/jobs

Lists processing jobs.

JSON

```
[
 {
  "id":"P1001",
  "status":"Running"
 }
]
```

## GET /admin/models

Returns deployed models.

JSON

```
{
 "cloud_model":"UNet v2",
 "fusion_model":"CrossAttention v1",
 "qan":"QAN v1"
}
```

## POST /admin/retrain

Starts model retraining.

JSON

```
{
 "epochs":100,
 "batch_size":8
}
```

# 20. Security Specification

## Authentication

* JWT Tokens

* Role-Based Access Control

## Encryption

* HTTPS TLS 1.3

* Encrypted GeoTIFF storage

## Validation

* File type verification

* CRS validation

* Band validation

* Metadata integrity check

## Rate Limiting

|
API

|

Limit

|
| --- | --- |
|

Upload

|

20/hour

|
|

Predict

|

50/day

|
|

Download

|

200/day

|

# Complete API Workflow

```
POST /upload
      │
      ▼
Image ID
      │
      ▼
POST /predict
      │
      ▼
Cloud Detection
      │
      ▼
Change Detection
      │
      ▼
Adaptive Fusion
      │
      ▼
Reconstruction
      │
      ▼
Quality Assessment
      │
      ▼
Confidence Map
      │
      ▼
GET /download
```

## API Technologies

|
Layer

|

Technology

|
| --- | --- |
|

Framework

|

FastAPI

|
|

AI Service

|

TensorFlow

|
|

Image Processing

|

Rasterio + GDAL

|
|

Authentication

|

JWT

|
|

Documentation

|

OpenAPI / Swagger

|
|

Data Format

|

JSON + GeoTIFF

|
|

Deployment

|

Docker + Uvicorn

|

This `API_SPEC.md` defines the complete backend interface for the ISRO-inspired multi-agent cloud removal system and is suitable for implementation using FastAPI with TensorFlow model serving.
