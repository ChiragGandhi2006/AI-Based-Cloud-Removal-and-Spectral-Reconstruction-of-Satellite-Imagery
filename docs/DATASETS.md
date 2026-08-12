# Datasets

## 1. Primary Dataset: SEN12MS-CR

SEN12MS-CR is the primary supervised training dataset.

Each sample provides:

- Sentinel-1 SAR information
- Cloudy Sentinel-2 multispectral information
- Cloud-free Sentinel-2 multispectral target

The dataset contains 122,218 patch triplets and uses 256×256 patches with 13 Sentinel-2 bands.

Official dataset page:
https://patricktum.github.io/cloud_removal/sen12mscr/

## 2. Optional Temporal Dataset: SEN12MS-CR-TS

SEN12MS-CR-TS can be introduced later if temporal modeling is required.

It provides multimodal and multitemporal Sentinel observations and is useful for research into temporal cloud removal.

## 3. Real-World Testing Data

New Sentinel-2 Level-2A imagery can be used for final testing/demonstration.

Copernicus Sentinel data access:
https://dataspace.copernicus.eu/

## Dataset Role

| Dataset | Training | Validation | Testing | Purpose |
|---|---:|---:|---:|---|
| SEN12MS-CR | Yes | Yes | Yes | Main supervised development |
| SEN12MS-CR-TS | Optional | Optional | Optional | Temporal extension |
| New Sentinel-2 scenes | No/optional | No | Yes | Real-world generalization |

## Data Governance

Before redistribution or publication, verify the current dataset license and attribution requirements from the official source.
