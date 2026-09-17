# Bi-Temporal Sentinel-2 Change Detection Pipeline

A geospatial intelligence (GEOINT) change detection and land-cover classification pipeline built with Python, QGIS, and Scikit-Learn. The system ingests multi-spectral Sentinel-2 Level-2A imagery across two time periods ($T_1$ and $T_2$), resamples spatial grids dynamically, computes spectral index vectors, executes Change Vector Analysis (CVA) with automated Otsu thresholding, and validates land cover using a Random Forest classifier.

---

## Technical Architecture

                       +-----------------------------+
                       |   Sentinel-2 L2A Imagery    |
                       | (T1: Q1 2024 / T2: Q1 2025) |
                       +--------------+--------------+
                                      |
                                      v
                       +-----------------------------+
                       | Bilinear Band Resampling    |
                       | (20m SWIR -> 10m RGB/NIR)   |
                       +--------------+--------------+
                                      |
                                      v
                       +-----------------------------+
                       | Multi-Spectral Feature Stack|
                       | (RGB, NIR, SWIR, NDVI,      |
                       |  NDWI, NDBI)                |
                       +--------------+--------------+
                                      |
               +----------------------+----------------------+
               |                                             |
               v                                             v
+------------------------------+             +------------------------------+
| Change Vector Analysis (CVA) |             | Random Forest Classification |
| & Otsu Auto-Thresholding     |             | (Digitized GT Points)        |
+--------------+---------------+             +--------------+---------------+
               |                                             |
               +----------------------+----------------------+
                                      |
                                      v
                       +-----------------------------+
                       | Output Thematic Change Maps |
                       | & Validation Report Metrics |
                       +-----------------------------+

---

## Methodology & Formulation

### 1. Spatial Harmonization
Sentinel-2 optical bands arrive at mismatched spatial resolutions. Short-Wave Infrared (SWIR, Band 11) is delivered at 20m resolution, while visible RGB and Near-Infrared (NIR) are delivered at 10m. SWIR bands are dynamically resampled using **Bilinear Interpolation** to match the baseline 10m grid (10980 x 10980 matrix).

### 2. Spectral Feature Index Generation
For each date T, the pipeline generates three indices to capture vegetation, hydrological, and built-up shifts:

- **NDVI:** (NIR - Red) / (NIR + Red)
- **NDWI:** (Green - NIR) / (Green + NIR)
- **NDBI:** (SWIR - NIR) / (SWIR + NIR)

### 3. Change Vector Analysis (CVA) & Otsu Thresholding
The spectral change magnitude is evaluated pixel-wise across the 8-dimensional feature stack using Euclidean distance. An optimal binary threshold is determined automatically by maximizing between-class variance on the magnitude histogram using **Otsu's Method**. Pixels above the threshold are classified as significant ground changes.

---

## Model Evaluation Metrics

Evaluated across 123 digitized ground-truth points across four land-cover classes (`Water`, `Vegetation`, `Infra`, `BLand`):

* **Overall Accuracy:** 100.00%
* **Cohen's Kappa Coefficient:** 1.0000
* **Calculated Otsu Threshold:** 1146.75

### Confusion Matrix

                  Predicted
             Water  Veg  Infra  BLand
Actual Water   41    0     0      0
       Veg      0   40     0      0
       Infra    0    0    25      0
       BLand    0    0     0     17

---

## Repository Structure

GIS_v1/
│
├── data/
│   ├── raw/
│   │   ├── T1/                     # Raw Sentinel-2 T1 GeoTIFF bands
│   │   └── T2/                     # Raw Sentinel-2 T2 GeoTIFF bands
│   └── shapefiles/
│       ├── ground_truth.shp        # Digitized point ground truth
│       ├── ground_truth.dbf
│       ├── ground_truth.prj
│       └── ground_truth.shx
│
├── outputs/
│   ├── 01_raw_aoi_visual.png       # QGIS verification snapshot
│   ├── 02_cva_change_map.png       # Primary multi-panel CVA output
│   └── 03_class_specific_changes.png # 2x2 Class-wise change grid
│
├── scripts/
│   ├── download_data.py            # STAC catalog automated downloader
│   └── train_eval_pipeline.py      # Core CVA & ML execution script
│
├── .gitignore
├── README.md
└── requirements.txt

---

## Installation & Execution

1. Clone repository:
   git clone https://github.com/sarvapatel25/Sentinel2-Change-Detection.git
   cd Sentinel2-Change-Detection

2. Install dependencies:
   pip install -r requirements.txt

3. Run pipeline:
   python3 scripts/train_eval_pipeline.py