import os
import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt

# Directories setup
DATA_DIR = "/Volumes/Drive/VS Code/GIS_v1/data"
OUTPUT_DIR = "/Volumes/Drive/VS Code/GIS_v1/outputs"
SHAPEFILE_PATH = os.path.join(DATA_DIR, "shapefiles", "ground_truth.shp")

T1_DIR = os.path.join(DATA_DIR, "raw", "T1")
T2_DIR = os.path.join(DATA_DIR, "raw", "T2")

def load_band(folder_path, band_keyword, target_shape=None):
    pattern = os.path.join(folder_path, f"*{band_keyword}*.tif")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Could not find band file matching '{band_keyword}' in {folder_path}")
    
    with rasterio.open(files[0]) as src:
        if target_shape is not None and (src.height, src.width) != target_shape:
            # Resample 20m SWIR band up to 10m to match 10m RGB/NIR grid dimensions
            data = src.read(
                1,
                out_shape=target_shape,
                resampling=Resampling.bilinear
            ).astype(np.float32)
        else:
            data = src.read(1).astype(np.float32)
        return data, src.profile

print("[1/5] Loading Sentinel-2 spectral bands for T1 and T2...")
# Load T1 10m anchor band to set baseline target grid shape
blue_t1, profile = load_band(T1_DIR, "blue")
target_shape = blue_t1.shape

green_t1, _ = load_band(T1_DIR, "green", target_shape)
red_t1, _ = load_band(T1_DIR, "red", target_shape)
nir_t1, _ = load_band(T1_DIR, "nir", target_shape)
swir_t1, _ = load_band(T1_DIR, "swir", target_shape)

# Load T2 bands with matching dimensions
blue_t2, _ = load_band(T2_DIR, "blue", target_shape)
green_t2, _ = load_band(T2_DIR, "green", target_shape)
red_t2, _ = load_band(T2_DIR, "red", target_shape)
nir_t2, _ = load_band(T2_DIR, "nir", target_shape)
swir_t2, _ = load_band(T2_DIR, "swir", target_shape)

print("[2/5] Computing spectral indices (NDVI, NDWI, NDBI)...")
def compute_indices(blue, green, red, nir, swir):
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    ndbi = (swir - nir) / (swir + nir + eps)
    return ndvi, ndwi, ndbi

ndvi_t1, ndwi_t1, ndbi_t1 = compute_indices(blue_t1, green_t1, red_t1, nir_t1, swir_t1)
ndvi_t2, ndwi_t2, ndbi_t2 = compute_indices(blue_t2, green_t2, red_t2, nir_t2, swir_t2)

# Stack features into 3D arrays: [Height, Width, Num_Features]
stack_t1 = np.dstack([blue_t1, green_t1, red_t1, nir_t1, swir_t1, ndvi_t1, ndwi_t1, ndbi_t1])
stack_t2 = np.dstack([blue_t2, green_t2, red_t2, nir_t2, swir_t2, ndvi_t2, ndwi_t2, ndbi_t2])

print("[3/5] Performing Change Vector Analysis (CVA) & Otsu Thresholding...")
cva_diff = np.sqrt(np.sum((stack_t2 - stack_t1) ** 2, axis=2))

# Compute Otsu Threshold on CVA magnitude matrix
valid_cva = cva_diff[~np.isnan(cva_diff)]
hist, bin_edges = np.histogram(valid_cva, bins=256)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
total_pixels = valid_cva.size

weight1 = np.cumsum(hist) / total_pixels
weight2 = np.cumsum(hist[::-1])[::-1] / total_pixels
mean1 = np.cumsum(hist * bin_centers) / (weight1 * total_pixels + 1e-6)
mean2 = (np.cumsum((hist * bin_centers)[::-1])[::-1]) / (weight2 * total_pixels + 1e-6)

variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
otsu_threshold = bin_centers[np.argmax(variance)]

change_mask = (cva_diff > otsu_threshold).astype(np.uint8)
print(f"    Calculated CVA Otsu Threshold: {otsu_threshold:.4f}")

print("[4/5] Extracting ground-truth sample points and training Random Forest...")
gdf = gpd.read_file(SHAPEFILE_PATH)

X_train = []
y_train = []

with rasterio.open(glob.glob(os.path.join(T1_DIR, "*blue*.tif"))[0]) as src:
    for idx, row in gdf.iterrows():
        coord = (row.geometry.x, row.geometry.y)
        py, px = src.index(coord[0], coord[1])
        if 0 <= py < stack_t1.shape[0] and 0 <= px < stack_t1.shape[1]:
            sample_features = stack_t1[py, px, :]
            if not np.isnan(sample_features).any():
                X_train.append(sample_features)
                y_train.append(row['class_id'])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_train)

print("\n" + "="*50)
print("             ACCURACY EVALUATION METRICS")
print("="*50)
print(f"Overall Accuracy: {accuracy_score(y_train, y_pred) * 100:.2f}%")
print(f"Cohen's Kappa Coefficient: {cohen_kappa_score(y_train, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_train, y_pred, target_names=["Water", "Vegetation", "Infra", "BLand"]))
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_pred))
print("="*50 + "\n")

print("[5/5] Generating Class-Specific Change Maps...")

# 1. Classify the entire T1 image scene using the trained Random Forest
# Reshape feature stack from [H, W, Bands] -> [H*W, Bands]
height, width, num_bands = stack_t1.shape
flat_stack_t1 = stack_t1.reshape(-1, num_bands)

# Handle potential NaN values safely for full-scene inference
nan_mask = np.isnan(flat_stack_t1).any(axis=1)
flat_stack_t1_clean = np.nan_to_num(flat_stack_t1, nan=0.0)

# Predict class labels across every pixel in the image
full_t1_preds = rf.predict(flat_stack_t1_clean)
full_t1_preds[nan_mask] = 0  # Assign 0 to invalid/NaN pixels
t1_class_map = full_t1_preds.reshape(height, width)

# 2. Extract class-specific change masks
# Class IDs: 1=Water, 2=Vegetation, 3=Infra, 4=BLand
water_change = ((t1_class_map == 1) & (change_mask == 1)).astype(np.uint8)
veg_change   = ((t1_class_map == 2) & (change_mask == 1)).astype(np.uint8)
infra_change = ((t1_class_map == 3) & (change_mask == 1)).astype(np.uint8)
bland_change = ((t1_class_map == 4) & (change_mask == 1)).astype(np.uint8)

# 3. Plot 2x2 grid for class-specific change maps
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(water_change, cmap="gray")
axes[0, 0].set_title("Water Surface Changes (Class 1)", fontsize=12)
axes[0, 0].axis("off")

axes[0, 1].imshow(veg_change, cmap="gray")
axes[0, 1].set_title("Vegetation Disturbances/Harvest (Class 2)", fontsize=12)
axes[0, 1].axis("off")

axes[1, 0].imshow(infra_change, cmap="gray")
axes[1, 0].set_title("Infrastructure/Pavement Shifts (Class 3)", fontsize=12)
axes[1, 0].axis("off")

axes[1, 1].imshow(bland_change, cmap="gray")
axes[1, 1].set_title("Bare Land/Soil Transformations (Class 4)", fontsize=12)
axes[1, 1].axis("off")

output_fig_path = os.path.join(OUTPUT_DIR, "03_class_specific_changes.png")
plt.tight_layout(pad=2.0)
plt.savefig(output_fig_path, dpi=300)
print(f"Class-specific change maps generated and saved to: {output_fig_path}")