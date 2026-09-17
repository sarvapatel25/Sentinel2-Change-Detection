import os
import requests
from pystac_client import Client

# Target location on your Mac Drive
BASE_DIR = "/Volumes/Drive/VS Code/GIS_v1/data/raw"

# Area of Interest Bounding Box: [Min Longitude, Min Latitude, Max Longitude, Max Latitude]
# Default set over Ahmedabad / Gandhinagar area
AOI_BBOX = [72.45, 22.95, 72.65, 23.15]

# Date ranges for Date 1 (Older) and Date 2 (Recent)
DATES = {
    "T1": "2024-01-01/2024-03-30",
    "T2": "2025-01-01/2025-03-30"
}

# The specific spectral bands needed for NDVI, NDWI, NDBI & True Color
BANDS = {
    "blue": "B_blue.tif",
    "green": "B_green.tif",
    "red": "B_red.tif",
    "nir": "B_nir.tif",
    "swir16": "B_swir1.tif"
}

def fetch_sentinel_anonymous(date_window, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Open-access STAC catalog hosted by Element 84 on AWS (No auth needed)
    catalog = Client.open("https://earth-search.aws.element84.com/v1")
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=AOI_BBOX,
        datetime=date_window,
        query={"eo:cloud_cover": {"lt": 5}}  # Under 5% cloud cover
    )
    
    items = list(search.items())
    if not items:
        print(f"No clear scene found for date window: {date_window}")
        return
        
    scene = items[0]
    print(f"Found scene: {scene.id} | Date: {scene.datetime.strftime('%Y-%m-%d')}")
    
    for band_key, file_name in BANDS.items():
        if band_key in scene.assets:
            download_url = scene.assets[band_key].href
            save_path = os.path.join(output_folder, file_name)
            
            print(f" Downloading {band_key} band...")
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            print(f" Saved to: {save_path}")

print("=== Starting Anonymous Sentinel-2 Data Download ===")
fetch_sentinel_anonymous(DATES["T1"], os.path.join(BASE_DIR, "T1"))
fetch_sentinel_anonymous(DATES["T2"], os.path.join(BASE_DIR, "T2"))
print("=== Download Complete! ===")