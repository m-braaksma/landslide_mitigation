# Configuration
YEARS = list(range(2000, 2021))
# YEARS = [2000]
COUNTERFACTUAL_YEAR = 2019

BASE_DATA_DIR = "Files/base_data/"
PROJ_DATA_DIR = "Files/base_data/gep_landslides/"
S3_ENDPOINT = "https://s3.msi.umn.edu"
BUCKET_NAME = "jajohns-tier2" 

SDR_INPUT_RASTERS = {
    'alt_m': 'seals/static_regressors/alt_m.tif',
    'global_erosivity': 'global_invest/sediment_delivery/Global Erosivity/GlobalR_NoPol-002.tif',
    'soil_erodibility': 'global_invest/sediment_delivery/Global Soil Erodibility/Data_25km/RUSLE_KFactor_v1.1_25km.tif',
}

# Add year-specific LULC rasters
SDR_INPUT_RASTERS.update({f'lulc_{year}': f'lulc/esa/lulc_esa_{year}.tif' for year in YEARS})

# Job resource defaults
DEFAULT_JOB_RESOURCES = {
    'time': '02:00:00',
    'mem': '8000M',
    'cpus': '1'
}