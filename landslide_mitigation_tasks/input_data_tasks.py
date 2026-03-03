import os
import sys

import hazelbean as hb
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from osgeo import gdal, ogr, osr
import pygeoprocessing as pygeo
import tempfile
import rasterio

def preprocess_data(p):
    """
    Creates unified dir for preprocessed data
    """
    if p.run_this:
        return p

def preprocess_nasa_glc(p):
    """
    Preprocess NASA GLC landslide data to create annual pixel-level landslide occurrence panel.
    1. Load raw NASA GLC landslide data
        format: csv
        source: https://catalog.data.gov/dataset/global-landslide-catalog-export
        citation: 
    2. Create blank raster with matching resolution
    3. Subset:
        - Temporal: 2000-2019 (20 years)
        - Accuracy: only include landslides with accuracy <= 1000m
    4. For each year, rasterize the landslide points that occurred in that year
        - Binary raster: 1 if landslide occurred, 0 otherwise
        - Mortality raster: use reported mortality to create a raster of estimated mortality per pixel
    """
    if p.run_this:
    
        binary_out_paths = [os.path.join(p.preprocess_data_dir, f'glc_binary_{year}.tif') for year in p.time_range]
        mortality_out_paths = [os.path.join(p.preprocess_data_dir, f'glc_mortality_{year}.tif') for year in p.time_range]
        out_paths = binary_out_paths + mortality_out_paths
        if all(os.path.exists(path) for path in out_paths):
            p.L.info(f'All GLC binary and mortality rasters already exist in {p.preprocess_data_dir}')
            return p
        else:
            p.L.info(f'Generating GLC binary and mortality rasters in {p.preprocess_data_dir}')

        glc_path = os.path.join(p.base_data_dir, 'nasa_glc', 'Global_Landslide_Catalog_Export_rows.csv')
        glc_df = pd.read_csv(glc_path)
        glc_gdf = gpd.GeoDataFrame(glc_df, geometry=gpd.points_from_xy(glc_df['longitude'], glc_df['latitude']))
        glc_gdf['event_date'] = pd.to_datetime(glc_gdf['event_date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

        # Filter by accuracy
        accuracy_map = {
            'exact': 0,
            'unknown': np.nan,
            '1km': 1000,
            '5km': 5000,
            '10km': 10000,
            '25km': 25000,
            '50km': 50000
        }
        glc_gdf['location_accuracy_m'] = glc_gdf['location_accuracy'].map(accuracy_map)
        glc_gdf = glc_gdf[glc_gdf['location_accuracy_m'] <= p.max_location_accuracy_m]

        # Buffer each point by its location accuracy (in meters)
        # Ensure CRS is projected in meters for accurate buffering
        if glc_gdf.crs is None or glc_gdf.crs.is_geographic:
            glc_gdf = glc_gdf.set_crs('EPSG:4326')
            glc_gdf = glc_gdf.to_crs('EPSG:3857')
        glc_gdf['geometry'] = glc_gdf.buffer(glc_gdf['location_accuracy_m'])

        # Create annual rasters
        for year in p.time_range:
            binary_out_path = os.path.join(p.preprocess_data_dir, f'glc_binary_{year}.tif')
            mortality_out_path = os.path.join(p.preprocess_data_dir, f'glc_mortality_{year}.tif')
            
            if os.path.exists(binary_out_path) and os.path.exists(mortality_out_path):
                p.L.info(f'GLC binary and mortality rasters for {year} already exist, skipping.')
                continue
            
            # Create blank rasters
            pygeo.new_raster_from_base(
                p.reference_raster_path,
                binary_out_path,
                gdal.GDT_Byte,
                band_nodata_list=[-1],
                fill_value_list=[0])
            pygeo.new_raster_from_base(
                p.reference_raster_path,
                mortality_out_path,
                gdal.GDT_Float32,
                band_nodata_list=[-9999.0],
                fill_value_list=[0.0])

            yearly_gdf = glc_gdf[glc_gdf['event_date'].dt.year == year]
            if yearly_gdf.empty:
                p.L.warning(f'No landslide events found for year {year}, creating empty rasters.')
                continue
            
            # Save yearly_gdf to a temporary vector file
            with tempfile.TemporaryDirectory() as tmpdir:
                vector_path = os.path.join(tmpdir, f'yearly_{year}.gpkg')
                yearly_gdf.to_file(vector_path, driver='GPKG')

                # Rasterize binary occurrence
                pygeo.rasterize(
                    vector_path,
                    binary_out_path,
                    burn_values=[1],
                    option_list=['ALL_TOUCHED=TRUE'],
                    layer_id=0
                )
                
                # Check landslide count vs raster
                # landslide_count = len(yearly_gdf)
                # with rasterio.open(binary_out_path) as src:
                #     arr = src.read(1)
                #     ndv = src.nodatavals[0]
                #     raster_count = np.sum((arr != ndv) & (arr == 1))
                # p.L.info(f'Year {year}: {landslide_count} landslides in GDF, {raster_count} pixels with landslides in raster.')
                # # Assert binary raster count matches GDF count
                # assert landslide_count == raster_count, f"Mismatch: {landslide_count} landslides in GDF, {raster_count} pixels in raster for year {year}"
                # Distance decay mortality rasterization
                with rasterio.open(mortality_out_path, 'r+') as dst:
                    arr = dst.read(1)
                    transform = dst.transform
                    for idx, row in yearly_gdf.iterrows():
                        if pd.isna(row['fatality_count']) or row['fatality_count'] == 0:
                            continue
                        # Use original point geometry for center
                        center = row['geometry'].centroid
                        buffer_radius = row['location_accuracy_m']
                        # Create a circular buffer around the point
                        buffer_geom = center.buffer(buffer_radius)
                        # Get bounds and pixel window
                        minx, miny, maxx, maxy = buffer_geom.bounds
                        row_start, col_start = rasterio.transform.rowcol(transform, minx, maxy)
                        row_stop, col_stop = rasterio.transform.rowcol(transform, maxx, miny)
                        # Ensure window is within raster
                        row_start = max(row_start, 0)
                        col_start = max(col_start, 0)
                        row_stop = min(row_stop, arr.shape[0] - 1)
                        col_stop = min(col_stop, arr.shape[1] - 1)
                        # For each pixel in window, compute distance to center and apply decay
                        for r in range(row_start, row_stop + 1):
                            for c in range(col_start, col_stop + 1):
                                x, y = rasterio.transform.xy(transform, r, c, offset='center')
                                point = gpd.points_from_xy([x], [y], crs=yearly_gdf.crs)[0]
                                dist = center.distance(point)
                                if dist <= buffer_radius:
                                    # Linear decay: weight = 1 - (dist / buffer_radius)
                                    weight = 1 - (dist / buffer_radius)
                                    arr[r, c] += row['fatality_count'] * weight
                    dst.write(arr, 1)

                    # Assert total mortality matches
                    # gdf_mortality_total = yearly_gdf['fatality_count'].sum()
                    # with rasterio.open(mortality_out_path) as src:
                    #     arr = src.read(1)
                    #     ndv = src.nodatavals[0]
                    #     raster_mortality_total = arr[arr != ndv].sum()
                    # assert np.isclose(gdf_mortality_total, raster_mortality_total, rtol=0.01), f"Mismatch: {gdf_mortality_total} total fatalities in GDF, {raster_mortality_total} in raster for year {year}"
                    
            p.L.info(f'Created GLC binary and mortality rasters for {year} at {binary_out_path} and {mortality_out_path}')

        # Save buffered GeoDataFrame as GeoPackage for inspection
        buffered_gpkg_path = os.path.join(p.preprocess_data_dir, 'glc_buffered_points.gpkg')
        glc_gdf.to_file(buffered_gpkg_path, driver='GPKG')
        p.L.info(f'Saved buffered landslide points to {buffered_gpkg_path}')


def preprocess_lsfactor_gedtm(p):
    """
    Preprocess LS-Factor data from GEDTM to create a time-invariant raster of landslide susceptibility.
    1. Load raw LS-Factor data
        format: GeoTIFF raster
        source: https://zenodo.org/records/18702591
        citation: @dataset{ho_2026_18702591,
  author       = {Ho, Yufeng and
                  Hengl, Tom},
  title        = {Multiscale Land Surface Parameters of GEDTM30: LS
                   Factor
                  },
  month        = feb,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.2.0},
  doi          = {10.5281/zenodo.18702591},
  url          = {https://doi.org/10.5281/zenodo.18702591},
}
    2. Reproject and align with reference
    """
    if p.run_this:
        return p
    
def preprocess_worldpop(p):
    """
    Preprocess WorldPop population data to create annual population rasters.
    1. Load raw WorldPop data
        format: GeoTIFF raster
        source: https://hub.worldpop.org/geodata/listing?id=64
        citation: 
    2. For each year, reproject and align with reference
    """
    if p.run_this:
        return p

def preprocess_era5(p):
    """
    Preprocess ERA5 climate data to compute annual extreme rainfall metrics.
    1. Load raw ERA5 data
        format: NetCDF
        source: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download
        citation: 
    2. For each year, compute extreme rainfall metrics (e.g., annual maximum daily rainfall)
    3. Reproject and align with reference
    """
    if p.run_this:
        return p

def preprocess_esacci_to_veg(p):
    """
    Preprocess ESA CCI land cover data to create annual vegetation share rasters and counterfactual vegetation scenarios.
    1. Load raw ESA CCI land cover data
        format: GeoTIFF raster
        source: 
        citation: 
    2. For each year, calculate vegetation share (proportion of vegetated land cover types)
    3. Create counterfactual vegetation scenarios (e.g., no natural vegetation)
    4. Reproject and align with reference
    """
    if p.run_this:
        return p