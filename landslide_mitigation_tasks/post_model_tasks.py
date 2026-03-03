import os
import sys

import hazelbean as hb
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from osgeo import gdal, ogr, osr
import pygeoprocessing as pygeo

def compute_avoided_mortality(p):
    """
    Compute avoided mortality by comparing observed and counterfactual predictions.
    1. Load observed and counterfactual predicted mortality rasters
    2. For each pixel, compute avoided mortality = predicted_observed - predicted_counterfactual
    3. Save avoided mortality raster
    """
    if p.run_this:
        return p
    

def monetize_with_vsl(p):
    """
    Monetize avoided mortality using Value of Statistical Life (VSL).
    1. Load avoided mortality raster
    2. Load VSL estimates (can be constant or vary by country/region)
    3. For each pixel, compute monetized value = avoided_mortality * VSL
    4. Save monetized value raster
    """
    if p.run_this:
        return p
    
def aggregate_results(p):
    """
    Aggregate results to compute total avoided mortality and monetized value at different spatial scales.
    1. Load monetized value raster
    2. Load population raster (to compute population-weighted metrics)
    3. For each administrative unit (e.g., country, region), sum monetized value and population
    4. Save aggregated results as CSV or GeoDataFrame
    """
    if p.run_this:
        return p

def visualize_results(p):
    """
    Create visualizations of the results.
    1. Load monetized value raster and/or aggregated results
    2. Create maps showing spatial distribution of avoided mortality and monetized value
    3. Create charts summarizing results by country/region
    4. Save visualizations to files
    """
    if p.run_this:
        return p
    
    