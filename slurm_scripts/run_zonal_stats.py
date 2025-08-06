"""
Zonal statistics script for sediment export, worldpop, and era5 data
Usage: python3 scripts/run_zonal_stats.py <data_type> <year> [--counterfactual] [--output-dir-name OUTPUT] [--s3-output-base BASE] [--gadm GADM] [--sediment-export SED] [--worldpop WP] [--worldpop-reference WPR] [--era5-precip ERA5]
"""

from osgeo import gdal, osr
import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import pygeoprocessing
import logging
import pickle
import argparse
from pathlib import Path

from utils.s3_utils import s3_handler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S"
)
logger = logging.getLogger(__name__)


def run_sediment_export_zonal_stats(year, counterfactual=False, **paths):
    """Run zonal statistics for sediment export data."""
    
    # Use provided paths (all should be provided via command line)
    gadm_s3_path = paths.get('gadm')
    sed_export_s3_path = paths.get('sediment_export')
    output_s3_path = paths.get('s3_output_base') + paths.get('output_dir_name')
    
    if not all([gadm_s3_path, sed_export_s3_path, paths.get('s3_output_base'), paths.get('output_dir_name')]):
        raise ValueError("Missing required path arguments for sediment export zonal stats")
    
    if counterfactual:
        print(f"Preparing to run zonal stats for counterfactual sediment export ({year} no forest)...")
        workspace_name = f"zonal_sed_cf_{year}"
    else:
        print(f"Preparing to run zonal stats for sediment export {year}...")
        workspace_name = f"zonal_sed_{year}"
    
    # Check if output already exists
    if s3_handler.file_exists(output_s3_path):
        print(f"Output already exists: {output_s3_path}")
        return output_s3_path
    
    # Use s3_handler to manage the entire process
    with s3_handler.temp_workspace(workspace_name) as temp_dir:
        # Download input files
        local_gadm = s3_handler.download_to_temp(gadm_s3_path, "gadm_410-levels.gpkg")
        local_sed_export = s3_handler.download_to_temp(sed_export_s3_path, f"sed_export_{year}.tif")
        
        # Define output path in temp workspace
        local_output = s3_handler.get_temp_path(f"sed_export_zonal_stats_{year}.pkl")
        
        # Run zonal statistics
        print(f"Running zonal statistics for sediment export {year}...")
        sed_export_zonal_stats = pygeoprocessing.geoprocessing.zonal_statistics(
            [(str(local_sed_export), 1)], 
            str(local_gadm), 
            aggregate_layer_name='ADM_2', 
            ignore_nodata=True, 
            polygons_might_overlap=False, 
            include_value_counts=False, 
            working_dir=str(temp_dir)
        )[0]  # Extract the single-element list
        
        # Save results to pickle
        with open(local_output, 'wb') as f:
            pickle.dump(sed_export_zonal_stats, f)
        
        print(f"Zonal stats completed, uploading results...")
        
        # Upload output file
        output_filename = os.path.basename(local_output)
        s3_handler.upload_from_temp(output_filename, output_s3_path)
    
    print(f"Sediment export zonal stats saved to: {output_s3_path}")
    return output_s3_path

def run_worldpop_zonal_stats(year, **paths):
    """Run zonal statistics for worldpop data."""
    
    # Use provided paths (all should be provided via command line)
    gadm_s3_path = paths.get('gadm')
    worldpop_s3_path = paths.get('worldpop')
    worldpop_reference_s3_path = paths.get('worldpop_reference')
    output_s3_path = paths.get('s3_output_base') + paths.get('output_dir_name')
    workspace_name = f"zonal_pop_{year}"
    
    if not all([gadm_s3_path, worldpop_s3_path, worldpop_reference_s3_path, paths.get('s3_output_base'), paths.get('output_dir_name')]):
        raise ValueError("Missing required path arguments for worldpop zonal stats")
    
    # Check if output already exists
    if s3_handler.file_exists(output_s3_path):
        print(f"Output already exists: {output_s3_path}")
        return output_s3_path
    
    print(f"Preparing to run zonal stats for worldpop {year}...")
    
    # Use s3_handler to manage the entire process
    with s3_handler.temp_workspace(workspace_name) as temp_dir:
        # Download input files
        local_gadm = s3_handler.download_to_temp(gadm_s3_path, "gadm_410-levels.gpkg")
        local_worldpop = s3_handler.download_to_temp(worldpop_s3_path, f"ppp_{year}_1km_Aggregated.tif")
        local_reference = s3_handler.download_to_temp(worldpop_reference_s3_path, "ppp_reference.tif")
        
        # Define aligned path in temp workspace
        local_worldpop_aligned = s3_handler.get_temp_path(f"ppp_{year}_1km_Aggregated_aligned.tif")
        
        # Define output path in temp workspace
        local_output = s3_handler.get_temp_path(f"worldpop_zonal_stats_{year}.pkl")
        
        # Align raster if needed
        print(f"Aligning worldpop raster for {year}...")
        
        pygeoprocessing.align_and_resize_raster_stack(
            [str(local_worldpop)],
            [str(local_worldpop_aligned)],
            resample_method_list=['average'],
            target_pixel_size=pygeoprocessing.get_raster_info(str(local_reference))['pixel_size'],
            bounding_box_mode='union',
            gdal_warp_options=[
                'INIT_DEST=NO_DATA',
                'UNIFIED_SRC_NODATA=YES'
            ]
        )
        
        # Run zonal statistics
        print(f"Running zonal statistics for worldpop {year}...")
        worldpop_zonal_stats = pygeoprocessing.geoprocessing.zonal_statistics(
            [(str(local_worldpop_aligned), 1)], 
            str(local_gadm), 
            aggregate_layer_name='ADM_2', 
            ignore_nodata=True, 
            polygons_might_overlap=False, 
            include_value_counts=False, 
            working_dir=None
        )[0]  # Extract the single-element list
        
        # Save results to pickle
        with open(local_output, 'wb') as f:
            pickle.dump(worldpop_zonal_stats, f)
        
        print(f"Zonal stats completed, uploading results...")
        
        # Upload output file
        output_filename = os.path.basename(local_output)
        s3_handler.upload_from_temp(output_filename, output_s3_path)
    
    print(f"Worldpop zonal stats saved to: {output_s3_path}")
    return output_s3_path

def run_era5_zonal_stats(year, **paths):
    """Run zonal statistics for ERA5 precipitation data."""
    
    # Use provided paths (all should be provided via command line)
    gadm_s3_path = paths.get('gadm')
    era5_precip_s3_path = paths.get('era5_precip')
    output_s3_path = paths.get('s3_output_base') + paths.get('output_dir_name')
    workspace_name = f"zonal_precip_{year}"
    
    if not all([gadm_s3_path, era5_precip_s3_path, paths.get('s3_output_base'), paths.get('output_dir_name')]):
        raise ValueError("Missing required path arguments for ERA5 zonal stats")
    
    # Check if output already exists
    if s3_handler.file_exists(output_s3_path):
        print(f"Output already exists: {output_s3_path}")
        return output_s3_path
    
    print(f"Preparing to run zonal stats for ERA5 precipitation {year}...")
    
    # Use s3_handler to manage the entire process
    with s3_handler.temp_workspace(workspace_name) as temp_dir:
        # Download input files
        local_gadm = s3_handler.download_to_temp(gadm_s3_path, "gadm_410-levels.gpkg")
        local_era5_precip = s3_handler.download_to_temp(era5_precip_s3_path, f"precip_{year}.tif")
        
        # Define output path in temp workspace
        local_output = s3_handler.get_temp_path(f"era5_precip_zonal_stats_{year}.pkl")
        
        # Run zonal statistics
        print(f"Running zonal statistics for ERA5 precipitation {year}...")
        era5_zonal_stats = pygeoprocessing.geoprocessing.zonal_statistics(
            [(str(local_era5_precip), 1)], 
            str(local_gadm), 
            aggregate_layer_name='ADM_2', 
            ignore_nodata=True, 
            polygons_might_overlap=False, 
            include_value_counts=False, 
            working_dir=str(temp_dir)
        )[0]  # Extract the single-element list
        
        # Save results to pickle
        with open(local_output, 'wb') as f:
            pickle.dump(era5_zonal_stats, f)
        
        print(f"Zonal stats completed, uploading results...")
        
        # Upload output file
        output_filename = os.path.basename(local_output)
        s3_handler.upload_from_temp(output_filename, output_s3_path)
    
    print(f"ERA5 precipitation zonal stats saved to: {output_s3_path}")
    return output_s3_path

def main():
    parser = argparse.ArgumentParser(description='Run zonal statistics for sediment export, worldpop, or era5 data')
    parser.add_argument('data_type', choices=['sediment', 'worldpop', 'era5'], help='Type of data to process')
    parser.add_argument('year', type=int, help='Year to process')
    parser.add_argument('--counterfactual', action='store_true', help='Run counterfactual analysis (sediment only)')
    
    # Path arguments
    parser.add_argument('--output-dir-name', help='Output directory name')
    parser.add_argument('--s3-output-base', help='S3 output base path')
    parser.add_argument('--gadm', help='GADM shapefile path')
    parser.add_argument('--sediment-export', help='Sediment export raster path')
    parser.add_argument('--worldpop', help='Worldpop raster path')
    parser.add_argument('--worldpop-reference', help='Worldpop reference raster path')
    parser.add_argument('--era5-precip', help='ERA5 precipitation raster path')
    
    args = parser.parse_args()
    
    # Create paths dictionary from arguments
    paths = {}
    if args.output_dir_name:
        paths['output_dir_name'] = args.output_dir_name
    if args.s3_output_base:
        paths['s3_output_base'] = args.s3_output_base
    if args.gadm:
        paths['gadm'] = args.gadm
    if args.sediment_export:
        paths['sediment_export'] = args.sediment_export
    if args.worldpop:
        paths['worldpop'] = args.worldpop
    if args.worldpop_reference:
        paths['worldpop_reference'] = args.worldpop_reference
    if args.era5_precip:
        paths['era5_precip'] = args.era5_precip
    
    try:
        if args.data_type == 'sediment':
            result = run_sediment_export_zonal_stats(args.year, args.counterfactual, **paths)
        elif args.data_type == 'worldpop':
            if args.counterfactual:
                print("WARNING: Counterfactual analysis not supported for worldpop data")
            result = run_worldpop_zonal_stats(args.year, **paths)
        elif args.data_type == 'era5':
            if args.counterfactual:
                print("WARNING: Counterfactual analysis not supported for era5 data")
            result = run_era5_zonal_stats(args.year, **paths)
        
        if result:
            print(f"Successfully completed {args.data_type} zonal statistics for {args.year}")
        else:
            print(f"Failed to complete {args.data_type} zonal statistics for {args.year}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
