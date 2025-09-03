"""
Refactored zonal statistics script with fixed layer name handling
Usage: python3 scripts/run_zonal_stats.py <data_type> <year> [--counterfactual] [--output-dir-name OUTPUT] [--s3-output-base BASE] [--vector GAUL] [--sediment-export SED] [--worldpop WP] [--worldpop-reference WPR] [--era5-precip ERA5]
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


class ZonalStatsProcessor:
    """Handles zonal statistics processing with projection management."""
    
    def __init__(self, **paths):
        self.paths = paths
        self._validate_paths()
    
    def _validate_paths(self):
        """Validate that required paths are provided."""
        required = ['gaul', 's3_output_base', 'output_dir_name']
        missing = [p for p in required if not self.paths.get(p)]
        if missing:
            raise ValueError(f"Missing required path arguments: {missing}")
    
    def _get_gaul_layer_name(self, layer_type='level2'):
        """
        Return the appropriate GAUL layer name.
        GAUL files have layers: 'level0', 'level1', 'level2' (default administrative levels)
        """
        valid_layers = ['level0', 'level1', 'level2']
        if layer_type not in valid_layers:
            logger.warning(f"Invalid layer type '{layer_type}', using 'level2' as default")
            return 'level2'
        return layer_type
    
    def _check_and_align_projections(self, raster_path, vector_path, temp_dir, layer_name='level2'):
        """
        Check projections and reproject vector to raster CRS if needed.
        Returns path to potentially reprojected vector and the layer name.
        """
        # Get CRS info - read from specific GAUL layer
        vector_gdf = gpd.read_file(vector_path, layer=layer_name)
        vector_crs = vector_gdf.crs
        
        raster_info = pygeoprocessing.get_raster_info(raster_path)
        raster_crs_wkt = raster_info['projection_wkt']
        
        # Convert raster CRS to comparable format
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(raster_crs_wkt)
        raster_crs = raster_srs.GetAuthorityCode(None)
        
        logger.info(f"Vector CRS: {vector_crs}")
        logger.info(f"Raster CRS: EPSG:{raster_crs}")
        logger.info(f"Using GAUL layer: {layer_name}")

        # Handle missing/invalid CRS for ERA5 data
        if raster_crs is None or raster_crs == 'None':
            logger.warning("Raster has no CRS defined. Assuming ERA5 data with EPSG:4326")
            raster_crs = '4326'
        
        # Check if reprojection is needed
        vector_epsg = vector_crs.to_epsg() if hasattr(vector_crs, 'to_epsg') else None
        
        if vector_epsg and str(vector_epsg) != str(raster_crs):
            logger.warning(f"CRS mismatch detected. Reprojecting vector from EPSG:{vector_epsg} to EPSG:{raster_crs}")
            
            # Create reprojected vector path
            reprojected_path = os.path.join(temp_dir, f"reprojected_{os.path.basename(vector_path)}")
            
            # Reproject vector to raster CRS and save as GPKG preserving the layer name
            vector_gdf_reprojected = vector_gdf.to_crs(f"EPSG:{raster_crs}")
            vector_gdf_reprojected.to_file(reprojected_path, driver='GPKG', layer=layer_name)
            
            logger.info(f"Vector reprojected to: {reprojected_path} preserving layer: {layer_name}")
            return reprojected_path, layer_name
        else:
            logger.info("Projections are compatible, no reprojection needed")
            return vector_path, layer_name
    
    def _align_raster_to_reference(self, raster_path, reference_path, temp_dir):
        """Align raster to reference raster geometry."""
        aligned_path = os.path.join(temp_dir, f"aligned_{os.path.basename(raster_path)}")
        
        reference_info = pygeoprocessing.get_raster_info(reference_path)
        
        logger.info("Aligning raster to reference...")
        pygeoprocessing.align_and_resize_raster_stack(
            [raster_path],
            [aligned_path],
            resample_method_list=['average'],
            target_pixel_size=reference_info['pixel_size'],
            bounding_box_mode='union',
            gdal_warp_options=[
                'INIT_DEST=NO_DATA',
                'UNIFIED_SRC_NODATA=YES'
            ]
        )
        
        return aligned_path
    
    def _run_zonal_statistics(self, raster_path, vector_path, layer_name, temp_dir):
        """Run the actual zonal statistics computation."""
        logger.info(f"Running zonal statistics with GAUL layer: {layer_name}")
        
        zonal_stats = pygeoprocessing.geoprocessing.zonal_statistics(
            [(str(raster_path), 1)], 
            vector_path, 
            aggregate_layer_name=layer_name,
            ignore_nodata=True, 
            polygons_might_overlap=False, 
            include_value_counts=False, 
            working_dir=temp_dir
        )[0]  # Extract the single-element list
        
        return zonal_stats
    
    def process_data(self, data_type, year, counterfactual=False, gaul_layer='level2'):
        """
        Generic method to process any data type with proper projection handling.
        
        Args:
            data_type: Type of data to process
            year: Year to process
            counterfactual: Whether this is counterfactual analysis
            gaul_layer: Which GAUL administrative level to use ('level0', 'level1', 'level2')
        """
        # Validate and get the correct GAUL layer name
        layer_name = self._get_gaul_layer_name(gaul_layer)
        
        # Determine data-specific parameters
        data_config = self._get_data_config(data_type, year, counterfactual)
        
        # Check if output already exists
        output_s3_path = self.paths['s3_output_base'] + self.paths['output_dir_name']
        if s3_handler.file_exists(output_s3_path):
            logger.info(f"Output already exists: {output_s3_path}")
            return output_s3_path
        
        logger.info(f"Processing {data_type} zonal statistics for {year} using {layer_name}...")
        
        # Use s3_handler to manage the entire process
        with s3_handler.temp_workspace(data_config['workspace_name']) as temp_dir:
            # Download input files
            local_vector = s3_handler.download_to_temp(
                self.paths['gaul'], "gaul.gpkg"
            )
            local_raster = s3_handler.download_to_temp(
                data_config['raster_s3_path'], data_config['raster_filename']
            )
            
            # Convert to strings to ensure compatibility with pygeoprocessing
            local_vector = str(local_vector)
            local_raster = str(local_raster)
            
            # Handle worldpop alignment if needed (only for pixel size/extent alignment)
            if data_config.get('reference_needed'):
                local_reference = s3_handler.download_to_temp(
                    self.paths['worldpop_reference'], "ppp_reference.tif"
                )
                local_reference = str(local_reference)
                # Align to reference first (for pixel size/extent), then handle projections
                local_raster = self._align_raster_to_reference(
                    local_raster, local_reference, temp_dir
                )
                local_raster = str(local_raster)
            
            # Check and align projections (reproject vector to match raster CRS)
            aligned_vector, final_layer_name = self._check_and_align_projections(
                local_raster, local_vector, temp_dir, layer_name
            )
            aligned_vector = str(aligned_vector)
            
            # Run zonal statistics with the correct layer name
            zonal_stats = self._run_zonal_statistics(
                local_raster, aligned_vector, final_layer_name, temp_dir
            )
            
            # Save results
            local_output = s3_handler.get_temp_path(data_config['output_filename'])
            local_output = str(local_output)
            with open(local_output, 'wb') as f:
                pickle.dump(zonal_stats, f)
            
            logger.info("Zonal stats completed, uploading results...")
            
            # Upload output file
            output_filename = os.path.basename(local_output)
            s3_handler.upload_from_temp(output_filename, output_s3_path)
        
        logger.info(f"{data_type.title()} zonal stats saved to: {output_s3_path}")
        return output_s3_path
    
    def _get_data_config(self, data_type, year, counterfactual=False):
        """Get data-type specific configuration."""
        if data_type == 'sediment':
            config = {
                'raster_s3_path': self.paths['sediment_export'],
                'raster_filename': f"sed_export_{year}.tif",
                'output_filename': f"sed_export_zonal_stats_{year}.pkl",
                'workspace_name': f"zonal_sed_cf_{year}" if counterfactual else f"zonal_sed_{year}",
                'reference_needed': False
            }
            # Validate required path
            if not self.paths.get('sediment_export'):
                raise ValueError("Missing required path for sediment: sediment_export")
        
        elif data_type == 'worldpop':
            config = {
                'raster_s3_path': self.paths['worldpop'],
                'raster_filename': f"ppp_{year}_1km_Aggregated.tif",
                'output_filename': f"worldpop_zonal_stats_{year}.pkl",
                'workspace_name': f"zonal_pop_{year}",
                'reference_needed': True
            }
            # Validate required paths
            if not self.paths.get('worldpop'):
                raise ValueError("Missing required path for worldpop: worldpop")
            if not self.paths.get('worldpop_reference'):
                raise ValueError("Missing required path for worldpop: worldpop_reference")
        
        elif data_type == 'era5':
            config = {
                'raster_s3_path': self.paths['era5_precip'],
                'raster_filename': f"precip_{year}.tif",
                'output_filename': f"era5_precip_zonal_stats_{year}.pkl",
                'workspace_name': f"zonal_precip_{year}",
                'reference_needed': False
            }
            # Validate required path
            if not self.paths.get('era5_precip'):
                raise ValueError("Missing required path for era5: era5_precip")
        
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        return config


def run_zonal_stats(data_type, year, counterfactual=False, gaul_layer='level2', **paths):
    """
    Main function to run zonal statistics for any supported data type.
    
    Args:
        data_type (str): Type of data ('sediment', 'worldpop', 'era5')
        year (int): Year to process
        counterfactual (bool): Whether to run counterfactual analysis (sediment only)
        gaul_layer (str): GAUL administrative level ('level0', 'level1', 'level2')
        **paths: Dictionary of file paths
    
    Returns:
        str: Path to output file
    """
    if counterfactual and data_type != 'sediment':
        logger.warning(f"Counterfactual analysis not supported for {data_type} data")
        counterfactual = False
    
    processor = ZonalStatsProcessor(**paths)
    return processor.process_data(data_type, year, counterfactual, gaul_layer)


def main():
    parser = argparse.ArgumentParser(description='Run zonal statistics for sediment export, worldpop, or era5 data')
    parser.add_argument('data_type', choices=['sediment', 'worldpop', 'era5'], help='Type of data to process')
    parser.add_argument('year', type=int, help='Year to process')
    parser.add_argument('--counterfactual', action='store_true', help='Run counterfactual analysis (sediment only)')
    
    parser.add_argument('--gaul-layer', default='level2', choices=['level0', 'level1', 'level2'], 
                        help='GAUL administrative level to use (default: level2)')
    parser.add_argument('--output-dir-name', dest='output_dir_name', help='Output directory name')
    parser.add_argument('--s3-output-base', dest='s3_output_base', help='S3 output base path')
    parser.add_argument('--vector', dest='gaul', help='GAUL shapefile path')
    parser.add_argument('--sediment-export', dest='sediment_export', help='Sediment export raster path')
    parser.add_argument('--worldpop', dest='worldpop', help='Worldpop raster path')
    parser.add_argument('--worldpop-reference', dest='worldpop_reference', help='Worldpop reference raster path')
    parser.add_argument('--era5-precip', dest='era5_precip', help='ERA5 precipitation raster path')
    
    args = parser.parse_args()
    
    # Debug logging
    logger.info(f"Starting zonal statistics processing:")
    logger.info(f"  Data type: {args.data_type}")
    logger.info(f"  Year: {args.year}")
    logger.info(f"  Counterfactual: {args.counterfactual}")
    
    # Create paths dictionary from arguments, excluding the main parameters
    paths = {k: v for k, v in vars(args).items() 
             if v is not None and k not in ['data_type', 'year', 'counterfactual', 'gaul_layer']}
    
    logger.info(f"  Paths provided: {list(paths.keys())}")
    
    try:
        result = run_zonal_stats(args.data_type, args.year, args.counterfactual, args.gaul_layer, **paths)
        
        if result:
            logger.info(f"Successfully completed {args.data_type} zonal statistics for {args.year}")
            print(f"Successfully completed {args.data_type} zonal statistics for {args.year}")
        else:
            logger.error(f"Failed to complete {args.data_type} zonal statistics for {args.year}")
            print(f"Failed to complete {args.data_type} zonal statistics for {args.year}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"ERROR: {e}")
        print(f"ERROR: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# """
# Zonal statistics script for sediment export, worldpop, and era5 data
# Usage: python3 scripts/run_zonal_stats.py <data_type> <year> [--counterfactual] [--output-dir-name OUTPUT] [--s3-output-base BASE] [--vector GAUL] [--sediment-export SED] [--worldpop WP] [--worldpop-reference WPR] [--era5-precip ERA5]
# """

# from osgeo import gdal, osr
# import os
# import sys
# import geopandas as gpd
# import pandas as pd
# import numpy as np
# import pygeoprocessing
# import logging
# import pickle
# import argparse
# from pathlib import Path

# from utils.s3_utils import s3_handler

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S"
# )
# logger = logging.getLogger(__name__)


# def run_sediment_export_zonal_stats(year, counterfactual=False, **paths):
#     """Run zonal statistics for sediment export data."""
    
#     # Use provided paths (all should be provided via command line)
#     gaul_s3_path = paths.get('gaul')
#     sed_export_s3_path = paths.get('sediment_export')
#     output_s3_path = paths.get('s3_output_base') + paths.get('output_dir_name')
    
#     if not all([gaul_s3_path, sed_export_s3_path, paths.get('s3_output_base'), paths.get('output_dir_name')]):
#         raise ValueError("Missing required path arguments for sediment export zonal stats")
    
#     if counterfactual:
#         print(f"Preparing to run zonal stats for counterfactual sediment export ({year} no forest)...")
#         workspace_name = f"zonal_sed_cf_{year}"
#     else:
#         print(f"Preparing to run zonal stats for sediment export {year}...")
#         workspace_name = f"zonal_sed_{year}"
    
#     # Check if output already exists
#     if s3_handler.file_exists(output_s3_path):
#         print(f"Output already exists: {output_s3_path}")
#         return output_s3_path
    
#     # Use s3_handler to manage the entire process
#     with s3_handler.temp_workspace(workspace_name) as temp_dir:
#         # Download input files
#         local_gaul = s3_handler.download_to_temp(gaul_s3_path, "gaul.gpkg")
#         local_sed_export = s3_handler.download_to_temp(sed_export_s3_path, f"sed_export_{year}.tif")
        
#         # Define output path in temp workspace
#         local_output = s3_handler.get_temp_path(f"sed_export_zonal_stats_{year}.pkl")
        
#         # Run zonal statistics
#         print(f"Running zonal statistics for sediment export {year}...")
#         sed_export_zonal_stats = pygeoprocessing.geoprocessing.zonal_statistics(
#             [(str(local_sed_export), 1)], 
#             str(local_gaul), 
#             aggregate_layer_name='level2', 
#             ignore_nodata=True, 
#             polygons_might_overlap=False, 
#             include_value_counts=False, 
#             working_dir=str(temp_dir)
#         )[0]  # Extract the single-element list
        
#         # Save results to pickle
#         with open(local_output, 'wb') as f:
#             pickle.dump(sed_export_zonal_stats, f)
        
#         print(f"Zonal stats completed, uploading results...")
        
#         # Upload output file
#         output_filename = os.path.basename(local_output)
#         s3_handler.upload_from_temp(output_filename, output_s3_path)
    
#     print(f"Sediment export zonal stats saved to: {output_s3_path}")
#     return output_s3_path

# def run_worldpop_zonal_stats(year, **paths):
#     """Run zonal statistics for worldpop data."""
    
#     # Use provided paths (all should be provided via command line)
#     gaul_s3_path = paths.get('gaul')
#     worldpop_s3_path = paths.get('worldpop')
#     worldpop_reference_s3_path = paths.get('worldpop_reference')
#     output_s3_path = paths.get('s3_output_base') + paths.get('output_dir_name')
#     workspace_name = f"zonal_pop_{year}"
    
#     if not all([gaul_s3_path, worldpop_s3_path, worldpop_reference_s3_path, paths.get('s3_output_base'), paths.get('output_dir_name')]):
#         raise ValueError("Missing required path arguments for worldpop zonal stats")
    
#     # Check if output already exists
#     if s3_handler.file_exists(output_s3_path):
#         print(f"Output already exists: {output_s3_path}")
#         return output_s3_path
    
#     print(f"Preparing to run zonal stats for worldpop {year}...")
    
#     # Use s3_handler to manage the entire process
#     with s3_handler.temp_workspace(workspace_name) as temp_dir:
#         # Download input files
#         local_gaul = s3_handler.download_to_temp(gaul_s3_path, "gaul.gpkg")
#         local_worldpop = s3_handler.download_to_temp(worldpop_s3_path, f"ppp_{year}_1km_Aggregated.tif")
#         local_reference = s3_handler.download_to_temp(worldpop_reference_s3_path, "ppp_reference.tif")
        
#         # Define aligned path in temp workspace
#         local_worldpop_aligned = s3_handler.get_temp_path(f"ppp_{year}_1km_Aggregated_aligned.tif")
        
#         # Define output path in temp workspace
#         local_output = s3_handler.get_temp_path(f"worldpop_zonal_stats_{year}.pkl")
        
#         # Align raster if needed
#         print(f"Aligning worldpop raster for {year}...")
        
#         pygeoprocessing.align_and_resize_raster_stack(
#             [str(local_worldpop)],
#             [str(local_worldpop_aligned)],
#             resample_method_list=['average'],
#             target_pixel_size=pygeoprocessing.get_raster_info(str(local_reference))['pixel_size'],
#             bounding_box_mode='union',
#             gdal_warp_options=[
#                 'INIT_DEST=NO_DATA',
#                 'UNIFIED_SRC_NODATA=YES'
#             ]
#         )
        
#         # Run zonal statistics
#         print(f"Running zonal statistics for worldpop {year}...")
#         worldpop_zonal_stats = pygeoprocessing.geoprocessing.zonal_statistics(
#             [(str(local_worldpop_aligned), 1)], 
#             str(local_gaul), 
#             aggregate_layer_name='level2', 
#             ignore_nodata=True, 
#             polygons_might_overlap=False, 
#             include_value_counts=False, 
#             working_dir=None
#         )[0]  # Extract the single-element list
        
#         # Save results to pickle
#         with open(local_output, 'wb') as f:
#             pickle.dump(worldpop_zonal_stats, f)
        
#         print(f"Zonal stats completed, uploading results...")
        
#         # Upload output file
#         output_filename = os.path.basename(local_output)
#         s3_handler.upload_from_temp(output_filename, output_s3_path)
    
#     print(f"Worldpop zonal stats saved to: {output_s3_path}")
#     return output_s3_path

# def run_era5_zonal_stats(year, **paths):
#     """Run zonal statistics for ERA5 precipitation data."""
    
#     # Use provided paths (all should be provided via command line)
#     gaul_s3_path = paths.get('gaul')
#     era5_precip_s3_path = paths.get('era5_precip')
#     output_s3_path = paths.get('s3_output_base') + paths.get('output_dir_name')
#     workspace_name = f"zonal_precip_{year}"
    
#     if not all([gaul_s3_path, era5_precip_s3_path, paths.get('s3_output_base'), paths.get('output_dir_name')]):
#         raise ValueError("Missing required path arguments for ERA5 zonal stats")
    
#     # Check if output already exists
#     if s3_handler.file_exists(output_s3_path):
#         print(f"Output already exists: {output_s3_path}")
#         return output_s3_path
    
#     print(f"Preparing to run zonal stats for ERA5 precipitation {year}...")
    
#     # Use s3_handler to manage the entire process
#     with s3_handler.temp_workspace(workspace_name) as temp_dir:
#         # Download input files
#         local_gaul = s3_handler.download_to_temp(gaul_s3_path, "gaul.gpkg")
#         local_era5_precip = s3_handler.download_to_temp(era5_precip_s3_path, f"precip_{year}.tif")
        
#         # Define output path in temp workspace
#         local_output = s3_handler.get_temp_path(f"era5_precip_zonal_stats_{year}.pkl")
        
#         # Run zonal statistics
#         print(f"Running zonal statistics for ERA5 precipitation {year}...")
#         era5_zonal_stats = pygeoprocessing.geoprocessing.zonal_statistics(
#             [(str(local_era5_precip), 1)], 
#             str(local_gaul), 
#             aggregate_layer_name='level2', 
#             ignore_nodata=True, 
#             polygons_might_overlap=False, 
#             include_value_counts=False, 
#             working_dir=str(temp_dir)
#         )[0]  # Extract the single-element list
        
#         # Save results to pickle
#         with open(local_output, 'wb') as f:
#             pickle.dump(era5_zonal_stats, f)
        
#         print(f"Zonal stats completed, uploading results...")
        
#         # Upload output file
#         output_filename = os.path.basename(local_output)
#         s3_handler.upload_from_temp(output_filename, output_s3_path)
    
#     print(f"ERA5 precipitation zonal stats saved to: {output_s3_path}")
#     return output_s3_path

# def main():
#     parser = argparse.ArgumentParser(description='Run zonal statistics for sediment export, worldpop, or era5 data')
#     parser.add_argument('data_type', choices=['sediment', 'worldpop', 'era5'], help='Type of data to process')
#     parser.add_argument('year', type=int, help='Year to process')
#     parser.add_argument('--counterfactual', action='store_true', help='Run counterfactual analysis (sediment only)')
    
#     # Path arguments
#     parser.add_argument('--output-dir-name', help='Output directory name')
#     parser.add_argument('--s3-output-base', help='S3 output base path')
#     parser.add_argument('--vector', help='GAUL shapefile path')
#     parser.add_argument('--sediment-export', help='Sediment export raster path')
#     parser.add_argument('--worldpop', help='Worldpop raster path')
#     parser.add_argument('--worldpop-reference', help='Worldpop reference raster path')
#     parser.add_argument('--era5-precip', help='ERA5 precipitation raster path')
    
#     args = parser.parse_args()
    
#     # Create paths dictionary from arguments
#     paths = {}
#     if args.output_dir_name:
#         paths['output_dir_name'] = args.output_dir_name
#     if args.s3_output_base:
#         paths['s3_output_base'] = args.s3_output_base
#     if args.gaul:
#         paths['gaul'] = args.gaul
#     if args.sediment_export:
#         paths['sediment_export'] = args.sediment_export
#     if args.worldpop:
#         paths['worldpop'] = args.worldpop
#     if args.worldpop_reference:
#         paths['worldpop_reference'] = args.worldpop_reference
#     if args.era5_precip:
#         paths['era5_precip'] = args.era5_precip
    
#     try:
#         if args.data_type == 'sediment':
#             result = run_sediment_export_zonal_stats(args.year, args.counterfactual, **paths)
#         elif args.data_type == 'worldpop':
#             if args.counterfactual:
#                 print("WARNING: Counterfactual analysis not supported for worldpop data")
#             result = run_worldpop_zonal_stats(args.year, **paths)
#         elif args.data_type == 'era5':
#             if args.counterfactual:
#                 print("WARNING: Counterfactual analysis not supported for era5 data")
#             result = run_era5_zonal_stats(args.year, **paths)
        
#         if result:
#             print(f"Successfully completed {args.data_type} zonal statistics for {args.year}")
#         else:
#             print(f"Failed to complete {args.data_type} zonal statistics for {args.year}")
#             sys.exit(1)
            
#     except Exception as e:
#         print(f"ERROR: {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()
