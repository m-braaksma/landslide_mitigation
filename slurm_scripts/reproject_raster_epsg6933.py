from osgeo import osr
import os
import sys
import pygeoprocessing
import logging
import shutil

from utils.s3_utils import s3_handler

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    # Get parameters from command line arguments
    if len(sys.argv) < 5:
        print("ERROR: Please provide required arguments")
        print("Usage: python reproject_raster_epsg6933.py <raster_name> <year> <input_s3_path> <output_s3_path> [--force]")
        sys.exit(1)
    
    raster_name = sys.argv[1]
    year = sys.argv[2]
    input_s3_path = sys.argv[3]
    output_s3_path = sys.argv[4]
    force_run = '--force' in sys.argv
    
    print(f'Processing {raster_name} for year {year}')
    print(f'Input S3 path: {input_s3_path}')
    print(f'Output S3 path: {output_s3_path}')
    
    # Define target projection (Equal Earth EPSG:6933)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(6933)
    equal_earth_wkt = srs.ExportToWkt()
    
    # Use s3_handler to manage the entire process
    workspace_name = f"reproject_{raster_name}_{year}"
    input_filename = f"{raster_name}_input.tif"
    output_filename = f"{raster_name}_epsg6933.tif"
    
    # Download, process, and upload using s3_handler
    with s3_handler.temp_workspace(workspace_name) as temp_dir:
        # Download input file
        local_input = s3_handler.download_to_temp(input_s3_path, input_filename)
        
        # Define output path in temp workspace
        local_output = s3_handler.get_temp_path(output_filename)
        
        # Get raster info
        raster_info = pygeoprocessing.get_raster_info(str(local_input))
        print(f'Input projection: {raster_info["projection_wkt"]}')
        
        # Check if reprojection is needed
        if raster_info['projection_wkt'] != equal_earth_wkt:
            print('Reprojecting to Equal Earth...')
            pygeoprocessing.warp_raster(
                str(local_input),
                (300, -300),  # 300m resolution
                str(local_output),
                'near',
                target_projection_wkt=equal_earth_wkt,
            )
            print(f'Saved reprojected raster to {local_output}')
        else:
            print('Projection already matches Equal Earth, copying file...')
            shutil.copy2(str(local_input), str(local_output))
        
        # Upload output file
        print("File exists before upload:", os.path.exists(local_output))
        print("File size:", os.path.getsize(local_output))
        s3_handler.upload_from_temp(output_filename, output_s3_path)
    
    print(f"Successfully processed {raster_name} for year {year}")

if __name__ == "__main__":
    main()