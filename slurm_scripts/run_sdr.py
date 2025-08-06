import logging
import sys
import argparse
import shutil
from pathlib import Path

from utils.s3_utils import s3_handler
import natcap.invest.sdr.sdr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S"
)
logger = logging.getLogger(__name__)

def upload_directory_to_s3(local_dir: Path, s3_base_path: str):
    """Recursively upload all files in local_dir to S3 under s3_base_path"""
    for file_path in local_dir.rglob('*'):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_dir)
            s3_path = f"{s3_base_path}/{relative_path.as_posix()}"
            logger.info(f"Uploading {file_path} to {s3_path}")
            s3_handler.upload_from_temp(str(file_path), s3_path)


def run_sdr_for_year(year: int, s3_paths: dict, output_dir_name: str, s3_output_base: str, 
                     counterfactual_run: bool = False, force_run: bool = False):
    """Run SDR model with provided paths (no config imports needed)"""
    logger.info(f"Starting SDR model run for year {year}")
    
    # Only check if force_run is provided - otherwise assume we want to run since pre-check was done
    if not force_run:
        s3_sed_export_path = s3_output_base + "/sed_export.tif"
        if s3_handler.file_exists(s3_sed_export_path):
            logger.info(f"Output {s3_sed_export_path} exists and force_run=False. Skipping processing.")
            sys.exit(0)

    with s3_handler.temp_workspace(f"sdr_{year}") as workspace_dir:
        workspace_dir = Path(workspace_dir)

        local_files = {}
        for key, s3_path in s3_paths.items():
            filename = s3_path.split("/")[-1]
            local_path = workspace_dir / filename
            logger.info(f"Downloading {key} from {s3_path} to {local_path}")
            s3_handler.download_to_temp(s3_path, filename)
            local_files[key] = str(local_path)

        sdr_args = {
            "biophysical_table_path": local_files["biophysical_table"],
            "dem_path": local_files["dem"],
            "drainage_path": "",
            "erodibility_path": local_files["erodibility"],
            "erosivity_path": local_files["erosivity"],
            "ic_0_param": "0.5",
            "k_param": "2",
            "l_max": "122",
            "lulc_path": local_files["lulc"],
            "n_workers": "-1",
            "results_suffix": "",
            "sdr_max": "0.8",
            "threshold_flow_accumulation": "1000",
            "watersheds_path": local_files["watersheds"],
            "workspace_dir": str(workspace_dir / output_dir_name),
            "flow_dir_algorithm": "D8",
        }

        local_output_dir = workspace_dir / output_dir_name
        sdr_success = False
        
        try:
            logger.info(f"Running SDR model for year {year}")
            natcap.invest.sdr.sdr.execute(sdr_args)
            logger.info(f"SDR model completed successfully for year {year}")
            sdr_success = True
            
        except Exception as e:
            logger.error(f"SDR model failed during execution for year {year}: {e}")
            logger.warning("Proceeding with upload of partial results...")
            if local_output_dir.exists():
                logger.info(f"Output directory exists, will attempt to upload partial results")
            else:
                logger.error(f"No output directory found at {local_output_dir}")
                raise

        # Upload results regardless of whether SDR completed fully
        if local_output_dir.exists():
            logger.info(f"Uploading SDR output directory {local_output_dir} to S3 at {s3_output_base}")
            try:
                upload_directory_to_s3(local_output_dir, s3_output_base)
                logger.info(f"Successfully uploaded results to S3")
            except Exception as upload_error:
                logger.error(f"Failed to upload results to S3: {upload_error}")
                raise
        else:
            logger.error(f"No output directory found at {local_output_dir} - cannot upload results")
            raise Exception("No results to upload")

        # Log final status
        if sdr_success:
            logger.info(f"SDR processing and upload completed successfully for year {year}")
        else:
            logger.warning(f"SDR processing failed but partial results were uploaded for year {year}")


def main():
    parser = argparse.ArgumentParser(description='Run SDR model for a specific year')
    parser.add_argument('year', type=int, help='Year to process')
    parser.add_argument('--counterfactual', action='store_true', help='Run counterfactual scenario')
    parser.add_argument('--force', action='store_true', help='Force run even if output exists')
    
    # Path arguments (passed from the main script)
    parser.add_argument('--output-dir-name', required=True, help='Output directory name')
    parser.add_argument('--s3-output-base', required=True, help='S3 output base path')
    parser.add_argument('--lulc', required=True, help='LULC file path')
    parser.add_argument('--biophysical-table', required=True, help='Biophysical table path')
    parser.add_argument('--dem', required=True, help='DEM file path')
    parser.add_argument('--erodibility', required=True, help='Erodibility file path')
    parser.add_argument('--erosivity', required=True, help='Erosivity file path')
    parser.add_argument('--watersheds', required=True, help='Watersheds file path')
    
    args = parser.parse_args()
    
    # Build s3_paths dict from arguments
    s3_paths = {
        "lulc": args.lulc,
        "biophysical_table": args.biophysical_table,
        "dem": args.dem,
        "erodibility": args.erodibility,
        "erosivity": args.erosivity,
        "watersheds": args.watersheds,
    }
    
    run_sdr_for_year(
        year=args.year,
        s3_paths=s3_paths,
        output_dir_name=args.output_dir_name,
        s3_output_base=args.s3_output_base,
        counterfactual_run=args.counterfactual,
        force_run=args.force
    )


if __name__ == "__main__":
    main()

# import logging
# import sys
# import shutil
# from pathlib import Path

# import natcap.invest.sdr.sdr

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S"
# )
# logger = logging.getLogger(__name__)

# project_root = Path(__file__).parent.parent
# sys.path.append(str(project_root))

# from config.config import PROJ_DATA_DIR
# from utils.s3_utils import s3_handler

# def upload_directory_to_s3(local_dir: Path, s3_base_path: str):
#     """Recursively upload all files in local_dir to S3 under s3_base_path"""
#     for file_path in local_dir.rglob('*'):
#         if file_path.is_file():
#             relative_path = file_path.relative_to(local_dir)
#             s3_path = f"{s3_base_path}/{relative_path.as_posix()}"
#             logger.info(f"Uploading {file_path} to {s3_path}")
#             s3_handler.upload_from_temp(str(file_path), s3_path)

# def run_sdr_for_year(year: int, counterfactual_run: bool = False, force_run: bool = False):
#     logger.info(f"Starting SDR model run for year {year}")
#     INVEST_INPUT_DIR = PROJ_DATA_DIR + "invest_sdr_input/"
#     s3_paths = {
#         "lulc": INVEST_INPUT_DIR + f"lulc_{year}_epsg8857.tif",
#         "biophysical_table": INVEST_INPUT_DIR + "expanded_biophysical_table_gura.csv" if not counterfactual_run else INVEST_INPUT_DIR + "expanded_biophysical_table_gura_CF1.csv",
#         "dem": INVEST_INPUT_DIR + "alt_m_epsg8857.tif",
#         "erodibility": INVEST_INPUT_DIR + "soil_erodibility_epsg8857.tif",
#         "erosivity": INVEST_INPUT_DIR + "global_erosivity_epsg8857.tif",
#         "watersheds": INVEST_INPUT_DIR + "hybas_global_lev06_v1c.gpkg",
#     }

#     # Check inputs exist
#     missing = [(k, v) for k, v in s3_paths.items() if not s3_handler.file_exists(v)]
#     if missing:
#         for k, v in missing:
#             logger.error(f"Missing required input: {k} at path: s3://{s3_handler.bucket_name}/{v}")
#         sys.exit(1)

#     output_dir_name = f"invest_sdr_output/lulc_esa_{year}" if not counterfactual_run else f"invest_sdr_output/lulc_esa_{year}_cf"
#     s3_output_base = PROJ_DATA_DIR + output_dir_name 

#     # Check if sed_export.tif already exists
#     s3_sed_export_path = s3_output_base + "/sed_export.tif"
#     if s3_handler.file_exists(s3_sed_export_path) and not force_run:
#         print(f"Output {s3_sed_export_path} exists. Skipping processing.")
#         sys.exit(0)

#     # # Check if output already exists
#     # if s3_handler.dir_exists(s3_output_base) and not force_run:
#     #     print(f"Output {s3_output_base} exists. Skipping processing.")
#     #     sys.exit(0)

#     with s3_handler.temp_workspace(f"sdr_{year}") as workspace_dir:
#         workspace_dir = Path(workspace_dir)

#         local_files = {}
#         for key, s3_path in s3_paths.items():
#             filename = s3_path.split("/")[-1]
#             local_path = workspace_dir / filename
#             logger.info(f"Downloading {key} from {s3_path} to {local_path}")
#             s3_handler.download_to_temp(s3_path, filename)
#             local_files[key] = str(local_path)

#         sdr_args = {
#             "biophysical_table_path": local_files["biophysical_table"],
#             "dem_path": local_files["dem"],
#             "drainage_path": "",
#             "erodibility_path": local_files["erodibility"],
#             "erosivity_path": local_files["erosivity"],
#             "ic_0_param": "0.5",
#             "k_param": "2",
#             "l_max": "122",
#             "lulc_path": local_files["lulc"],
#             "n_workers": "-1",
#             "results_suffix": "",
#             "sdr_max": "0.8",
#             "threshold_flow_accumulation": "1000",
#             "watersheds_path": local_files["watersheds"],
#             "workspace_dir": str(workspace_dir / output_dir_name),
#             "flow_dir_algorithm": "D8",
#         }

#         local_output_dir = workspace_dir / output_dir_name
#         sdr_success = False
        
#         try:
#             logger.info(f"Running SDR model for year {year}")
#             natcap.invest.sdr.sdr.execute(sdr_args)
#             logger.info(f"SDR model completed successfully for year {year}")
#             sdr_success = True
            
#         except Exception as e:
#             logger.error(f"SDR model failed during execution for year {year}: {e}")
#             logger.warning("Proceeding with upload of partial results...")
#             # Check if we have any output files to upload
#             if local_output_dir.exists():
#                 logger.info(f"Output directory exists, will attempt to upload partial results")
#             else:
#                 logger.error(f"No output directory found at {local_output_dir}")
#                 raise  # Re-raise if we have nothing to upload

#         # Upload results regardless of whether SDR completed fully
#         if local_output_dir.exists():
#             logger.info(f"Uploading SDR output directory {local_output_dir} to S3 at {s3_output_base}")
#             try:
#                 upload_directory_to_s3(local_output_dir, s3_output_base)
#                 logger.info(f"Successfully uploaded results to S3")
#             except Exception as upload_error:
#                 logger.error(f"Failed to upload results to S3: {upload_error}")
#                 raise
#         else:
#             logger.error(f"No output directory found at {local_output_dir} - cannot upload results")
#             raise Exception("No results to upload")

#         # Log final status
#         if sdr_success:
#             logger.info(f"SDR processing and upload completed successfully for year {year}")
#         else:
#             logger.warning(f"SDR processing failed but partial results were uploaded for year {year}")

# def main():
#     if len(sys.argv) < 2 or len(sys.argv) > 4:
#         print("Usage: python run_sdr.py <year> [--counterfactual] [--force]")
#         sys.exit(1)

#     try:
#         year = int(sys.argv[1])
#     except ValueError:
#         print("Year must be an integer")
#         sys.exit(1)

#     counterfactual_run = '--counterfactual' in sys.argv
#     force_run = '--force' in sys.argv

#     run_sdr_for_year(year, counterfactual_run, force_run)


# if __name__ == "__main__":
#     main()




    
# import logging
# import sys
# import shutil
# from pathlib import Path

# import natcap.invest.sdr.sdr

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S"
# )
# logger = logging.getLogger(__name__)

# project_root = Path(__file__).parent.parent
# sys.path.append(str(project_root))

# from config.config import PROJ_DATA_DIR
# from utils.s3_utils import s3_handler

# def upload_directory_to_s3(local_dir: Path, s3_base_path: str):
#     """Recursively upload all files in local_dir to S3 under s3_base_path"""
#     for file_path in local_dir.rglob('*'):
#         if file_path.is_file():
#             relative_path = file_path.relative_to(local_dir)
#             s3_path = f"{s3_base_path}/{relative_path.as_posix()}"
#             logger.info(f"Uploading {file_path} to {s3_path}")
#             s3_handler.upload_from_temp(str(file_path), s3_path)

# def run_sdr_for_year(year: int, counterfactual_run: bool = False, force_run: bool = False):
#     logger.info(f"Starting SDR model run for year {year}")
#     INVEST_INPUT_DIR = PROJ_DATA_DIR + "invest_sdr_input/"
#     s3_paths = {
#         "lulc": INVEST_INPUT_DIR + f"lulc_{year}_epsg8857.tif",
#         "biophysical_table": INVEST_INPUT_DIR + "expanded_biophysical_table_gura.csv" if not counterfactual_run else INVEST_INPUT_DIR + "expanded_biophysical_table_gura_CF1.csv",
#         "dem": INVEST_INPUT_DIR + "alt_m_epsg8857.tif",
#         "erodibility": INVEST_INPUT_DIR + "soil_erodibility_epsg8857.tif",
#         "erosivity": INVEST_INPUT_DIR + "global_erosivity_epsg8857.tif",
#         "watersheds": INVEST_INPUT_DIR + "hybas_global_lev06_v1c.gpkg",
#     }

#     # Check inputs exist
#     missing = [(k, v) for k, v in s3_paths.items() if not s3_handler.file_exists(v)]
#     if missing:
#         for k, v in missing:
#             logger.error(f"Missing required input: {k} at path: s3://{s3_handler.bucket_name}/{v}")
#         sys.exit(1)

#     output_dir_name = f"invest_sdr_output/lulc_esa_{year}"
#     s3_output_base = PROJ_DATA_DIR + output_dir_name

#     # Check if output already exists
#     if s3_handler.dir_exists(s3_output_base) and not force_run:
#         print(f"Output {s3_output_base} exists. Skipping processing.")
#         sys.exit(0)

#     with s3_handler.temp_workspace(f"sdr_{year}") as workspace_dir:
#         workspace_dir = Path(workspace_dir)

#         local_files = {}
#         for key, s3_path in s3_paths.items():
#             filename = s3_path.split("/")[-1]
#             local_path = workspace_dir / filename
#             logger.info(f"Downloading {key} from {s3_path} to {local_path}")
#             s3_handler.download_to_temp(s3_path, filename)
#             local_files[key] = str(local_path)

#         sdr_args = {
#             "biophysical_table_path": local_files["biophysical_table"],
#             "dem_path": local_files["dem"],
#             "drainage_path": "",
#             "erodibility_path": local_files["erodibility"],
#             "erosivity_path": local_files["erosivity"],
#             "ic_0_param": "0.5",
#             "k_param": "2",
#             "l_max": "122",
#             "lulc_path": local_files["lulc"],
#             "n_workers": "-1",
#             "results_suffix": "",
#             "sdr_max": "0.8",
#             "threshold_flow_accumulation": "1000",
#             "watersheds_path": local_files["watersheds"],
#             "workspace_dir": str(workspace_dir / output_dir_name),
#             "flow_dir_algorithm": "D8",
#         }

#         try:
#             logger.info(f"Running SDR model for year {year}")
#             natcap.invest.sdr.sdr.execute(sdr_args)
#             logger.info(f"SDR model completed successfully for year {year}")

#             # After successful run, upload entire output folder to S3
#             local_output_dir = workspace_dir / output_dir_name
#             logger.info(f"Uploading SDR output directory {local_output_dir} to S3 at {s3_output_base}")
#             upload_directory_to_s3(local_output_dir, s3_output_base)

#         except Exception as e:
#             logger.error(f"Error running SDR for year {year}: {e}")
#             raise

# def main():
#     if len(sys.argv) < 2 or len(sys.argv) > 4:
#         print("Usage: python run_sdr.py <year> [--counterfactual] [--force]")
#         sys.exit(1)

#     try:
#         year = int(sys.argv[1])
#     except ValueError:
#         print("Year must be an integer")
#         sys.exit(1)

#     counterfactual_run = '--counterfactual' in sys.argv
#     force_run = '--force' in sys.argv

#     run_sdr_for_year(year, counterfactual_run, force_run)


# if __name__ == "__main__":
#     main()
