import os
import sys
import hazelbean as hb
import subprocess
import re
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import pickle

from utils.s3_utils import s3_handler

# Reset GDAL_DATA path after importing hazelbean
gdal_data_path = os.environ.get("GDAL_DATA")
print("GDAL_DATA:", gdal_data_path)
os.environ['GDAL_DATA'] = '/users/0/braak014/miniforge3/envs/teems02/share/gdal'


def submit_slurm_job(command, job_dir, job_name, time="01:00:00", mem="8000M", tmp="4000M", cpus=1, dependency=None):
    """Submit a SLURM job with specified parameters using the correct format."""
    
    # Create log directory
    log_dir = Path("slurm_logs") / job_dir / job_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build the wrapped command with environment setup
    wrap_cmd = (
        "echo \"Job started at $(date)\"; "
        "source ~/miniforge3/etc/profile.d/conda.sh && "
        "conda activate teems02 && "
        "cd /users/0/braak014/Files/gep_landslides/landslide_mitigation && "  # Set to project root
        f"{command}; "
        "if [ $? -eq 0 ]; then "
        "echo \"Job completed successfully at $(date)\"; "
        "else "
        "echo \"Job FAILED at $(date)\" >&2; "
        "fi"
    )

    # Build SLURM command
    slurm_cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output={log_dir}/%j.out",
        "--partition=msismall",
        f"--time={time}",
        f"--mem={mem}",
        f"--tmp={tmp}",
        f"--cpus-per-task={cpus}",
    ]

    if dependency:
        if isinstance(dependency, list):
            dependency_str = ":".join(dependency)
        else:
            dependency_str = dependency
        slurm_cmd.append(f"--dependency=afterok:{dependency_str}")
    
    slurm_cmd.extend(["--wrap", wrap_cmd])

    # Submit job and capture output
    result = subprocess.run(slurm_cmd, capture_output=True, text=True, check=True)
    
    # Parse the job ID from sbatch output
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = match.group(1)
        hb.log(f"Submitted job {job_name} with Job ID: {job_id}")
        return job_id
    else:
        raise RuntimeError(f"Could not parse job ID from sbatch output:\n{result.stdout}")

def reproject_rasters(p):
    """Reproject all input rasters to EPSG:6933."""
    if p.run_this:
        hb.log("Checking existing outputs and submitting reprojection jobs...")
        job_ids = []
        job_dir = "01_reproject_raster_epsg6933"
        
        # Track statistics
        total_rasters = len(p.sdr_input_rasters.keys())
        skipped_rasters = []
        submitted_count = 0
        
        for raster_name in p.sdr_input_rasters.keys():
            # Generate expected output S3 path
            input_s3_path = p.base_data_dir + p.sdr_input_rasters[raster_name]
            output_s3_path = p.proj_data_dir + f"invest_sdr_input/{raster_name}_epsg6933.tif"
            
            # Check if output already exists on S3
            if s3_handler.file_exists(output_s3_path) and not getattr(p, 'force_run', False):
                skipped_rasters.append(raster_name)
                continue
            
            # Determine command based on raster type
            if raster_name.startswith("lulc_"):
                year = raster_name.split("_")[1]
                command = f"python3 slurm_scripts/reproject_raster_epsg6933.py {raster_name} {year} {input_s3_path} {output_s3_path}"
                job_name = f"{raster_name}"
                time = "01:00:00"
            else:
                command = f'python3 slurm_scripts/reproject_raster_epsg6933.py "{raster_name}" static "{input_s3_path}" "{output_s3_path}"'
                # command = f"python3 slurm_scripts/reproject_raster_epsg6933.py {raster_name} static {input_s3_path} {output_s3_path}"
                job_name = f"{raster_name}"
                time = "00:20:00"
            
            job_id = submit_slurm_job(
                command,
                job_dir,
                job_name,
                time=time,
                mem="8000M",
                cpus=1,
            )
            
            job_ids.append(job_id)
            submitted_count += 1
        
        p.reprojection_job_ids = job_ids
        hb.log(f"Reprojection SLURM summary:")
        hb.log(f"\tTotal rasters: {total_rasters}")
        hb.log(f"\tSkipped (outputs exist): {len(skipped_rasters)}")
        hb.log(f"\tSubmitted jobs: {submitted_count}")
        if skipped_rasters:
            hb.log(f"\tSkipped reprojection jobs for existing outputs (rasters: {', '.join(skipped_rasters)})")
        
    return p

def get_sdr_paths(input_dir: str, year: int, counterfactual_run: bool = False):
    """Get all SDR-related paths for a given year and run type"""

    invest_input_dir = input_dir + "invest_sdr_input/"
    
    paths = {
        "input_paths": {
            "lulc": invest_input_dir + f"lulc_{year}_epsg6933.tif",
            "biophysical_table": invest_input_dir + "expanded_biophysical_table_gura.csv" if not counterfactual_run else invest_input_dir + "expanded_biophysical_table_gura_CF1.csv",
            "dem": invest_input_dir + "alt_m_epsg6933.tif",
            "erodibility": invest_input_dir + "soil_erodibility_epsg6933.tif",
            "erosivity": invest_input_dir + "global_erosivity_epsg6933.tif",
            "watersheds": invest_input_dir + "hybas_global_lev06_v1c.gpkg",
        },
        "output_dir_name": f"invest_sdr_output/lulc_esa_{year}" if not counterfactual_run else f"invest_sdr_output/lulc_esa_{year}_cf",
        "s3_output_base": None,
        "s3_sed_export_path": None,
    }
    
    paths["s3_output_base"] = input_dir + paths["output_dir_name"]
    paths["s3_sed_export_path"] = paths["s3_output_base"] + "/sed_export.tif"
    
    return paths

def check_sdr_inputs_exist(paths_dict):
    """Check if all required SDR inputs exist in S3"""
    missing = []
    for key, s3_path in paths_dict["input_paths"].items():
        if not s3_handler.file_exists(s3_path):
            missing.append((key, s3_path))
    
    return missing


def check_sdr_output_exists(input_dir: str, year: int, counterfactual_run: bool = False):
    """Check if SDR output already exists in S3"""
    paths = get_sdr_paths(input_dir, year, counterfactual_run)
    return s3_handler.file_exists(paths["s3_sed_export_path"])


def run_invest_sdr(p):
    """Run SDR model for all years including counterfactual."""
    if p.run_this:
        hb.log("Checking existing outputs and submitting SDR jobs...")
        job_ids = []
        job_dir = "02_run_sdr"
        dependency_str = ":".join(p.reprojection_job_ids)
        
        # Track statistics
        total_jobs = len(p.years) * 2  # Regular + counterfactual for each year
        submitted_count = 0
        skipped_regular = []
        skipped_counterfactual = []
        
        # Check inputs exist for all years first
        input_errors = []
        for year in p.years:
            # Check regular run inputs
            paths = get_sdr_paths(input_dir=p.proj_data_dir, year=year, counterfactual_run=False)
            missing = check_sdr_inputs_exist(paths)
            if missing:
                for key, s3_path in missing:
                    input_errors.append(f"Missing input for year {year}: {key} at {s3_path}")
            
            # Check counterfactual run inputs
            cf_paths = get_sdr_paths(input_dir=p.proj_data_dir, year=year, counterfactual_run=True)
            cf_missing = check_sdr_inputs_exist(cf_paths)
            if cf_missing:
                for key, s3_path in cf_missing:
                    input_errors.append(f"Missing counterfactual input for year {year}: {key} at {s3_path}")
        
        if input_errors:
            for error in input_errors:
                hb.log(f"ERROR: {error}")
            raise ValueError("Missing required inputs for SDR processing")
        
        # Regular year jobs
        for year in p.years:
            # Check if output already exists
            if check_sdr_output_exists(input_dir=p.proj_data_dir, year=year, counterfactual_run=False):
                skipped_regular.append(year)
                continue
                
            # Get paths for this year and pass them as arguments
            paths = get_sdr_paths(input_dir=p.proj_data_dir, year=year, counterfactual_run=False)
            
            # Create command with all paths as arguments
            command = (
                f"python3 slurm_scripts/run_sdr.py {year} "
                f"--output-dir-name '{paths['output_dir_name']}' "
                f"--s3-output-base '{paths['s3_output_base']}' "
                f"--lulc '{paths['input_paths']['lulc']}' "
                f"--biophysical-table '{paths['input_paths']['biophysical_table']}' "
                f"--dem '{paths['input_paths']['dem']}' "
                f"--erodibility '{paths['input_paths']['erodibility']}' "
                f"--erosivity '{paths['input_paths']['erosivity']}' "
                f"--watersheds '{paths['input_paths']['watersheds']}'"
            )
            
            job_name = f"sdr_{year}"
            job_id = submit_slurm_job(
                command,
                job_dir,
                job_name,
                time="16:00:00",
                mem="100GB",
                tmp="100GB",
                cpus=8,
                dependency=dependency_str,
            )
            job_ids.append(job_id)
            submitted_count += 1
            hb.log(f"Submitted SDR job for year {year} (Job ID: {job_id})")
        
        # Counterfactual runs
        for year in p.years:
            # Check if counterfactual output already exists
            if check_sdr_output_exists(input_dir=p.proj_data_dir, year=year, counterfactual_run=True):
                skipped_counterfactual.append(year)
                continue
            
            # Get paths for counterfactual run
            paths = get_sdr_paths(input_dir=p.proj_data_dir, year=year, counterfactual_run=True)
            
            # Create command with all paths as arguments
            command = (
                f"python3 slurm_scripts/run_sdr.py {year} --counterfactual "
                f"--output-dir-name '{paths['output_dir_name']}' "
                f"--s3-output-base '{paths['s3_output_base']}' "
                f"--lulc '{paths['input_paths']['lulc']}' "
                f"--biophysical-table '{paths['input_paths']['biophysical_table']}' "
                f"--dem '{paths['input_paths']['dem']}' "
                f"--erodibility '{paths['input_paths']['erodibility']}' "
                f"--erosivity '{paths['input_paths']['erosivity']}' "
                f"--watersheds '{paths['input_paths']['watersheds']}'"
            )
            
            job_name = f"sdr_cf_{year}"
            job_id = submit_slurm_job(
                command,
                job_dir,
                job_name,
                time="16:00:00",
                mem="100GB",
                tmp="100GB",
                cpus=8,
                dependency=dependency_str,
            )
            job_ids.append(job_id)
            submitted_count += 1
            hb.log(f"Submitted SDR counterfactual job for year {year} (Job ID: {job_id})")
        
        # Print summary
        hb.log(f"SDR SLURM summary:")
        hb.log(f"\tTotal jobs: {total_jobs}")
        hb.log(f"\tSkipped (outputs exist): {len(skipped_regular) + len(skipped_counterfactual)}")
        hb.log(f"\tSubmitted jobs: {submitted_count}")
        
        if skipped_regular:
            hb.log(f"\tSkipped SDR jobs for existing outputs (years: {', '.join(map(str, skipped_regular))})")
        if skipped_counterfactual:
            hb.log(f"\tSkipped SDR counterfactual jobs for existing outputs (years: {', '.join(map(str, skipped_counterfactual))})")
        
        p.sdr_job_ids = job_ids
    
    return p


def get_zonal_stats_paths(input_dir, year, data_type, counterfactual_run=False):
    """Get all required paths for zonal statistics processing."""
    paths = {
        'output_dir_name': '',
        's3_output_base': input_dir + "zonal_stats/",
        'input_paths': {}
    }
    
    # Common inputs
    paths['input_paths']['gaul'] = input_dir + "emdat/gaul2014_2015.gpkg"
    
    if data_type == 'sediment':
        if counterfactual_run:
            paths['output_dir_name'] = f'sed_export_cf_{year}_zonal_stats.pkl'
            paths['input_paths']['sediment_export'] = input_dir + f"invest_sdr_output/lulc_esa_{year}_cf/sed_export.tif"
        else:
            paths['output_dir_name'] = f'sed_export_{year}_zonal_stats.pkl'
            paths['input_paths']['sediment_export'] = input_dir + f"invest_sdr_output/lulc_esa_{year}/sed_export.tif"
    
    elif data_type == 'worldpop':
        paths['output_dir_name'] = f'worldpop_{year}_zonal_stats.pkl'
        paths['input_paths']['worldpop'] = input_dir + f"worldpop/ppp_{year}_1km_Aggregated.tif"
        paths['input_paths']['worldpop_reference'] = input_dir + "worldpop/ppp_2000_1km_Aggregated.tif"
    
    elif data_type == 'era5':
        paths['output_dir_name'] = f'era5_precip_{year}_zonal_stats.pkl'
        paths['input_paths']['era5_precip'] = input_dir + f"era5_precip/ERA5_annual_precip_{year}.tif"  # Adjust path as needed
    
    return paths

def check_zonal_stats_inputs_exist(paths):
    """Check if all required input files exist for zonal statistics."""
    missing = []
    
    for key, s3_path in paths['input_paths'].items():
        if not s3_handler.file_exists(s3_path):
            missing.append((key, s3_path))
    
    return missing

def check_zonal_stats_output_exists(input_dir, year, data_type, counterfactual_run=False):
    """Check if zonal statistics output already exists."""
    if data_type == 'sediment':
        if counterfactual_run:
            output_path = input_dir + f"zonal_stats/sed_export_cf_{year}_zonal_stats.pkl"
        else:
            output_path = input_dir + f"zonal_stats/sed_export_{year}_zonal_stats.pkl"
    elif data_type == 'worldpop':
        output_path = input_dir + f"zonal_stats/worldpop_{year}_zonal_stats.pkl"
    elif data_type == 'era5':
        output_path = input_dir + f"zonal_stats/era5_precip_{year}_zonal_stats.pkl"
    
    return s3_handler.file_exists(output_path)

def run_zonal_stats(p):
    """Run zonal statistics for sediment and population data."""
    if p.run_this:
        hb.log("Checking existing outputs and submitting zonal statistics jobs...")
        job_ids = []
        job_dir = "03_zonal_stats"
        dependency_str = ":".join(p.sdr_job_ids)
        
        # Track statistics
        total_jobs = len(p.years) * 4  # sediment, counterfactual, worldpop, era5 for each year
        submitted_count = 0
        skipped_sediment = []
        skipped_counterfactual = []
        skipped_worldpop = []
        skipped_era5 = []
        
        # Check inputs exist for all data types and years first
        input_errors = []
        
        # Check sediment export inputs (regular and counterfactual)
        for year in p.years:
            # Regular sediment export
            paths = get_zonal_stats_paths(input_dir=p.proj_data_dir, year=year, data_type='sediment', counterfactual_run=False)
            missing = check_zonal_stats_inputs_exist(paths)
            if missing:
                for key, s3_path in missing:
                    input_errors.append(f"Missing sediment input for year {year}: {key} at {s3_path}")
            
            # Counterfactual sediment export
            cf_paths = get_zonal_stats_paths(input_dir=p.proj_data_dir, year=year, data_type='sediment', counterfactual_run=True)
            cf_missing = check_zonal_stats_inputs_exist(cf_paths)
            if cf_missing:
                for key, s3_path in cf_missing:
                    input_errors.append(f"Missing counterfactual sediment input for year {year}: {key} at {s3_path}")
        
        # Check worldpop inputs
        for year in p.years:
            paths = get_zonal_stats_paths(input_dir=p.proj_data_dir, year=year, data_type='worldpop')
            missing = check_zonal_stats_inputs_exist(paths)
            if missing:
                for key, s3_path in missing:
                    input_errors.append(f"Missing worldpop input for year {year}: {key} at {s3_path}")
        
        # Check era5 inputs
        for year in p.years:
            paths = get_zonal_stats_paths(input_dir=p.proj_data_dir, year=year, data_type='era5')
            missing = check_zonal_stats_inputs_exist(paths)
            if missing:
                for key, s3_path in missing:
                    input_errors.append(f"Missing era5 input for year {year}: {key} at {s3_path}")
        
        if input_errors:
            for error in input_errors:
                hb.log(f"ERROR: {error}")
            raise ValueError("Missing required inputs for zonal statistics processing")
        
        # Submit sediment export zonal stats jobs
        for year in p.years:
            # Check if output already exists
            if check_zonal_stats_output_exists(input_dir=p.proj_data_dir, year=year, data_type='sediment', counterfactual_run=False):
                skipped_sediment.append(year)
                continue
            
            # Get paths for this year and pass them as arguments
            paths = get_zonal_stats_paths(input_dir=p.proj_data_dir, year=year, data_type='sediment', counterfactual_run=False)
            
            # Create command with all paths as arguments
            command = (
                f"python3 slurm_scripts/run_zonal_stats.py sediment {year} "
                f"--output-dir-name '{paths['output_dir_name']}' "
                f"--s3-output-base '{paths['s3_output_base']}' "
                f"--vector '{paths['input_paths']['gaul']}' "
                f"--sediment-export '{paths['input_paths']['sediment_export']}'"
            )
            
            job_name = f"zonal_sed_{year}"
            job_id = submit_slurm_job(
                command,
                job_dir,
                job_name,
                time="02:00:00",
                mem="16GB",
                tmp="8GB",
                cpus=2,
                dependency=dependency_str,
            )
            job_ids.append(job_id)
            submitted_count += 1
            hb.log(f"Submitted sediment zonal stats job for year {year}")
        
        # Submit counterfactual sediment export zonal stats jobs
        for year in p.years:
            # Check if counterfactual output already exists
            if check_zonal_stats_output_exists(input_dir=p.proj_data_dir, year=year, data_type='sediment', counterfactual_run=True):
                skipped_counterfactual.append(year)
                continue
            
            # Get paths for counterfactual run
            paths = get_zonal_stats_paths(input_dir=p.proj_data_dir, year=year, data_type='sediment', counterfactual_run=True)
            
            # Create command with all paths as arguments
            command = (
                f"python3 slurm_scripts/run_zonal_stats.py sediment {year} --counterfactual "
                f"--output-dir-name '{paths['output_dir_name']}' "
                f"--s3-output-base '{paths['s3_output_base']}' "
                f"--vector '{paths['input_paths']['gaul']}' "
                f"--sediment-export '{paths['input_paths']['sediment_export']}'"
            )
            
            job_name = f"zonal_sed_cf_{year}"
            job_id = submit_slurm_job(
                command,
                job_dir,
                job_name,
                time="02:00:00",
                mem="16GB",
                tmp="8GB",
                cpus=2,
                dependency=dependency_str,
            )
            job_ids.append(job_id)
            submitted_count += 1
            hb.log(f"Submitted counterfactual sediment zonal stats job for year {year}")
        
        # Submit worldpop zonal stats jobs
        for year in p.years:
            # Check if output already exists
            if check_zonal_stats_output_exists(input_dir=p.proj_data_dir, year=year, data_type='worldpop'):
                skipped_worldpop.append(year)
                continue
            
            # Get paths for this year and pass them as arguments
            paths = get_zonal_stats_paths(input_dir=p.proj_data_dir, year=year, data_type='worldpop')
            
            # Create command with all paths as arguments
            command = (
                f"python3 slurm_scripts/run_zonal_stats.py worldpop {year} "
                f"--output-dir-name '{paths['output_dir_name']}' "
                f"--s3-output-base '{paths['s3_output_base']}' "
                f"--vector '{paths['input_paths']['gaul']}' "
                f"--worldpop '{paths['input_paths']['worldpop']}' "
                f"--worldpop-reference '{paths['input_paths']['worldpop_reference']}'"
            )
            
            job_name = f"zonal_pop_{year}"
            job_id = submit_slurm_job(
                command,
                job_dir,
                job_name,
                time="02:00:00",
                mem="16GB",
                tmp="8GB",
                cpus=2,
                dependency=dependency_str,
            )
            job_ids.append(job_id)
            submitted_count += 1
            hb.log(f"Submitted worldpop zonal stats job for year {year}")
        
        # Submit era5 precip zonal stats jobs
        for year in p.years:
            # Check if output already exists
            if check_zonal_stats_output_exists(input_dir=p.proj_data_dir, year=year, data_type='era5'):
                skipped_era5.append(year)
                continue
            
            # Get paths for this year and pass them as arguments
            paths = get_zonal_stats_paths(input_dir=p.proj_data_dir, year=year, data_type='era5')
            
            # Create command with all paths as arguments
            command = (
                f"python3 slurm_scripts/run_zonal_stats.py era5 {year} "
                f"--output-dir-name '{paths['output_dir_name']}' "
                f"--s3-output-base '{paths['s3_output_base']}' "
                f"--vector '{paths['input_paths']['gaul']}' "
                f"--era5-precip '{paths['input_paths']['era5_precip']}'"
            )
            
            job_name = f"zonal_precip_{year}"
            job_id = submit_slurm_job(
                command,
                job_dir,
                job_name,
                time="02:00:00",
                mem="16GB",
                tmp="8GB",
                cpus=2,
                dependency=dependency_str,
            )
            job_ids.append(job_id)
            submitted_count += 1
            hb.log(f"Submitted ERA5 zonal stats job for year {year}")
        
        # Print summary
        hb.log(f"Zonal stats SLURM summary:")
        hb.log(f"\tTotal jobs: {total_jobs}")
        hb.log(f"\tSkipped (outputs exist): {len(skipped_sediment) + len(skipped_counterfactual) + len(skipped_worldpop) + len(skipped_era5)}")
        hb.log(f"\tSubmitted jobs: {submitted_count}")
        
        if skipped_sediment:
            hb.log(f"\tSkipped sediment zonal stats jobs for existing outputs (years: {', '.join(map(str, skipped_sediment))})")
        if skipped_counterfactual:
            hb.log(f"\tSkipped counterfactual sediment zonal stats jobs for existing outputs (years: {', '.join(map(str, skipped_counterfactual))})")
        if skipped_worldpop:
            hb.log(f"\tSkipped worldpop zonal stats jobs for existing outputs (years: {', '.join(map(str, skipped_worldpop))})")
        if skipped_era5:
            hb.log(f"\tSkipped ERA5 zonal stats jobs for existing outputs (years: {', '.join(map(str, skipped_era5))})")
        
        p.zonal_stats_job_ids = job_ids
    
    return p