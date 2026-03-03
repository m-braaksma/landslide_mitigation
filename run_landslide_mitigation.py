
import os
import hazelbean as hb
import subprocess
# from landslide_mitigation_tasks import slurm_tasks, local_tasks
from landslide_mitigation_tasks import local_tasks

# TO-DO: consider tobit or other censored regression

def build_landslide_mitigation_task_tree(p):
    # SLURM-based tasks
    if p.use_slurm:
        hb.log("Adding SLURM-based tasks to the project flow. Ensure access to a SLURM cluster and S3 bucket.")
        p.reproject_task = p.add_task(slurm_tasks.reproject_rasters, creates_dir=False)
        p.sdr_task = p.add_task(slurm_tasks.run_invest_sdr, creates_dir=False)
        p.rasterize_gfld_data = p.add_task(slurm_tasks.rasterize_gfld_data, creates_dir=False)
        p.align_rasters_task = p.add_task(slurm_tasks.align_rasters, creates_dir=False)

    # Process aligned rasters  
    p.preprocess_rasters_task = p.add_task(
        local_tasks.preprocess_aligned_rasters,
        creates_dir=False
    )

    # Generate tile zones (iterator)
    p.generate_tile_zones_task = p.add_iterator(
        local_tasks.tile_zones,
        run_in_parallel=p.run_in_parallel
    )

    # Single combined task for all regression and prediction work
    p.tile_analysis_task = p.add_task(
        local_tasks.tile_regression_and_prediction,
        parent=p.generate_tile_zones_task
    )

    # Stitch results
    p.stitch_avoided_mortality_raster_task = p.add_task(
        local_tasks.stitch_avoided_mortality_raster,
    )

    # Aggregate results
    p.aggregate_damage_results_task = p.add_task(
        local_tasks.aggregate_damage_results
    )

    return p

    
if __name__ == '__main__':
    hb.log('Starting landslide mitigation workflow...')
    
    # Create the ProjectFlow object
    p = hb.ProjectFlow()
    p.force_run = False
    p.L = hb.get_logger('landslide_mitigation_workflow')

    # Base directories
    p.base_data_dir = "Files/base_data/"
    p.proj_data_dir = "Files/base_data/gep_landslides/"
    p.user_dir = os.path.expanduser('~')        
    p.extra_dirs = ['Files', 'gep_landslides', 'projects']
    p.project_name = 'global_results'
    p.project_dir = os.path.join(p.user_dir, os.sep.join(p.extra_dirs), p.project_name)
    p.set_project_dir(p.project_dir)

    # S3 configuration
    p.s3_bucket = 'jajohns-tier2'
    p.s3_proj_dir = os.path.join('/vsis3', p.s3_bucket, 'Files', 'base_data', 'gep_landslides')
    # Load AWS S3 credentials
    aws_creds_path = os.path.expanduser('~/.aws_s3_credentials')
    if os.path.exists(aws_creds_path):
        with open(aws_creds_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('export '):
                    line = line[7:]  # Remove 'export '
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
        hb.log('✓ AWS S3 credentials loaded')
    else:
        hb.log('WARNING: ~/.aws_s3_credentials not found - S3 access may fail')
    os.environ['CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE'] = 'YES'  # Enable temp files for random writes to S3
    
    # Set operating system
    p.use_slurm = False

    # Processing parameters
    p.processing_resolution = 2000  # Tile size in pixels (increase for less output)
    p.run_in_parallel = True
    p.save_tile_geotiffs = False
    p.num_workers = 4

    # Build task tree
    build_landslide_mitigation_task_tree(p)
    
    # Run
    p.execute()