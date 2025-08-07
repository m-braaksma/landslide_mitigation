
import os
import hazelbean as hb
from landslide_mitigation_tasks import slurm_tasks, local_tasks


def build_landslide_mitigation_task_tree(p):
    # SLURM-based tasks
    if p.use_slurm:
        hb.log("Adding SLURM-based tasks to the project flow. Ensure access to a SLURM cluster and S3 bucket.")
        p.reproject_task = p.add_task(slurm_tasks.reproject_raster, task_name="reproject_rasters", creates_dir=False)
        p.sdr_task = p.add_task(slurm_tasks.run_sdr, task_name="run_sdr_model", creates_dir=False)
        p.zonal_stats_task = p.add_task(slurm_tasks.run_zonal_stats, task_name="run_zonal_stats", creates_dir=False)
        p.combine_task = p.add_task(slurm_tasks.combine_zonal_stats, task_name="combine_zonal_stats", creates_dir=True)
    # Local tasks
    p.damage_function_task = p.add_task(local_tasks.damage_function_task, task_name="damage_function_analysis", creates_dir=True)
    
    return p

if __name__ == '__main__':
    hb.log('Starting SLURM-based landslide workflow...')
    
    # Create the ProjectFlow object
    p = hb.ProjectFlow()
    p.force_run = False
    p.L = hb.get_logger('landslide_slurm_workflow')

    # Base directories
    p.base_data_dir = "Files/base_data/"
    p.proj_data_dir = "Files/base_data/gep_landslides/"
    p.user_dir = os.path.expanduser('~')        
    p.extra_dirs = ['Files', 'gep_landslides', 'projects']
    p.project_name = 'global_results'
    p.project_dir = os.path.join(p.user_dir, os.sep.join(p.extra_dirs), p.project_name)
    p.set_project_dir(p.project_dir)

    try:
        from config.config import SDR_INPUT_RASTERS, YEARS, COUNTERFACTUAL_YEAR
        p.sdr_input_rasters = SDR_INPUT_RASTERS
        p.years = YEARS
        p.counterfactual_year = COUNTERFACTUAL_YEAR
    except ImportError:
        hb.log("Warning: Could not import config.config, using fallback configuration", level='WARNING')
        p.years = list(range(2000, 2021))
        p.counterfactual_year = 2019
        p.sdr_input_rasters = {
            'alt_m': 'seals/static_regressors/alt_m.tif',
            'global_erosivity': 'global_invest/sediment_delivery/Global Erosivity/GlobalR_NoPol-002.tif',
            'soil_erodibility': 'global_invest/sediment_delivery/Global Soil Erodibility/Data_25km/RUSLE_KFactor_v1.1_25km.tif',
        }
        p.sdr_input_rasters.update({
            f'lulc_{year}': f'lulc/esa/lulc_esa_{year}.tif' for year in p.years
        })
    
    # Set operating system
    p.use_slurm = True

    # Build task tree
    build_landslide_mitigation_task_tree(p)
    
    # Run
    p.execute() 