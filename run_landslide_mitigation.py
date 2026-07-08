import os
import hazelbean as hb
import pygeoprocessing as pygeo
from landslide_mitigation_tasks import input_data_tasks, model_tasks, post_model_tasks

def build_landslide_mitigation_task_tree(p):
    # --------------------------------------------------
    # 1. PREPROCESS & BUILD ANNUAL PIXEL PANEL
    # --------------------------------------------------
    p.preprocess_base_data_task = p.add_task(
        input_data_tasks.preprocess_data, 
        creates_dir=False)

    p.preprocess_uglc_task = p.add_task(
        input_data_tasks.preprocess_uglc,
        parent=p.preprocess_base_data_task,
        creates_dir=False
    )  # create annual pixel-level landslide annual panel

    p.preprocess_geomorpho90m_task = p.add_task(
        input_data_tasks.preprocess_geomorpho90m,
        parent=p.preprocess_base_data_task,
        creates_dir=False
    )  # Geomorpho90m terrain variables: slope, roughness, tpi, elev_stdev, aspect

    p.preprocess_gedtm_task = p.add_task(
        input_data_tasks.preprocess_gedtm,
        parent=p.preprocess_base_data_task,
        creates_dir=False
    )  # LS-factor, slope, twi (time invariant)

    p.preprocess_gaez_task = p.add_task(
        input_data_tasks.preprocess_gaez,
        parent=p.preprocess_base_data_task,
        creates_dir=False
    )  # global agro-ecological suitability (time invariant)

    p.preprocess_landscan_task = p.add_task(
        input_data_tasks.preprocess_landscan,
        parent=p.preprocess_base_data_task,
        creates_dir=False
    )  # annual population raster annual panel

    p.preprocess_era5_task = p.add_task(
        input_data_tasks.preprocess_era5,
        parent=p.preprocess_base_data_task,
        creates_dir=False
    )  # extreme rainfall metrics annual panel

    p.preprocess_esacci_task = p.add_task(
        input_data_tasks.preprocess_esacci_to_share,
        parent=p.preprocess_base_data_task,
        creates_dir=False
    )  # annual vegetation share + counterfactual vegetation annual panel

    p.preprocess_deforestation_task = p.add_task(
        input_data_tasks.preprocess_deforestation_from_esacci,
        parent=p.preprocess_base_data_task,
        creates_dir=False
    )  # annual deforestation exposures from ESA-CCI (1yr + 3yr, coarse + refined)

    p.preprocess_sdr_task = p.add_task(
        input_data_tasks.preprocess_sdr,
        parent=p.preprocess_base_data_task,
        creates_dir=False
    ) # avoided erosion from invest SDR model annual panel

    p.preprocess_grip_roads_task = p.add_task(
        input_data_tasks.preprocess_grip_roads,
        parent=p.preprocess_base_data_task,
        creates_dir=False
    )  # global roads (time invariant)

    p.preprocess_gem_faults_task = p.add_task(
        input_data_tasks.preprocess_gem_faults,
        parent=p.preprocess_base_data_task,
        creates_dir=False
    )  # global active fault lines (time invariant)
    
    p.preprocess_travel_time_task = p.add_task(
        input_data_tasks.preprocess_travel_time,
        parent=p.preprocess_base_data_task,
        creates_dir=False
    )  # global travel time to healthcare (time invariant)

    # --------------------------------------------------
    # 2. ESTIMATE MODELS
    # --------------------------------------------------

    p.build_estimation_table_task = p.add_task(model_tasks.build_estimation_table)
    p.estimate_hazard_model_task = p.add_task(
        model_tasks.estimate_hazard_model
    )

    # --------------------------------------------------
    # 3. PREDICTION (OBSERVED + COUNTERFACTUAL)
    # --------------------------------------------------

    p.generate_tile_zones_task = p.add_iterator(
        model_tasks.tile_zones,
        run_in_parallel=p.run_in_parallel
    )

    # Prediction tasks (tile-level)
    p.predict_landslides_observed_task = p.add_task(
        model_tasks.predict_landslides_observed,
        parent=p.generate_tile_zones_task
    )

    p.predict_landslides_no_deforestation_task = p.add_task(
        model_tasks.predict_landslides_no_deforestation,
        parent=p.generate_tile_zones_task
    )

    p.predict_landslides_scenarios_task = p.add_task(
        model_tasks.predict_landslides_scenarios,
        parent=p.generate_tile_zones_task
    )

    p.predict_mortality_task = p.add_task(
        model_tasks.predict_mortality,
        parent=p.generate_tile_zones_task
    )

    p.stitch_tiles_task = p.add_task(model_tasks.stitch_tiles)

    # --------------------------------------------------
    # 4. AGGREGATE & VALUE
    # --------------------------------------------------

    p.compute_avoided_mortality_task = p.add_task(post_model_tasks.compute_avoided_mortality)
    p.compute_zonal_statistics_task = p.add_task(post_model_tasks.compute_zonal_statistics)

    p.visualizations_task = p.add_task(post_model_tasks.visualizations)
    p.plot_glc_from_vector_task = p.add_task(post_model_tasks.plot_glc_from_vector, creates_dir=False)
    p.plot_global_rasters_png_task = p.add_task(post_model_tasks.plot_global_rasters_png, creates_dir=False)    
    p.export_hazard_model_table_task = p.add_task(post_model_tasks.export_hazard_model_table, creates_dir=False)
    p.export_mortality_model_table_task = p.add_task(post_model_tasks.export_mortality_model_table, creates_dir=False)
    p.export_mortality_representative_cases_table_task = p.add_task(post_model_tasks.export_mortality_representative_cases_table, creates_dir=False)
    p.export_results_tables_task = p.add_task(post_model_tasks.export_results_tables, creates_dir=False)
    p.export_summary_stats_task = p.add_task(post_model_tasks.export_summary_stats, creates_dir=False)
    p.sync_to_dissertation_task = p.add_task(post_model_tasks.sync_to_dissertation, creates_dir=False)

    return p


if __name__ == '__main__':
    hb.log('Starting landslide mitigation workflow...')
    
    # Create the ProjectFlow object
    p = hb.ProjectFlow()
    p.force_run = False
    p.L = hb.get_logger('landslide_mitigation_workflow')

    # DIR configuration
    p.base_data = ['Files', 'base_data', 'landslide_mitigation']
    project_dir = ['Files', 'landslide_mitigation']
    p.project_name = 'global_results'
    p.user_dir = os.path.expanduser('~')
    p.base_data_dir = os.path.join(p.user_dir, *p.base_data)
    p.project_dir = os.path.join(p.user_dir, *project_dir, p.project_name)
    p.set_project_dir(p.project_dir)

    p.dissertation_dir = os.path.join(p.user_dir, 'Files', 'dissertation', 'dissertation', 'assets', 'landslide_mitigation')

    # S3 configuration
    # p.s3_bucket = 'jajohns-tier2'
    # p.s3_data_dir = os.path.join('/vsis3', p.s3_bucket, *base_data_dir)
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
    os.environ['AWS_S3_MULTIPART_THRESHOLD'] = '1GB'  # Effectively disables multipart for most files

    # Processing parameters
    p.processing_resolution = 2000  # Tile size in pixels (increase for less output)
    p.run_in_parallel = True
    # p.save_tile_geotiffs = False
    p.num_workers = 8

    start_year = 2007
    end_year = 2017
    p.time_range = range(start_year, end_year+1)  # Temporal range for analysis
    p.deforestation_max_lag_years = 3
    p.estimation_start_year = 2010  # use full-lag support window for deforestation exposures
    p.prediction_years = [2017]
    p.max_location_accuracy_m = 1000  # Max location accuracy for GLC events (in meters)
    p.reference_raster_path = input_data_tasks.ensure_global_reference_raster(p)
    p.reference_raster_info = pygeo.get_raster_info(p.reference_raster_path)
    p.control_ratio = 25   # controls per event
    p.include_grassland_stage2 = False  # keep grassland out of default refined Stage 2 spec
    
    # Forest value scenarios: (name, defor_1yr_rate, defor_3yr_rate)
    # Counterfactual deforestation is forest_share × rate
    p.forest_value_scenarios = [
        ('cf_5pct_annual', 0.05, 0.15),   # 5% annual → 15% cumulative over 3yr
        ('cf_10pct_annual', 0.10, 0.30),  # 10% annual → 30% cumulative over 3yr
    ]

    # Build task tree
    build_landslide_mitigation_task_tree(p)
    
    # Run
    p.execute()
