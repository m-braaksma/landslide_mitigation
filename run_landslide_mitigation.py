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


"""
GLOBAL LANDSLIDE MITIGATION MODEL
----------------------------------

CORE OBJECTIVE
Estimate the mortality-reducing value of vegetation via landslide hazard mitigation.

Conceptual structure:

    E[Deaths_it] =
        Pr(Landslide_it) *
        E[Deaths_it | Landslide_it]

Vegetation affects mortality ONLY through hazard probability.

------------------------------------------------------------
COMPUTATIONAL STRATEGY (CONFIRMED IMPLEMENTATION)
------------------------------------------------------------

STAGE 1: BUILD ESTIMATION DATASET (TABLE)
------------------------------------------
Unit: pixel-year
Resolution: 1 km
Time: annual (2007-2017)

A) Event observations:
    - Unified Global Landslide Catalogue (UGLC) aggregated to annual pixel indicator
    - Extract covariates at event pixel-year (vegetation, slope, rainfall, population)
    - Result: 15,016 event observations

B) Non-event observations:
    - Stratified random sampling with 50 controls per event
    - Stratified by GAEZ agro-ecological zone to reflect event distribution
    - Result: 750,793 control observations

Output: 765,809-row parquet table (table-based, NOT raster-tiled)

STAGE 2: ESTIMATE TWO-STAGE MODELS
---------------------------------------

Stage 1 - Hazard model (Logit):
    Pr(L_it = 1 | X) = logit(β0 + β1·slope + β2·twi + β3·road_density + β4·dist_to_fault + FE_gaez)
    with intercept correction from the control sampling fraction
    
    Result: Hazard probabilities used by the prediction pipeline

Stage 2 - Mortality model (Tweedie GLM):
    E[Deaths_it | X] = exp(γ0 + γ1·population_std + γ2·rain_std 
                            + γ3·forest_share_centered + γ4·othernat_share_centered 
                            + Σ γ_z·(GAEZ_zone == z))
    
    Fitted on: All 765,809 observations (events + controls)
    
    Forest effect: β_forest = -1.07
    Interpretation: A 10 percentage point increase in forest cover → ~10% reduction 
    in expected mortality.
    
    NOTE: NOT conditional on a landslide occurring. E[Deaths|X] includes both the probability 
    of landslide and severity given landslide, so vegetation's effect is population-level.

Output: Fitted coefficients and scaler params for global prediction

STAGE 3: GLOBAL PREDICTION (TILED)
-----------------------------------

For each of 164 land tiles:
    For each year (2007-2017):
        A) OBSERVED scenario:
            - Load observed vegetation, slope, rainfall, population rasters
            - Apply hazard and mortality models
            - Compute E[Deaths_it | observed vegetation]
            
        B) COUNTERFACTUAL scenario (noveg):
            - Set forest_share and othernat_share to zero
            - Keep all other covariates constant
            - Apply hazard and mortality models
            - Compute E[Deaths_it | noveg scenario]

Output: 164 tiles × 11 years × 2 scenarios = 3,608 GeoTIFFs

STAGE 4: STITCH & COMPUTE AVOIDED MORTALITY
---------------------------------------------

Stitch per-scenario global rasters:
    - observed_mortality_{year}.tif
    - counterfactual_mortality_{year}.tif

Compute avoided mortality:
    AvoidedMortality_it = E[Deaths_it | noveg] - E[Deaths_it | observed]
    
    Positive values indicate vegetation reduces mortality (as expected).

Output: 11 annual avoided mortality rasters (global, 1 km resolution)

------------------------------------------------------------
TEMPORAL STRUCTURE
------------------------------------------------------------

Annual analysis (not daily rasters to avoid memory explosion):
    - Annual max 1-day rainfall from ERA5
    - Vegetation cover from ESA-CCI LULC (annual)
    - Population from WorldPop (annual)
    - Static covariates: slope, TWI, GAEZ zones, roads, faults

------------------------------------------------------------
CRITICAL DESIGN PRINCIPLES
------------------------------------------------------------

✓ Estimation is table-based (no raster tiling)
✓ Prediction is raster-based (tile only for efficiency)
✓ Vegetation affects hazard probability (primary mechanism)
✓ Two-stage approach separates hazard from severity
✓ NoDataValue = -9999.0 for robust raster I/O

------------------------------------------------------------
END GOAL
------------------------------------------------------------

Produce a globally applicable ecosystem service valuation framework:

    Vegetation → Landslide Hazard Probability → 
    Conditional Severity → Expected Mortality → 
    (Later) Economic Valuation (VSL × avoided deaths)

"""