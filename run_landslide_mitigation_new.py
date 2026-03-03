import os
import hazelbean as hb
import pygeoprocessing as pygeo
from landslide_mitigation_tasks import input_data_tasks, model_tasks, post_model_tasks

# TO-DO: consider tobit or other censored regression

def build_landslide_mitigation_task_tree(p):
    # --------------------------------------------------
    # 1. PREPROCESS & BUILD ANNUAL PIXEL PANEL
    # --------------------------------------------------

    p.preprocess_base_data_task = p.add_task(input_data_tasks.preprocess_data, creates_dir=True)

    p.preprocess_nasa_glc_task = p.add_task(
        input_data_tasks.preprocess_nasa_glc,
        creates_dir=False
    )  # create annual pixel-level landslide panel

    p.preprocess_ls_factor_task = p.add_task(
        input_data_tasks.preprocess_lsfactor_gedtm,
        creates_dir=False
    )  # slope / LS-factor (time invariant)

    p.preprocess_worldpop_task = p.add_task(
        input_data_tasks.preprocess_worldpop,
        creates_dir=False
    )  # annual population raster

    p.preprocess_era5_task = p.add_task(
        input_data_tasks.preprocess_era5,
        creates_dir=False
    )  # compute annual extreme rainfall metrics

    p.preprocess_esacci_task = p.add_task(
        input_data_tasks.preprocess_esacci_to_veg,
        creates_dir=False
    )  # annual vegetation share + counterfactual vegetation

    # p.build_annual_pixel_panel_task = p.add_task(
    #     input_data_tasks.build_annual_pixel_panel
    # )

    # --------------------------------------------------
    # 2. ESTIMATE MODELS
    # --------------------------------------------------

    p.estimate_hazard_model_task = p.add_task(
        model_tasks.estimate_hazard_model
    )

    p.estimate_conditional_mortality_task = p.add_task(
        model_tasks.estimate_conditional_mortality_model
    )

    # --------------------------------------------------
    # 3. PREDICTION (OBSERVED + COUNTERFACTUAL)
    # --------------------------------------------------

    # p.generate_tile_zones_task = p.add_iterator(
    #     model_tasks.tile_zones,
    #     run_in_parallel=p.run_in_parallel
    # )

    # p.predict_observed_task = p.add_task(
    #     model_tasks.predict_observed,
    #     parent=p.generate_tile_zones_task
    # )

    # p.predict_counterfactual_task = p.add_task(
    #     model_tasks.predict_counterfactual,
    #     parent=p.generate_tile_zones_task
    # )

    # p.stitch_tiles_task = p.add_task(model_tasks.stitch_tiles)

    # --------------------------------------------------
    # 4. AGGREGATE & VALUE
    # --------------------------------------------------

    p.compute_avoided_mortality_task = p.add_task(
        post_model_tasks.compute_avoided_mortality
    )

    p.monetize_task = p.add_task(
        post_model_tasks.monetize_with_vsl
    )

    p.aggregate_results_task = p.add_task(
        post_model_tasks.aggregate_results
    )

    p.visualize_results_task = p.add_task(
        post_model_tasks.visualize_results,
        parent=p.aggregate_results_task
    )

    return p

    
if __name__ == '__main__':
    hb.log('Starting landslide mitigation workflow...')
    
    # Create the ProjectFlow object
    p = hb.ProjectFlow()
    p.force_run = False
    p.L = hb.get_logger('landslide_mitigation_workflow')

    # Base directories
    p.base_data_dir = "Files/base_data/landslide_mitigation"
    # p.proj_data_dir = "Files/base_data/gep_landslides/"
    p.user_dir = os.path.expanduser('~')   
    p.base_data_dir = os.path.join(p.user_dir, 'Files', 'base_data', 'landslide_mitigation')
    p.extra_dirs = ['Files', 'dissertation', 'projects']
    p.project_name = 'global_results'
    p.project_dir = os.path.join(p.user_dir, os.sep.join(p.extra_dirs), p.project_name)
    p.set_project_dir(p.project_dir)

    # S3 configuration
    # p.s3_bucket = 'jajohns-tier2'
    # p.s3_proj_dir = os.path.join('/vsis3', p.s3_bucket, 'Files', 'base_data', 'gep_landslides')
    # # Load AWS S3 credentials
    # aws_creds_path = os.path.expanduser('~/.aws_s3_credentials')
    # if os.path.exists(aws_creds_path):
    #     with open(aws_creds_path, 'r') as f:
    #         for line in f:
    #             line = line.strip()
    #             if line.startswith('export '):
    #                 line = line[7:]  # Remove 'export '
    #             if '=' in line and not line.startswith('#'):
    #                 key, value = line.split('=', 1)
    #                 value = value.strip().strip('"').strip("'")
    #                 os.environ[key] = value
    #     hb.log('✓ AWS S3 credentials loaded')
    # else:
    #     hb.log('WARNING: ~/.aws_s3_credentials not found - S3 access may fail')
    # os.environ['CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE'] = 'YES'  # Enable temp files for random writes to S3

    # Processing parameters
    p.processing_resolution = 2000  # Tile size in pixels (increase for less output)
    p.run_in_parallel = True
    p.save_tile_geotiffs = False
    p.num_workers = 4

    start_year = 2000
    end_year = 2020
    p.time_range = range(start_year, end_year+1)  # Temporal range for analysis
    p.max_location_accuracy_m = 1000  # Max location accuracy for GLC events (in meters)
    p.reference_raster_path = os.path.join(p.base_data_dir, 'worldpop', 'ppp_2000_1km_Aggregated.tif')  # Reference raster for alignment and metadata
    p.reference_raster_info = pygeo.get_raster_info(p.reference_raster_path)

    # Build task tree
    build_landslide_mitigation_task_tree(p)
    
    # Run
    p.execute()

hb.get_path()

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
COMPUTATIONAL STRATEGY
------------------------------------------------------------

IMPORTANT DISTINCTION:

1) MODEL ESTIMATION  -> table-based (no raster tiling required)
2) GLOBAL PREDICTION -> raster-based (must be tiled)

DO NOT run regressions on full global raster stacks.
DO NOT tile regressions.

------------------------------------------------------------
STAGE 1: BUILD ESTIMATION DATASET (TABLE)
------------------------------------------------------------

Unit: pixel-year
Resolution: 1 km
Time: annual

A) Event observations:
    - Use NASA Global Landslide Catalog (GLC)
    - Aggregate daily events -> annual pixel indicator
    - Extract covariates at event pixel-year:
        * Vegetation share
        * Slope / LS factor
        * Annual extreme rainfall
        * Population

B) Non-event observations:
    - Randomly sample pixel-years with no landslide
    - Extract same covariates
    - Use case-control sampling (rare event design)

Result:
    A manageable in-memory dataframe for regression
    (~10^5–10^6 rows, not billions)

------------------------------------------------------------
STAGE 2: ESTIMATE MODELS (GLOBAL COEFFICIENTS)
------------------------------------------------------------

Model 1: Hazard model (rare event)

    Pr(L_it = 1) =
        cloglog(
            β0
          + β1 VegShare_it
          + β2 Slope_i
          + β3 ExtremeRain_it
          + β4 VegShare_it * Slope_i
        )

Model 2: Conditional mortality model

    E[Deaths_it | L_it = 1] =
        exp(
            γ0
          + γ1 Population_it
          + γ2 ExtremeRain_it
        )

Store estimated coefficients.
These are global structural parameters.

------------------------------------------------------------
STAGE 3: GLOBAL PREDICTION (TILED)
------------------------------------------------------------

Now apply coefficients to full raster stack.

For each tile:
    1. Load vegetation, slope, rainfall, population rasters
    2. Compute hazard probability
    3. Compute conditional mortality
    4. Compute expected mortality
    5. Save tile output

Repeat for:

    A) Observed vegetation
    B) Counterfactual vegetation scenario

Stitch tiles into global rasters.

------------------------------------------------------------
STAGE 4: AVOIDED MORTALITY & VALUATION
------------------------------------------------------------

AvoidedMortality =
    E[Deaths]_counterfactual
  - E[Deaths]_observed

Then:

EconomicValue =
    AvoidedMortality * VSL_country

Aggregate:
    - Country
    - Admin units
    - Global totals

------------------------------------------------------------
TEMPORAL STRUCTURE
------------------------------------------------------------

DO NOT create daily raster panel.

Use daily ERA5 only to compute annual metrics:
    - Annual max 1-day rainfall
    - Annual total rainfall
    - Optional multi-day extreme metric

Final dataset remains annual.

------------------------------------------------------------
CRITICAL DESIGN PRINCIPLES
------------------------------------------------------------

- Estimation is table-based.
- Prediction is raster-based.
- Vegetation affects hazard, not vulnerability directly.
- Keep hazard and exposure conceptually separate.
- Avoid loading global raster stacks fully into memory.
- Tile only at prediction stage.

------------------------------------------------------------
END GOAL
------------------------------------------------------------

Produce a scalable, globally applicable ecosystem service
valuation framework linking:

    Vegetation -> Landslide Hazard -> Mortality -> Economic Value

"""