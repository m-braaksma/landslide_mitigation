import os
import hazelbean as hb
from landslide_mitigation_tasks import (
    input_data_tasks, 
    preprocessing_tasks, 
    model_tasks, 
    valuation_tasks, 
    tables_figures_tasks,
)

def build_landslide_mitigation_task_tree(p):
    # ---------------------------------------------------------------- #
    # INPUT DATA (input_data_tasks.py)
    # ---------------------------------------------------------------- #
    p.input_data_task = p.add_task(input_data_tasks.input_data, creates_dir=True)
    p.build_ease_grid_reference_task = p.add_task(input_data_tasks.build_ease_grid_reference, creates_dir=False)
    p.reproject_dem_task = p.add_task(input_data_tasks.reproject_dem, creates_dir=False)
    p.reproject_gaez_task = p.add_task(input_data_tasks.reproject_gaez, creates_dir=False)
    p.reproject_esacci_forest_share_task = p.add_task(input_data_tasks.reproject_esacci_forest_share, creates_dir=False)
    p.reproject_uglc_events_task = p.add_task(input_data_tasks.reproject_uglc_events, creates_dir=False)
    p.reproject_landscan_task = p.add_task(input_data_tasks.reproject_landscan_population, creates_dir=False)
    p.reproject_soilgrids_properties_task = p.add_task(input_data_tasks.reproject_soilgrids_properties, creates_dir=False)
    p.reproject_worldclim_bio12_task = p.add_task(input_data_tasks.reproject_worldclim_bio12, creates_dir=False)
    p.reproject_hihydrosoil_ksat_task = p.add_task(input_data_tasks.reproject_hihydrosoil_ksat, creates_dir=False)
    p.reproject_soil_depth_task = p.add_task(input_data_tasks.reproject_soil_depth, creates_dir=False)
    p.reproject_grip_roads_task = p.add_task(input_data_tasks.reproject_grip_roads, creates_dir=False)
    p.reproject_rain_daily_task = p.add_task(input_data_tasks.reproject_rain_daily, creates_dir=False)
    p.validate_input_rasters_task = p.add_task(input_data_tasks.validate_input_rasters, creates_dir=False)

    # ---------------------------------------------------------------- #
    # PREPROCESSING (preprocessing_tasks.py)
    # ---------------------------------------------------------------- #
    p.preprocessing_task = p.add_task(preprocessing_tasks.preprocessing, creates_dir=True)
    p.build_uglc_annual_panels_task = p.add_task(preprocessing_tasks.build_uglc_annual_panels,creates_dir=False)
    p.fill_pits_task = p.add_task(preprocessing_tasks.fill_pits, creates_dir=False)
    p.compute_flow_dir_d8_task = p.add_task(preprocessing_tasks.compute_flow_dir_d8, creates_dir=False)
    p.compute_upslope_area_task = p.add_task(preprocessing_tasks.compute_upslope_area, creates_dir=False)
    p.compute_slope_task = p.add_task(preprocessing_tasks.compute_slope, creates_dir=False)
    p.compute_soil_hydraulic_properties_task = p.add_task(preprocessing_tasks.compute_soil_hydraulic_properties, creates_dir=False)
    p.compute_static_q_task = p.add_task(preprocessing_tasks.compute_static_q, creates_dir=False)
    p.compute_si_scenarios_task = p.add_task(preprocessing_tasks.compute_si_scenarios, creates_dir=False)
    p.build_estimation_table_task = p.add_task(preprocessing_tasks.build_estimation_table, creates_dir=False)

    # ---------------------------------------------------------------- #
    # MODELING (model_tasks.py)
    # ---------------------------------------------------------------- #
    p.modeling_task = p.add_task(model_tasks.modeling, creates_dir=True)
    p.calibrate_si_to_probability_task = p.add_task(model_tasks.calibrate_si_to_probability, creates_dir=False)
    p.estimate_severity_model_task = p.add_task(model_tasks.estimate_severity_model, creates_dir=False)
    p.estimate_severity_model_si_sensitivity_task = p.add_task(model_tasks.estimate_severity_model_si_sensitivity, creates_dir=False)

    # ---------------------------------------------------------------- #
    # VALUATION / PREDICTION (valuation_tasks.py)
    # ---------------------------------------------------------------- #
    # A. Tile-level
    p.generate_tile_zones_task = p.add_iterator(valuation_tasks.tile_zones, run_in_parallel=p.run_in_parallel)
    p.predict_landslides_scenarios_task = p.add_task(valuation_tasks.predict_landslides_scenarios, parent=p.generate_tile_zones_task)
    p.predict_mortality_scenarios_task = p.add_task(valuation_tasks.predict_mortality_scenarios, parent=p.generate_tile_zones_task)
    p.stitch_tiles_task = p.add_task(valuation_tasks.stitch_tiles, creates_dir=True)
    # B. Global-level
    p.valuation_task = p.add_task(valuation_tasks.valuation, creates_dir=True)
    p.build_vsl_raster_task = p.add_task(valuation_tasks.build_vsl_raster, creates_dir=False)
    p.compute_avoided_mortality_task = p.add_task(valuation_tasks.compute_avoided_mortality, creates_dir=False)

    # ---------------------------------------------------------------- #
    # Tables & Figures (tables_figures_tasks.py)
    # ---------------------------------------------------------------- #
    p.tables_figures_task = p.add_task(tables_figures_tasks.tables_figures, creates_dir=True)
    p.compute_zonal_statistics_task = p.add_task(tables_figures_tasks.compute_zonal_statistics, creates_dir=False)
    p.export_regression_tables_task = p.add_task(tables_figures_tasks.export_regression_tables, creates_dir=False)
    p.plot_global_rasters_png_task = p.add_task(tables_figures_tasks.plot_global_rasters_png, creates_dir=False)
    p.plot_country_choropleth_maps_task = p.add_task(tables_figures_tasks.plot_country_choropleth_maps, creates_dir=False)
    p.plot_uglc_fatality_bins_task = p.add_task(tables_figures_tasks.plot_uglc_from_vector, creates_dir=False)
    p.export_results_tables_task = p.add_task(tables_figures_tasks.export_results_tables, creates_dir=False)
    p.export_si_severity_sensitivity_table_task = p.add_task(tables_figures_tasks.export_si_severity_sensitivity_table, creates_dir=False)
    p.export_pi_audit_table_task = p.add_task(tables_figures_tasks.export_pi_audit_table, creates_dir=False)

    return p


if __name__ == '__main__':
    hb.log('Starting landslide mitigation workflow...')
    p = hb.ProjectFlow()
    p.force_run = False
    p.L = hb.get_logger('landslide_mitigation_workflow')

    # ---- DIR configuration ----
    # Raw, untouched source data for THIS project (GDrive-synced). NOT the
    # same as p.base_data_dir below -- that name is reserved by hazelbean's
    # get_path() convention for the local base_data cache (it scans here,
    # and falls back to the cloud bucket, for anything requested via a Ref
    # Path). UGLC events (vector) are the only raw source still read from
    # here directly; every raster input now comes from base_data via
    # get_path(), either from this project's own submission
    # (submissions/landslide_mitigation/...) or another project's.
    p.raw_input_data_dir = (
        '/Users/mbraaksma/Library/CloudStorage/GoogleDrive-braak014@umn.edu/'
        'Shared drives/NatCapTEEMs/Projects/Global GEP/Ecosystem Services '
        'SubFolders/Landslides/global_results/input_data_raw'
    )
    p.user_dir = os.path.expanduser('~')
    p.base_data_dir = os.path.join(p.user_dir, 'Files', 'base_data')

    p.project_name = 'global_results_si'
    p.project_dir = os.path.join(p.user_dir, 'Files', 'landslide_mitigation', p.project_name)
    p.set_project_dir(p.project_dir)

    # ---- Processing parameters ----
    p.processing_resolution = 2000
    p.run_in_parallel = True
    p.num_workers = 8

    # ---- Model parameters ----
    p.data_processing_range = range(2007, 2020)
    p.modeling_range = range(2007, 2019)
    p.prediction_years = [2019]
    p.max_location_accuracy_m = 1000
    p.control_ratio = 25
    p.c_root_scenarios = {
        'observed': 'observed',
        'full_impacts': 0,
    }

    build_landslide_mitigation_task_tree(p)
    p.execute()