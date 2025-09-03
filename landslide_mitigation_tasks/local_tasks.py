import os
import sys
import hazelbean as hb
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import pickle
import re
import ast

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

from utils.s3_utils import s3_handler

# Reset GDAL_DATA path after importing hazelbean
gdal_data_path = os.environ.get("GDAL_DATA")
print("GDAL_DATA:", gdal_data_path)
os.environ['GDAL_DATA'] = '/users/0/braak014/miniforge3/envs/teems02/share/gdal'


def prepare_panel_data(p):
    """
    Master task that orchestrates the preparation of panel data for landslide analysis.
    Calls three sub-tasks: combine_zonal_stats, process_emdat_data, and merge_final_panel.
    """
    if not p.run_this:
        return p
    
    hb.log("Starting panel data preparation pipeline...")
    
    # Combine zonal statistics
    hb.log("Step 1: Combining zonal statistics...")
    p = combine_zonal_stats_subtask(p)
    
    # Process EMDAT data
    hb.log("Step 2: Processing EMDAT mortality data...")
    p = process_emdat_data_subtask(p)
    
    # Merge final panel
    hb.log("Step 3: Merging final panel dataset...")
    p = merge_final_panel_subtask(p)
    
    hb.log("Panel data preparation completed successfully")
    return p


def combine_zonal_stats_subtask(p):
    """Combine all zonal statistics results into a single DataFrame."""
    
    # Define expected output file 
    output_file = os.path.join(p.cur_dir, 'zonal_stats_combined.csv')
    
    # Check if output already exists and skip if not forcing
    if os.path.exists(output_file) and not getattr(p, 'force_run', False):
        hb.log(f"Output file already exists: {output_file}")
        hb.log("Skipping combine zonal statistics (use force_run=True to override)")
        return p
    
    hb.log("Starting combine zonal statistics subtask...")
    
    try:
        records = []
        hb.log("Starting to combine zonal statistics results...")
        hb.log(f"Project data dir: {p.proj_data_dir}")
        hb.log(f"Output dir: {p.cur_dir}")
        hb.log(f"Years: {p.years}")
        
        # Use s3_handler temp workspace for all downloads
        with s3_handler.temp_workspace("combine_zonal_stats") as temp_dir:
            # Download and load GAUL data for ID mapping
            gaul_s3_path = p.proj_data_dir + "emdat/gaul2014_2015.gpkg"
            hb.log(f"Downloading GAUL file from: {gaul_s3_path}")
            local_gaul = s3_handler.download_to_temp(gaul_s3_path, "gaul.gpkg")
            
            # Load GAUL data and create FID to ID mapping
            hb.log("Loading GAUL data for ID mapping...")
            gaul_gdf = gpd.read_file(local_gaul, layer='level2', ignore_geometry=True)
            gaul_gdf = gaul_gdf.reset_index(drop=True)
            
            fid_to_ids = {
                idx: {
                    'ADM2_CODE': row['ADM2_CODE'],
                    'ADM2_NAME': row['ADM2_NAME']
                }
                for idx, row in gaul_gdf.iterrows()
            }
            
            hb.log(f"Created FID mapping for {len(fid_to_ids)} administrative units")
            
            for year in p.years:
                hb.log(f"Processing year {year}...")
                
                # Define S3 paths
                sed_export_s3_path = p.proj_data_dir + f"zonal_stats/sed_export_{year}_zonal_stats.pkl"
                cf_sed_export_s3_path = p.proj_data_dir + f"zonal_stats/sed_export_cf_{year}_zonal_stats.pkl"
                worldpop_s3_path = p.proj_data_dir + f"zonal_stats/worldpop_{year}_zonal_stats.pkl"
                era5_s3_path = p.proj_data_dir + f"zonal_stats/era5_precip_{year}_zonal_stats.pkl"
                
                # Download and load data from S3
                sed_stats = None
                if s3_handler.file_exists(sed_export_s3_path):
                    try:
                        local_sed_file = s3_handler.download_to_temp(sed_export_s3_path, f"sed_export_{year}_zonal_stats.pkl")
                        with open(local_sed_file, 'rb') as f:
                            sed_stats = pickle.load(f)
                    except Exception as e:
                        hb.log(f"Warning: Could not load {sed_export_s3_path}: {e}", level=30)
                
                cf_sed_stats = None
                if s3_handler.file_exists(cf_sed_export_s3_path):
                    try:
                        local_cf_sed_file = s3_handler.download_to_temp(cf_sed_export_s3_path, f"sed_export_cf_{year}_zonal_stats.pkl")
                        with open(local_cf_sed_file, 'rb') as f:
                            cf_sed_stats = pickle.load(f)
                    except Exception as e:
                        hb.log(f"Warning: Could not load {cf_sed_export_s3_path}: {e}", level=30)
                
                pop_stats = None
                if s3_handler.file_exists(worldpop_s3_path):
                    try:
                        local_pop_file = s3_handler.download_to_temp(worldpop_s3_path, f"worldpop_{year}_zonal_stats.pkl")
                        with open(local_pop_file, 'rb') as f:
                            pop_stats = pickle.load(f)
                    except Exception as e:
                        hb.log(f"Warning: Could not load {worldpop_s3_path}: {e}", level=30)
                
                era5_stats = None
                if s3_handler.file_exists(era5_s3_path):
                    try:
                        local_era5_file = s3_handler.download_to_temp(era5_s3_path, f"era5_precip_{year}_zonal_stats.pkl")
                        with open(local_era5_file, 'rb') as f:
                            era5_stats = pickle.load(f)
                    except Exception as e:
                        hb.log(f"Warning: Could not load {era5_s3_path}: {e}", level=30)
                
                if sed_stats is None and pop_stats is None and cf_sed_stats is None:
                    missing = [name for name, value in [
                        ("sed_stats", sed_stats),
                        ("pop_stats", pop_stats), 
                        ("cf_sed_stats", cf_sed_stats),
                        ("era5_stats", era5_stats)
                    ] if value is None]

                    hb.log(f"Warning: No data found for year {year}. Missing: {', '.join(missing)}", level=30)
                
                # Use union of all adm_ids (FIDs) from all sources
                all_adm_ids = set()
                if sed_stats:
                    all_adm_ids.update(sed_stats.keys())
                if pop_stats:
                    all_adm_ids.update(pop_stats.keys())
                if cf_sed_stats:
                    all_adm_ids.update(cf_sed_stats.keys())
                if era5_stats:
                    all_adm_ids.update(era5_stats.keys())
                
                for fid in all_adm_ids:
                    record = {
                        'fid': fid,
                        'year': year,
                    }
                    
                    # Add GAUL ID columns if FID exists in mapping
                    if fid in fid_to_ids:
                        record.update(fid_to_ids[fid])
                    else:
                        hb.log(f"Warning: FID {fid} not found in GAUL mapping", level=30)
                        # Add empty values for missing FIDs
                        record.update({
                            'ADM2_CODE': None,
                            'ADM2_NAME': None
                        })
                    
                    # Add all sed_export stats
                    if sed_stats:
                        for stat_key, stat_value in sed_stats.get(fid, {}).items():
                            record[f'sed_export_{stat_key}'] = stat_value
                    
                    # Add all worldpop stats
                    if pop_stats:
                        for stat_key, stat_value in pop_stats.get(fid, {}).items():
                            record[f'worldpop_{stat_key}'] = stat_value

                    # Add all cf_sed_export stats
                    if cf_sed_stats:
                        for stat_key, stat_value in cf_sed_stats.get(fid, {}).items():
                            record[f'cf_sed_export_{stat_key}'] = stat_value
                
                    # Add all era5 stats
                    if era5_stats:
                        for stat_key, stat_value in era5_stats.get(fid, {}).items():
                            record[f'era5_stats_{stat_key}'] = stat_value
                    
                    records.append(record)
        
        # Convert to DataFrame
        hb.log(f"Creating DataFrame with {len(records)} records...")
        df = pd.DataFrame.from_records(records)
        
        # Reorder columns to put ID columns first
        id_columns = ['fid', 'year', 'ADM2_CODE', 'ADM2_NAME']
        other_columns = [col for col in df.columns if col not in id_columns]
        df = df[id_columns + other_columns]
        
        # Save to current directory
        df.to_csv(output_file, index=False)
        
        hb.log(f"Combined zonal statistics saved to: {output_file}")
        hb.log(f"DataFrame shape: {df.shape}")
        hb.log(f"Columns: {list(df.columns)}")
        
        # Log some sample data to verify the mapping worked
        if not df.empty:
            hb.log("Sample records:")
            hb.log(df[id_columns].head().to_string())
        
    except Exception as e:
        hb.log(f"Failed to combine zonal statistics: {e}", level=40)
        raise e
        
    return p


def process_emdat_data_subtask(p):
    """Process EMDAT mortality data and prepare landslide mortality dataset."""
    # Define expected output file
    output_file = os.path.join(p.cur_dir, 'emdat_landslide_mortality.csv')
    
    # Check if output already exists and skip if not forcing
    if os.path.exists(output_file) and not getattr(p, 'force_run', False):
        hb.log(f"Output file already exists: {output_file}")
        hb.log("Skipping EMDAT data processing (use force_run=True to override)")
        return p
    
    hb.log("Starting EMDAT data processing subtask...")
    
    try:
        # Define S3 paths
        s3_paths = {
            "emdat": p.proj_data_dir + "emdat/public_emdat_2024-09-09.xlsx"
        }
        
        # Check if files exist
        missing = [(k, v) for k, v in s3_paths.items() if not s3_handler.file_exists(v)]
        if missing:
            for k, v in missing:
                hb.log(f"ERROR: Missing required input: {k} at path: s3://{s3_handler.bucket_name}/{v}", level=40)
            raise FileNotFoundError(f"Missing required input files: {[k for k, v in missing]}")
        
        with s3_handler.temp_workspace("process_emdat_data") as workspace_dir:
            workspace_dir = Path(workspace_dir)

            # Download EMDAT data
            emdat_local = s3_handler.download_to_temp(s3_paths["emdat"], "emdat.xlsx")
            emdat = pd.read_excel(str(emdat_local))
            
            # Subset landslide data
            landslide_condition = (emdat['Disaster Type'] == 'Mass movement (wet)') & (
                (emdat['Disaster Subtype']=='Landslide (wet)') | 
                (emdat['Disaster Subtype']=='Mudslide')
            )
            emdat_landslides = emdat[landslide_condition]
            emdat_landslides = emdat_landslides.dropna(subset=['Total Deaths'])
            # emdat_landslides['disasterno'] = emdat_landslides['DisNo.'].str.extract(r'(\d{4}-\d{4})')
            emdat_landslides_deaths = emdat_landslides[['DisNo.', 'ISO', 'Country', 'Subregion', 'Region', 'Start Year', 'Total Deaths', 'Admin Units']].copy()
            emdat_landslides_deaths = emdat_landslides_deaths.rename(columns={
                'Start Year': 'year',
                'Total Deaths': 'mortality_count',
                'ISO': 'iso3',
                'DisNo.': 'disasterno'
            })
            # emdat_landslides_deaths = emdat_landslides_deaths[emdat_landslides_deaths['year'] < 2019]

            # Function to consistently parse admin units (use same logic everywhere)
            def parse_admin_units(units_str):
                if pd.isna(units_str) or units_str == '':
                    return []
                try:
                    # Try ast.literal_eval first
                    return ast.literal_eval(units_str)
                except Exception:
                    try:
                        # Fallback to json.loads with quote replacement
                        return json.loads(units_str.replace("'", '"'))
                    except Exception:
                        return []

            # Enhanced function to expand rows and extract ADM2 units
            def expand_admin_units(row):
                admin_units_raw = row.get('Admin Units', '')

                # Use consistent parsing
                admin_units = parse_admin_units(admin_units_raw)

                # Filter only adm2 units (those with adm2_code)
                adm2_units = [unit for unit in admin_units if 'adm2_code' in unit]

                # Debug: Track what's happening
                has_admin_data = len(admin_units) > 0
                has_adm2_data = len(adm2_units) > 0

                # If no adm2 units, return info about why
                if len(adm2_units) == 0:
                    return [], has_admin_data, has_adm2_data

                split_mortality = row['mortality_count'] / len(adm2_units)

                expanded = []
                for unit in adm2_units:
                    expanded.append({
                        'disasterno': row['disasterno'],
                        'Country': row['Country'],
                        'Subregion': row['Subregion'],
                        'Region': row['Region'],
                        'year': row['year'],
                        'adm2_code': unit.get('adm2_code'),
                        'adm2_name': unit.get('adm2_name'),
                        'mortality_count': split_mortality,
                    })
                return expanded, has_admin_data, has_adm2_data

            # Apply expansion with debugging
            expanded_rows = []
            debug_stats = {
                'total_rows': 0,
                'rows_with_admin_data': 0,
                'rows_with_adm2_data': 0,
                'rows_expanded': 0,
                'parsing_failures': 0
            }

            for _, row in emdat_landslides_deaths.iterrows():
                debug_stats['total_rows'] += 1
                expanded, has_admin, has_adm2 = expand_admin_units(row)

                if has_admin:
                    debug_stats['rows_with_admin_data'] += 1
                if has_adm2:
                    debug_stats['rows_with_adm2_data'] += 1
                    debug_stats['rows_expanded'] += 1

                expanded_rows.extend(expanded)

            # Create DataFrame
            adm2_landslide_df = pd.DataFrame(expanded_rows)

        # Save processed data
        adm2_landslide_df.to_csv(output_file, index=False)

    except Exception as e:
        hb.log(f"Failed to process EMDAT data: {e}", level=40)
        raise e

    return p


def merge_final_panel_subtask(p):
    """Merge zonal statistics with mortality data to create final panel dataset."""
    
    # Define input files from previous subtasks
    zonal_stats_file = os.path.join(p.cur_dir, 'zonal_stats_combined.csv')
    emdat_file = os.path.join(p.cur_dir, 'emdat_landslide_mortality.csv')
    
    # Define output file
    output_file = os.path.join(p.cur_dir, 'panel_dataset_final.csv')
    
    # Check if inputs exist
    if not os.path.exists(zonal_stats_file):
        raise FileNotFoundError(f"Required input not found: {zonal_stats_file}. Run combine_zonal_stats_subtask first.")
    if not os.path.exists(emdat_file):
        raise FileNotFoundError(f"Required input not found: {emdat_file}. Run process_emdat_data_subtask first.")
    
    # Check if output already exists and skip if not forcing
    if os.path.exists(output_file) and not getattr(p, 'force_run', False):
        hb.log(f"Output file already exists: {output_file}")
        hb.log("Skipping final panel merge (use force_run=True to override)")
        return p
    
    hb.log("Starting final panel dataset merge subtask...")

    try:
        # Load the processed data
        hb.log("Loading zonal statistics data...")
        zonal_stats_panel = pd.read_csv(zonal_stats_file)
        hb.log("Loading EMDAT mortality data...")
        emdat_gaul = pd.read_csv(emdat_file)

        hb.log("Zonal statistics columns:", list(zonal_stats_panel.columns))
        hb.log("Zonal statistics shape:", zonal_stats_panel.shape)
        hb.log(f"Zonal statistics year range: {zonal_stats_panel['year'].min()} - {zonal_stats_panel['year'].max()}")
        hb.log("EMDAT columns:", list(emdat_gaul.columns))
        hb.log("EMDAT shape:", emdat_gaul.shape)

        # Define merge variables and check if they exist in both datasets
        zonal_merge_cols = ['ADM2_CODE', 'year']  # columns in zonal_stats_panel
        emdat_merge_cols = ['adm2_code', 'year']  # corresponding columns in emdat_gaul

        # Check if merge columns exist in zonal statistics data
        missing_zonal_cols = [col for col in zonal_merge_cols if col not in zonal_stats_panel.columns]
        if missing_zonal_cols:
            raise ValueError(f"Missing merge columns in zonal statistics data: {missing_zonal_cols}")

        # Check if merge columns exist in EMDAT data
        missing_emdat_cols = [col for col in emdat_merge_cols if col not in emdat_gaul.columns]
        if missing_emdat_cols:
            raise ValueError(f"Missing merge columns in EMDAT data: {missing_emdat_cols}")

        hb.log("All merge columns found in both datasets")
        hb.log(f"Zonal merge columns: {zonal_merge_cols}")
        hb.log(f"EMDAT merge columns: {emdat_merge_cols}")

        # Perform the merge
        hb.log("Merging EMDAT and zonal statistics data...")
        panel_df = zonal_stats_panel.merge(
            emdat_gaul, 
            left_on=zonal_merge_cols, 
            right_on=emdat_merge_cols, 
            how='left', 
            suffixes=('_zonal', '_emdat')
        )

        hb.log("Merge completed successfully")
        hb.log("Final panel dataset shape:", panel_df.shape)
        hb.log("Final panel dataset columns:", list(panel_df.columns))

    except FileNotFoundError as e:
        hb.log(f"File not found error: {e}")
        raise
    except ValueError as e:
        hb.log(f"Merge column validation error: {e}")
        raise
    except Exception as e:
        hb.log(f"Unexpected error during merge: {e}")
        raise
            
    # Fill missing mortality with 0 (no landslide mortality recorded)
    panel_df['mortality_count'] = panel_df['mortality_count'].fillna(0)
    # Log-transform selected columns
    log_columns = ['sed_export_sum', 'worldpop_sum', 'era5_stats_sum', 'cf_sed_export_sum']
    for col in log_columns:
        if col in panel_df.columns:
            panel_df[f'ln_{col}'] = np.log(panel_df[col].clip(lower=1e-6))
        else:
            hb.log(f"Warning: Column '{col}' not found for log transformation.", level=30)
    
    # Save final panel dataset
    panel_df.to_csv(output_file, index=False)
    
    hb.log(f"Final panel dataset saved to: {output_file}")
    hb.log(f"Final panel data shape: {panel_df.shape}")
    hb.log(f"Years: {panel_df['year'].min()} - {panel_df['year'].max()}")
    hb.log(f"Administrative units: {panel_df['fid'].nunique()}")
    hb.log(f"Observations with mortality > 0: {sum(panel_df['mortality_count'] > 0)}")
    
    # Log summary statistics
    if 'mortality_count' in panel_df.columns:
        mortality_stats = panel_df['mortality_count'].describe()
        hb.log("Mortality count statistics:")
        hb.log(mortality_stats.to_string())
        
    return p



def estimate_damage_function(p):
    """
    Estimate damage function coefficients using the prepared panel dataset.
    
    Estimates: mortality_count ~ ln(sed_sum) + ln(pop_sum) + ln(era5_sum) + year_fixed_effects
    
    This function now focuses purely on the econometric analysis,
    with data preparation handled by the separate prepare_panel_data task.
    """
    if not p.run_this:
        return p
        
    # Define input file from panel data preparation
    panel_data_file = os.path.join(p.prepare_panel_data_dir, 'panel_dataset_final.csv')
    
    # Define expected output files
    coefficients_csv = os.path.join(p.cur_dir, 'damage_function_coefficients.csv')
    regression_table = os.path.join(p.cur_dir, 'regression_table.tex')
    
    # Check if input exists
    if not os.path.exists(panel_data_file):
        raise FileNotFoundError(f"Required panel dataset not found: {panel_data_file}. Run prepare_panel_data task first.")
    
    # Check if output already exists and skip if not forcing
    if os.path.exists(coefficients_csv) and not getattr(p, 'force_run', False):
        hb.log(f"Output files already exist: {coefficients_csv}")
        hb.log("Skipping damage function analysis (use force_run=True to override)")
        return p
    
    hb.log("Starting damage function analysis task...")
    
    try:
        # Load the prepared panel dataset
        hb.log(f"Loading panel dataset from: {panel_data_file}")
        panel_df = pd.read_csv(panel_data_file)
        
        # Run the damage function analysis
        model, regression_df = run_damage_function_analysis(panel_df, p.cur_dir)
        hb.log(f"Successfully completed damage function analysis")
        hb.log(f"Coefficients saved to: {coefficients_csv}")
        hb.log(f"Regression table saved to: {regression_table}")
        
        # Store results in project object for potential downstream use
        p.damage_function_model = model
        
    except Exception as e:
        hb.log(f"Failed to run damage function analysis: {e}", level=40)
        raise e
        
    return p


def run_damage_function_analysis(panel_df, output_dir):
    """Main function to run the complete damage function analysis on prepared data."""
    hb.log("=" * 60)
    hb.log("DAMAGE FUNCTION ANALYSIS")
    hb.log("Model: mortality_count ~ ln(sed_sum) + ln(pop_sum) + ln(era5_sum) + year_FE")
    hb.log("=" * 60)
    
    try:        
        hb.log("1. Preparing regression variables...")
        regression_df = prepare_regression_variables(panel_df)
        
        hb.log(f"   - Total observations: {len(regression_df)}")
        hb.log(f"   - Observations with mortality > 0: {sum(regression_df['mortality_count'] > 0)}")
        
        # Fit model
        hb.log("2. Fitting damage function model...")
        model = fit_damage_function_model(regression_df)
        
        # Save coefficients and table
        hb.log("3. Saving damage function results...")
        save_damage_function_results(model, output_dir)
        
        hb.log("Damage function analysis completed successfully")
        return model, regression_df
        
    except Exception as e:
        hb.log(f"ERROR: Damage function analysis failed: {e}", level=40)
        raise


def prepare_regression_variables(panel_df):
    """
    Prepare variables for regression analysis.
    Creates ln(sed_sum), ln(pop_sum), ln(era5_sum).
    """
    hb.log("Preparing regression variables...")
    hb.log("Available columns:", list(panel_df.columns))

    # Create a copy to avoid modifying original
    regression_df = panel_df.copy()
    
    # Define the specific variables we need
    required_vars = {
        'sed_export_sum': 'ln_sed_sum',
        'worldpop_sum': 'ln_pop_sum', 
        'era5_stats_sum': 'ln_era5_sum'
    }
    
    missing_vars = []
    for src_col, new_col in required_vars.items():
        if src_col in regression_df.columns:
            # Log transform safely (only positive values)
            regression_df[new_col] = regression_df[src_col].apply(
                lambda x: np.log(x) if pd.notnull(x) and x > 0 else np.nan
            )
            hb.log(f"Created {new_col} from {src_col}")
        else:
            missing_vars.append(src_col)
            hb.log(f"Warning: {src_col} not found in data", level=30)
    
    if missing_vars:
        hb.log(f"Missing required variables: {missing_vars}", level=30)
        # Continue with available variables
        available_log_vars = [new_col for src_col, new_col in required_vars.items() 
                             if src_col in panel_df.columns]
    else:
        available_log_vars = list(required_vars.values())
    
    hb.log(f"Log variables for regression: {available_log_vars}")
    
    # Count missing values per variable before dropping
    if available_log_vars:
        missing_counts = regression_df[available_log_vars].isna().sum()
        for var, count in missing_counts.items():
            hb.log(f"Missing values in {var}: {count}")

        # Drop rows with missing values in any log variable
        regression_df = regression_df.dropna(subset=available_log_vars)

    hb.log(f"Observations before cleaning: {len(panel_df)}")
    hb.log(f"Observations after cleaning: {len(regression_df)}")
    
    # Store available variables for model fitting
    regression_df._log_vars = available_log_vars
    
    return regression_df


def fit_damage_function_model(regression_df):
    """Fit the Poisson regression model: mortality_count ~ ln(sed_sum) + ln(pop_sum) + ln(era5_sum) + year_FE"""
    
    # Get available log variables
    log_vars = getattr(regression_df, '_log_vars', [])
    
    if not log_vars:
        raise ValueError("No valid log variables found for regression")
    
    # Build formula with year fixed effects
    predictors = log_vars + ['C(year)']
    formula = f"mortality_count ~ {' + '.join(predictors)}"
    
    hb.log(f"Regression formula: {formula}")
    
    # Fit Poisson GLM
    model = smf.glm(
        formula=formula, 
        data=regression_df, 
        family=sm.families.Poisson()
    ).fit()
    
    # Print model summary
    hb.log("Model Summary:")
    hb.log(str(model.summary()))
    
    return model


# def save_damage_function_results(model, output_dir):
#     """Save damage function coefficients as CSV and regression table as LaTeX."""
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 1. Save coefficients as CSV
#     coefficients_csv = save_coefficients_csv(model, output_dir)
    
#     # 2. Save regression table as LaTeX
#     latex_path = save_regression_table_latex(model, output_dir)
    
#     hb.log(f"Damage function results saved:")
#     hb.log(f"  Coefficients CSV: {coefficients_csv}")
#     hb.log(f"  Regression table LaTeX: {latex_path}")
    
#     return coefficients_csv, latex_path

def save_damage_function_results(model, output_dir):
    """Save damage function coefficients as CSV, regression table as LaTeX, and full model object."""
    import pickle
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save coefficients as CSV
    coefficients_csv = save_coefficients_csv(model, output_dir)
    
    # 2. Save regression table as LaTeX
    latex_path = save_regression_table_latex(model, output_dir)
    
    # 3. Save full model object
    model_pkl_path = os.path.join(output_dir, 'damage_function_model.pkl')
    with open(model_pkl_path, 'wb') as f:
        pickle.dump(model, f)
    
    hb.log(f"Damage function results saved:")
    hb.log(f"  Coefficients CSV: {coefficients_csv}")
    hb.log(f"  Regression table LaTeX: {latex_path}")
    hb.log(f"  Model object: {model_pkl_path}")
    
    return coefficients_csv, latex_path, model_pkl_path



def save_coefficients_csv(model, output_dir):
    """Save model coefficients to CSV format."""
    
    # Extract coefficient information
    params = model.params
    conf_int = model.conf_int()
    pvalues = model.pvalues
    std_errors = model.bse
    
    # Build coefficient data
    coef_data = []
    for param_name in params.index:
        coef_data.append({
            'variable': param_name,
            'coefficient': float(params[param_name]),
            'std_error': float(std_errors[param_name]),
            'p_value': float(pvalues[param_name]),
            'conf_int_lower': float(conf_int.loc[param_name, 0]),
            'conf_int_upper': float(conf_int.loc[param_name, 1]),
            'significant': pvalues[param_name] < 0.05
        })
    
    # Add model diagnostics as separate rows
    diagnostics = {
        'model_aic': float(model.aic),
        'model_bic': float(model.bic), 
        'model_log_likelihood': float(model.llf),
        'model_n_observations': int(model.nobs),
        'model_df_residuals': int(model.df_resid),
        'model_deviance': float(model.deviance)
    }
    
    for diag_name, diag_value in diagnostics.items():
        coef_data.append({
            'variable': diag_name,
            'coefficient': diag_value,
            'std_error': np.nan,
            'p_value': np.nan,
            'conf_int_lower': np.nan,
            'conf_int_upper': np.nan,
            'significant': False
        })
    
    # Save as CSV
    coef_df = pd.DataFrame(coef_data)
    csv_path = os.path.join(output_dir, 'damage_function_coefficients.csv')
    coef_df.to_csv(csv_path, index=False)
    
    hb.log(f"Coefficients saved to: {csv_path}")
    return csv_path


def save_regression_table_latex(model, output_dir):
    """Generate and save regression table as LaTeX."""
    
    # Create clean variable names for display
    var_name_mapping = {
        'ln_sed_sum': 'Sediment export (ln)',
        'ln_pop_sum': 'Population (ln)',
        'ln_era5_sum': 'Precipitation (ln)',
        'Intercept': 'Constant'
    }
    
    # Determine regressor order
    regressor_order = []
    for var in ['ln_sed_sum', 'ln_pop_sum', 'ln_era5_sum']:
        if var in model.params.index:
            regressor_order.append(var)
    regressor_order.extend(['C(year)', 'Intercept'])
    
    # Create summary table using statsmodels
    try:
        results_table = summary_col(
            [model],
            stars=True,
            float_format="%.4f",
            model_names=["Landslide Mortality"],
            regressor_order=regressor_order,
            info_dict={
                "N": lambda x: f"{int(x.nobs)}",
                "AIC": lambda x: f"{x.aic:.2f}",
                "BIC": lambda x: f"{x.bic:.2f}"
            },
        )
        
        # Convert to LaTeX
        latex_output = results_table.as_latex()
        
        # Clean up variable names in LaTeX output
        for old_name, new_name in var_name_mapping.items():
            latex_output = latex_output.replace(old_name.replace('_', r'\_'), new_name)
        
        # Clean up year fixed effect labels
        def clean_year_fe_labels(text):
            pattern = r"C\(year\)\[T\.(\d{4})\]"
            return re.sub(pattern, r"Year FE (\1)", text)
        
        latex_output = clean_year_fe_labels(latex_output)
        
        # Save LaTeX table
        latex_path = os.path.join(output_dir, 'regression_table.tex')
        with open(latex_path, "w") as f:
            f.write(latex_output)
        
        hb.log(f"LaTeX regression table saved to: {latex_path}")
        hb.log("\nRegression Table Preview:")
        hb.log(str(results_table))
        
        return latex_path
        
    except Exception as e:
        hb.log(f"Warning: Could not generate LaTeX table: {e}", level=30)
        
        # Fallback: create simple LaTeX table manually
        latex_path = os.path.join(output_dir, 'regression_table.tex')
        with open(latex_path, "w") as f:
            f.write("% LaTeX table generation failed - see CSV for coefficients\n")
            f.write("% Error: " + str(e) + "\n")
        
        return latex_path


# def extract_damage_coefficients(model):
#     """Automatically extract all coefficients and diagnostics from the model."""
#     params = model.params
#     conf_int = model.conf_int()
#     pvalues = model.pvalues
#     std_errors = model.bse

#     # Automatically build coefficient summary
#     coefficients = {}
#     for param_name in params.index:
#         coefficients[param_name] = {
#             'coefficient': float(params[param_name]),
#             'std_error': float(std_errors[param_name]),
#             'p_value': float(pvalues[param_name]),
#             'conf_int_lower': float(conf_int.loc[param_name, 0]),
#             'conf_int_upper': float(conf_int.loc[param_name, 1]),
#         }

#     # Add model diagnostics
#     coefficients['model_diagnostics'] = {
#         'aic': float(model.aic),
#         'bic': float(model.bic),
#         'log_likelihood': float(model.llf),
#         'n_observations': int(model.nobs),
#         'df_residuals': int(model.df_resid),
#         'df_model': int(model.df_model),
#         'deviance': float(model.deviance),
#         'pearson_chi2': float(model.pearson_chi2)
#     }

#     return coefficients


# def save_damage_coefficients_local(coefficients, output_dir):
#     """Save damage function coefficients to local directory."""

#     os.makedirs(output_dir, exist_ok=True)
    
#     # Save as JSON
#     json_path = os.path.join(output_dir, 'damage_function_coefficients.json')
#     with open(json_path, 'w') as f:
#         json.dump(coefficients, f, indent=2)
    
#     # Save as pickle for Python use
#     pickle_path = os.path.join(output_dir, 'damage_function_coefficients.pkl')
#     with open(pickle_path, 'wb') as f:
#         pickle.dump(coefficients, f)
    
#     # Save as CSV for easy viewing
#     csv_data = []
#     for var_name, var_data in coefficients.items():
#         if var_name == 'year_effects':
#             for year_var, year_data in var_data.items():
#                 csv_data.append({
#                     'variable': year_var,
#                     'coefficient': year_data['coefficient'],
#                     'std_error': year_data['std_error'],
#                     'p_value': year_data['p_value'],
#                     'conf_int_lower': year_data['conf_int_lower'],
#                     'conf_int_upper': year_data['conf_int_upper'],
#                 })
#         elif var_name == 'model_diagnostics':
#             continue
#         else:
#             csv_data.append({
#                 'variable': var_name,
#                 'coefficient': var_data['coefficient'],
#                 'std_error': var_data['std_error'],
#                 'p_value': var_data['p_value'],
#                 'conf_int_lower': var_data['conf_int_lower'],
#                 'conf_int_upper': var_data['conf_int_upper'],
#             })
    
#     csv_df = pd.DataFrame(csv_data)
#     csv_path = os.path.join(output_dir, 'damage_function_coefficients.csv')
#     csv_df.to_csv(csv_path, index=False)
    
#     hb.log(f"Damage function coefficients saved locally:")
#     hb.log(f"  JSON: {json_path}")
#     hb.log(f"  Pickle: {pickle_path}")
#     hb.log(f"  CSV: {csv_path}")
    
#     return json_path, pickle_path, csv_path


def compute_avoided_mortality(p, target_year=2019):
    """
    Compute avoided mortality by comparing predictions using sed_sum vs cf_sed_sum for a specific year.
    
    This task:
    1. Loads the damage function coefficients
    2. Loads the panel data with both sed_sum and cf_sed_sum for the target year
    3. Predicts mortality using both sediment measures
    4. Calculates avoided mortality as the difference
    5. Saves results
    
    Args:
        p: Project object
        target_year: Year to compute avoided mortality for (default: 2019)
    """
    hb.log(f"Computing avoided mortality for year {target_year} using damage function coefficients...")

    # Load model
    model_path = os.path.join(p.estimate_damage_function_dir, 'damage_function_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    hb.log(f"Loaded damage function model from: {model_path}")

    # Load panel data for target year
    panel_data_path = os.path.join(p.prepare_panel_data_dir, 'panel_dataset_final.csv')
    df = pd.read_csv(panel_data_path)
    df = df[df['year'] == target_year].copy()
    if df.empty:
        raise ValueError(f"No data found for year {target_year}")
    hb.log(f"Loaded panel data with {len(df)} observations for year {target_year}")

    # def predict_mortality(df, coeff_dict, rename_map):
    #     # Rename columns to match coefficient names
    #     df_renamed = df.rename(columns=rename_map)
        
    #     # Start with intercept (constant)
    #     intercept = coeff_dict.get('Intercept', 0)
    #     pred = np.full(len(df_renamed), intercept)
        
    #     # Variables to use (excluding intercept)
    #     vars_to_use = ['ln_sed_sum', 'ln_pop_sum', 'ln_era5_sum']
        
    #     for var in vars_to_use:
    #         coef = coeff_dict.get(var, 0)
    #         if var not in df_renamed.columns:
    #             raise ValueError(f"Variable '{var}' not found in DataFrame columns.")
    #         pred += coef * df_renamed[var].values
        
    #     return pred
    # def predict_mortality(df, coeff_dict, rename_map, target_year=2019):
    #     # Rename columns to match coefficient names
    #     df_renamed = df.rename(columns=rename_map)

    #     # Start with intercept (constant)
    #     intercept = coeff_dict.get('Intercept', 0)
    #     pred = np.full(len(df_renamed), intercept)

    #     # Add year fixed effect
    #     year_fe_key = f'C(year)[T.{target_year}]'
    #     year_coef = coeff_dict.get(year_fe_key, 0)
    #     pred += year_coef

    #     # Variables to use (excluding intercept and year FE)
    #     vars_to_use = ['ln_sed_sum', 'ln_pop_sum', 'ln_era5_sum']

    #     for var in vars_to_use:
    #         coef = coeff_dict.get(var, 0)
    #         if var not in df_renamed.columns:
    #             raise ValueError(f"Variable '{var}' not found in DataFrame columns.")
    #         pred += coef * df_renamed[var].values

    #     return pred
    
    # Load panel data
    df = pd.read_csv(panel_data_path)
    df = df[df['year'] == target_year].copy()

    damage_fn_rename_map = {
        'ln_sed_export_sum': 'ln_sed_sum',
        'ln_cf_sed_export_sum': 'ln_sed_sum',
        'ln_worldpop_sum': 'ln_pop_sum',
        'ln_era5_stats_sum': 'ln_era5_sum'
    }

    # Prepare baseline scenario (observed LULC)
    baseline_df = df[['mortality_count', 'ln_sed_export_sum', 'ln_worldpop_sum', 'ln_era5_stats_sum', 'year']].copy()
    baseline_df = baseline_df.rename(columns=damage_fn_rename_map)

    # Prepare counterfactual scenario (lower forest retention)
    counterfactual_df = df[['mortality_count', 'ln_cf_sed_export_sum', 'ln_worldpop_sum', 'ln_era5_stats_sum', 'year']].copy()
    counterfactual_df = counterfactual_df.rename(columns=damage_fn_rename_map)

    # Use model's predict method
    mortality_baseline = model.predict(baseline_df)
    mortality_counterfactual = model.predict(counterfactual_df)
    avoided_mortality = mortality_counterfactual - mortality_baseline
        
    print(f"Mean avoided mortality: {avoided_mortality.mean()}")
    
    # Create results dataframe
    results = df[['ADM2_CODE', 'year']].copy()
    results['mortality_baseline'] = mortality_baseline
    results['mortality_mitigation'] = mortality_counterfactual
    results['avoided_mortality'] = avoided_mortality
    results['sed_export_sum'] = df['sed_export_sum']
    results['cf_sed_export_sum'] = df['cf_sed_export_sum']
    results['sed_reduction'] = df['cf_sed_export_sum'] - df['sed_export_sum']
    
    # Add summary statistics
    results_summary = {
        'target_year': target_year,
        'total_observations': len(results),
        'mean_avoided_mortality': float(np.mean(avoided_mortality)),
        'median_avoided_mortality': float(np.median(avoided_mortality)),
        'std_avoided_mortality': float(np.std(avoided_mortality)),
        'total_predicted_mortality': float(np.sum(results['mortality_baseline'])),
        'total_avoided_mortality': float(np.sum(avoided_mortality)),
        'min_avoided_mortality': float(np.min(avoided_mortality)),
        'max_avoided_mortality': float(np.max(avoided_mortality)),
        'mean_sed_reduction': float(np.mean(results['sed_reduction'])),
        'total_sed_reduction': float(np.sum(results['sed_reduction']))
    }
    
    # Log summary
    hb.log(f"Avoided Mortality Summary for {target_year}:")
    hb.log(f"  Total observations: {results_summary['total_observations']:,}")
    hb.log(f"  Mean avoided mortality per observation: {results_summary['mean_avoided_mortality']:.6f}")
    hb.log(f"  Total predicted mortality: {results_summary['total_predicted_mortality']:.2f}")
    hb.log(f"  Total avoided mortality: {results_summary['total_avoided_mortality']:.2f}")
    hb.log(f"  Total observed mortality: {df[df['year'] == 2019]['mortality_count'].sum():.2f}")
    hb.log(f"  Mean sediment reduction: {results_summary['mean_sed_reduction']:.6f}")
    
    # Save results (local only)
    results_path = os.path.join(p.cur_dir, f'avoided_mortality_results_{target_year}.csv')
    summary_path = os.path.join(p.cur_dir, f'avoided_mortality_summary_{target_year}.json')
    
    results.to_csv(results_path, index=False)
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    hb.log(f"Results saved locally:")
    hb.log(f"  Results: {results_path}")
    hb.log(f"  Summary: {summary_path}")
    
    hb.log(f"Avoided mortality computation for {target_year} completed successfully!")
    
    return results_summary