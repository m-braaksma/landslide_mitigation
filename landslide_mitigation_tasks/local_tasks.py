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

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

# Reset GDAL_DATA path after importing hazelbean
gdal_data_path = os.environ.get("GDAL_DATA")
print("GDAL_DATA:", gdal_data_path)
os.environ['GDAL_DATA'] = '/users/0/braak014/miniforge3/envs/teems02/share/gdal'


def damage_function_task(p):
    """Estimate damage function coefficients."""
    if p.run_this:
        # Define expected output files
        json_output = os.path.join(p.cur_dir, 'damage_function_coefficients.json')
        
        # Check if output already exists and skip if not forcing
        if os.path.exists(json_output) and not p.force_run:
            hb.log(f"Output files already exist: {json_output}")
            hb.log("Skipping damage function analysis (use force_run=True to override)")
            return p
        
        hb.log("Starting damage function analysis task...")
        
        try:
            # Run the damage function analysis directly
            coefficients, model, regression_df = run_damage_function_analysis(p.proj_data_dir, p.cur_dir)
            hb.log(f"Successfully completed damage function analysis")
            hb.log(f"Coefficients saved to: {json_output}")
            
            # Store results in project object for potential downstream use
            p.damage_function_coefficients = coefficients
            p.damage_function_model = model
            
        except Exception as e:
            hb.log(f"Failed to run damage function analysis: {e}", level=40)
            raise e
            
    return p

def run_damage_function_analysis(proj_data_dir, output_dir):
    """Main function to run the complete damage function analysis."""
    hb.log("=" * 60)
    hb.log("DAMAGE FUNCTION ANALYSIS WITH PRECIPITATION")
    hb.log("=" * 60)
    
    try:
        # Load and prepare data
        hb.log("1. Loading and preparing data...")
        emdat_gdis = load_and_prepare_data(proj_data_dir)
        
        hb.log("Creating panel dataset with precipitation")
        panel_df = create_panel_data(proj_data_dir, emdat_gdis)
        
        hb.log("Preparing regression variables")
        var_prefixes = ['sed_export', 'worldpop', 'era5_stats']
        regression_df, log_vars = prepare_regression_variables(panel_df, var_prefixes)
        
        hb.log(f"   - Total observations: {len(regression_df)}")
        hb.log(f"   - Observations with mortality > 0: {sum(regression_df['mortality_count'] > 0)}")
        
        # Fit model
        hb.log("2. Fitting damage function model...")
        model = fit_damage_function_model(regression_df, log_vars)
        
        # Extract coefficients
        hb.log("3. Extracting damage function coefficients...")
        coefficients = extract_damage_coefficients(model)
        
        # Save coefficients to local output directory
        hb.log("4. Saving damage function coefficients...")
        json_path, pickle_path, csv_path = save_damage_coefficients_local(coefficients, output_dir)
        
        # Generate regression table
        hb.log("5. Generating regression table...")
        latex_path = generate_regression_table_local(model, output_dir)
        
        hb.log("Damage function analysis completed successfully")
        return coefficients, model, regression_df
        
    except Exception as e:
        hb.log(f"ERROR: Damage function analysis failed: {e}", level=40)
        raise


def load_and_prepare_data(proj_data_dir):
    """Load and prepare the landslide mortality data."""
    # Add utils to path for s3_handler
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    from utils.s3_utils import s3_handler
    
    # Define S3 paths
    s3_paths = {
        "gdis": proj_data_dir + "pend-gdis-1960-2018-disasterlocations/pend-gdis-1960-2018-disasterlocations.gpkg",
        "adm2_map": proj_data_dir + "borders/adm2_map.csv",
        "emdat": proj_data_dir + "emdat/public_emdat_2024-09-09.xlsx"
    }
    
    # Check if files exist
    missing = [(k, v) for k, v in s3_paths.items() if not s3_handler.file_exists(v)]
    if missing:
        for k, v in missing:
            hb.log(f"ERROR: Missing required input: {k} at path: s3://{s3_handler.bucket_name}/{v}", level=40)
        raise FileNotFoundError(f"Missing required input files: {[k for k, v in missing]}")
    
    with s3_handler.temp_workspace("damage_function_data") as workspace_dir:
        workspace_dir = Path(workspace_dir)

        # Download GDIS data
        gdis_local = s3_handler.download_to_temp(s3_paths["gdis"], "gdis.gpkg")
        gdis = gpd.read_file(str(gdis_local), ignore_geometry=True)
        gdis = gdis[gdis['disastertype'] == 'landslide']
        gdis_level2 = gdis[gdis['level'] == '2']
        
        # Download administrative mapping
        adm2_local = s3_handler.download_to_temp(s3_paths["adm2_map"], "adm2_map.csv")
        adm2_map = pd.read_csv(str(adm2_local))
        adm2_map_unique = adm2_map.drop_duplicates(subset=['NAME_1', 'NAME_2'])
        gdis_level2_merged = gdis_level2.merge(
            adm2_map_unique, 
            left_on=['adm1', 'adm2'], 
            right_on=['NAME_1', 'NAME_2'], 
            how='left', 
            validate='many_to_one'
        )

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
        emdat_landslides['disasterno'] = emdat_landslides['DisNo.'].str.extract(r'(\d{4}-\d{4})')
        emdat_landslides_deaths = emdat_landslides[['disasterno', 'Country', 'Subregion', 'Region', 'Start Year', 'Total Deaths']]
        emdat_landslides_deaths = emdat_landslides_deaths.rename(columns={
            'Start Year': 'year',
            'Total Deaths': 'mortality_count'
        })

        # Merge GDIS and EMDAT
        emdat_gdis = gdis_level2_merged.merge(emdat_landslides_deaths, how='inner', on='disasterno')
        
        return emdat_gdis


def create_panel_data(proj_data_dir, emdat_gdis):
    """Create panel dataset with zonal statistics and precipitation data."""
    import pandas as pd
    import pickle
    import sys
    from pathlib import Path
    
    # Add utils to path for s3_handler
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    from utils.s3_utils import s3_handler
    
    # Define S3 path for zonal statistics
    zonal_stats_s3_path = proj_data_dir + "zonal_stats/zonal_stats_adm2.csv"
    
    # Check if file exists
    if not s3_handler.file_exists(zonal_stats_s3_path):
        raise FileNotFoundError(f"Zonal statistics file not found: s3://{s3_handler.bucket_name}/{zonal_stats_s3_path}")
    
    with s3_handler.temp_workspace("panel_data") as workspace_dir:
        workspace_dir = Path(workspace_dir)
        
        # Download zonal statistics
        zonal_local = s3_handler.download_to_temp(zonal_stats_s3_path, "zonal_stats.csv")
        zonal_stats = pd.read_csv(str(zonal_local))
        
        hb.log("Zonal statistics columns:", list(zonal_stats.columns))
        hb.log("Zonal statistics shape:", zonal_stats.shape)
        
        # Drop current year (keep only 2000-2019)
        zonal_stats_panel = zonal_stats[zonal_stats['year'] < 2019]
        print(f"Zonal statistics year range after dropping current year: {zonal_stats_panel['year'].min()} - {zonal_stats_panel['year'].max()}")
        
        # Merge with mortality data
        panel_df = zonal_stats_panel.merge(emdat_gdis, on=['adm_id', 'year'], how='left')
        panel_df['mortality_count'] = panel_df['mortality_count'].fillna(0)
        
        hb.log(f"Final panel data shape: {panel_df.shape}")
        
        return panel_df


def prepare_regression_variables(panel_df, var_prefixes):
    """Prepare variables for regression analysis."""
    hb.log("Preparing regression variables...")
    hb.log("Available columns:", list(panel_df.columns))

    log_vars = []
    for prefix in var_prefixes:
        # Compute range
        range_col = f'{prefix}_range'
        panel_df[range_col] = panel_df[f'{prefix}_max'] - panel_df[f'{prefix}_min']
        
        # Define vars to log-transform
        for stat in ['sum', 'max', 'range']:
            src_col = f'{prefix}_{stat}'
            new_col = f'ln_{prefix}_{stat}'
            
            # Log transform safely
            panel_df[new_col] = panel_df[src_col].apply(lambda x: np.log(x) if x > 0 else np.nan)
            
            log_vars.append(new_col)
    
    hb.log("Log variables to use for regression:", log_vars)
    
    # Drop rows with missing values in these variables
    regression_df = panel_df.dropna(subset=log_vars)
    
    hb.log(f"Observations before cleaning: {len(panel_df)}")
    hb.log(f"Observations after cleaning: {len(regression_df)}")
    
    return regression_df, log_vars

def fit_damage_function_model(regression_df, log_vars):
    """Fit the Poisson regression model for damage function estimation."""

    # Fit the Poisson regression model
    predictors = log_vars
    predictors.append('C(year)')
    formula = f"mortality_count ~ {' + '.join(predictors)}"
    hb.log(f"Regression formula: {formula}")
    model = smf.glm(
        formula=formula, 
        data=regression_df, 
        family=sm.families.Poisson()
    ).fit()
    
    return model


def extract_damage_coefficients(model):
    """Automatically extract all coefficients and diagnostics from the model."""
    params = model.params
    conf_int = model.conf_int()
    pvalues = model.pvalues
    std_errors = model.bse

    # Automatically build coefficient summary
    coefficients = {}
    for param_name in params.index:
        coefficients[param_name] = {
            'coefficient': float(params[param_name]),
            'std_error': float(std_errors[param_name]),
            'p_value': float(pvalues[param_name]),
            'conf_int_lower': float(conf_int.loc[param_name, 0]),
            'conf_int_upper': float(conf_int.loc[param_name, 1]),
        }

    # Add model diagnostics
    coefficients['model_diagnostics'] = {
        'aic': float(model.aic),
        'bic': float(model.bic),
        'log_likelihood': float(model.llf),
        'n_observations': int(model.nobs),
        'df_residuals': int(model.df_resid),
        'df_model': int(model.df_model),
        'deviance': float(model.deviance),
        'pearson_chi2': float(model.pearson_chi2)
    }

    return coefficients



def save_damage_coefficients_local(coefficients, output_dir):
    """Save damage function coefficients to local directory."""

    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(output_dir, 'damage_function_coefficients.json')
    with open(json_path, 'w') as f:
        json.dump(coefficients, f, indent=2)
    
    # Save as pickle for Python use
    pickle_path = os.path.join(output_dir, 'damage_function_coefficients.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(coefficients, f)
    
    # Save as CSV for easy viewing
    csv_data = []
    for var_name, var_data in coefficients.items():
        if var_name == 'year_effects':
            for year_var, year_data in var_data.items():
                csv_data.append({
                    'variable': year_var,
                    'coefficient': year_data['coefficient'],
                    'std_error': year_data['std_error'],
                    'p_value': year_data['p_value'],
                    'conf_int_lower': year_data['conf_int_lower'],
                    'conf_int_upper': year_data['conf_int_upper'],
                })
        elif var_name == 'model_diagnostics':
            continue
        else:
            csv_data.append({
                'variable': var_name,
                'coefficient': var_data['coefficient'],
                'std_error': var_data['std_error'],
                'p_value': var_data['p_value'],
                'conf_int_lower': var_data['conf_int_lower'],
                'conf_int_upper': var_data['conf_int_upper'],
            })
    
    csv_df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, 'damage_function_coefficients.csv')
    csv_df.to_csv(csv_path, index=False)
    
    hb.log(f"Damage function coefficients saved locally:")
    hb.log(f"  JSON: {json_path}")
    hb.log(f"  Pickle: {pickle_path}")
    hb.log(f"  CSV: {csv_path}")
    
    return json_path, pickle_path, csv_path


def generate_regression_table_local(model, output_dir):
    """Generate and save regression table locally."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine regressor order based on what's in the model
    regressor_order = ["ln_sed_export", "ln_pop"]
    if 'ln_precip' in model.params.index:
        regressor_order.append("ln_precip")
    regressor_order.extend(["C(year)", "Intercept"])
    
    # Create summary table
    results_table = summary_col(
        [model],
        stars=True,
        float_format="%.3f",
        model_names=["Mortality Count"],
        regressor_order=regressor_order,
        info_dict={"N": lambda x: f"{int(x.nobs)}"},
    )
    
    # Convert to LaTeX
    latex_output = results_table.as_latex()
    
    # Clean up year fixed effect labels
    def clean_year_fe_labels(text):
        pattern = r"C\(year\)\[T\.(\d{4})\]"
        return re.sub(pattern, r"Year FE (\1)", text)
    
    latex_output = clean_year_fe_labels(latex_output)
    
    # Rename variables for display
    latex_output = (
        latex_output.replace("ln\_sed\_export", "Sediment export (tons; ln)")
                    .replace("ln\_pop", "Population (count; ln)")
                    .replace("ln\_precip", "Precipitation (mm; ln)")
                    .replace("C(year)", "Year FE")
    )
    
    # Save LaTeX table
    latex_path = os.path.join(output_dir, 'regression_table.tex')
    with open(latex_path, "w") as f:
        f.write(latex_output)
    
    hb.log(f"Regression table saved to: {latex_path}")
    hb.log("\nRegression Results Summary:")
    hb.log(str(results_table))
    
    return latex_path