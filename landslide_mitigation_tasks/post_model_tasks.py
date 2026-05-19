import os
import sys
import time
import json
import subprocess
import shutil

import hazelbean as hb
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from osgeo import gdal, ogr, osr
import pygeoprocessing as pygeo
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick

try:
    import contextily as ctx
except Exception:
    ctx = None


# VSL values by year (million USD)
# https://www.transportation.gov/office-policy/transportation-policy/revised-departmental-guidance-on-valuation-of-a-statistical-life-in-economic-analysis
VSL_BY_YEAR = {
    2024: 13.7e6,
    2023: 13.2e6,
    2022: 12.5e6,
    2021: 11.8e6,
    2020: 11.6e6,
    2019: 10.9e6,
    2018: 10.5e6,
    2017: 10.2e6,
    2016: 9.9e6,
    2015: 9.6e6,
    2014: 9.4e6,
    2013: 9.2e6,
    2012: 9.1e6,
}


def load_world_bank_gdp(gdp_csv_path):
    """
    Load World Bank GDP per capita CSV and reshape to long form.
    
    Handles metadata headers, normalizes to ISO3-year-gdp_usd format.
    Returns fallback-ready DataFrame with nearest-year logic built in.
    
    Parameters
    ----------
    gdp_csv_path : str
        Path to worldbank_gdp_per_capita.csv
    
    Returns
    -------
    gdp_long : pd.DataFrame
        Columns: iso3, year, gdp_per_capita_usd
    """
    import pandas as pd
    
    # Skip World Bank metadata headers (first 4 rows)
    df = pd.read_csv(gdp_csv_path, skiprows=4)
    
    # Rename columns to standardize
    df.rename(columns={'Country Code': 'iso3'}, inplace=True)
    
    # Identify year columns (all that are numeric strings)
    year_cols = [c for c in df.columns if str(c).isdigit()]
    year_cols = sorted([int(c) for c in year_cols])
    
    # Reshape to long form
    id_vars = ['iso3']
    gdp_long = df[id_vars + [str(y) for y in year_cols]].melt(
        id_vars=id_vars,
        var_name='year',
        value_name='gdp_per_capita_usd'
    )
    
    # Convert year to int, drop NaN/empty values
    gdp_long['year'] = gdp_long['year'].astype(int)
    gdp_long = gdp_long.dropna(subset=['gdp_per_capita_usd'])
    gdp_long['gdp_per_capita_usd'] = pd.to_numeric(gdp_long['gdp_per_capita_usd'], errors='coerce')
    gdp_long = gdp_long.dropna(subset=['gdp_per_capita_usd'])
    
    hb.log(f'  Loaded {len(gdp_long)} country-year GDP records from {gdp_csv_path}')
    return gdp_long.reset_index(drop=True)


def load_ee_r264_zones(zones_gpkg_path):
    """
    Load EE R264 correspondence zones and validate required columns.
    
    Parameters
    ----------
    zones_gpkg_path : str
        Path to ee_r264_correspondence.gpkg
    
    Returns
    -------
    gdf_zones : gpd.GeoDataFrame
        With required columns: ee_r264_id, iso3, geometry
    """
    import geopandas as gpd
    
    if not os.path.exists(zones_gpkg_path):
        raise FileNotFoundError(f'Zones vector not found: {zones_gpkg_path}')
    
    gdf = gpd.read_file(zones_gpkg_path)
    
    # Validate required columns
    required_cols = ['ee_r264_id', 'iso3']
    missing = [c for c in required_cols if c not in gdf.columns]
    if missing:
        raise ValueError(f'Missing required columns in {zones_gpkg_path}: {missing}')
    
    hb.log(f'  Loaded {len(gdf)} zones from {zones_gpkg_path}')
    return gdf


def get_ee_r264_path(user_path=None):
    """
    Resolve EE R264 GPKG path with compatibility fallback.
    
    Try user-provided path first, then fallback to legacy location.
    
    Parameters
    ----------
    user_path : str, optional
        If provided, use this path; otherwise search standard locations
    
    Returns
    -------
    resolved_path : str
        Path to GPKG (or raises FileNotFoundError if not found)
    """
    candidates = []
    
    if user_path:
        candidates.append(user_path)
    
    # Standard locations
    candidates.extend([
        os.path.join(os.path.expanduser('~'), 'Files', 'base_data', 'cartographic', 'ee_r264_correspondence.gpkg'),
        os.path.join(os.path.expanduser('~'), 'Files', 'base_data', 'cartographic', 'ee', 'ee_r264_correspondence.gpkg'),
    ])
    
    for path in candidates:
        if os.path.exists(path):
            hb.log(f'  Found EE R264 zones at: {path}')
            return path
    
    raise FileNotFoundError(
        f'EE R264 zones not found in standard locations.\n'
        f'  Checked: {candidates}'
    )


def build_gdp_ratio_raster(gdf_zones, gdp_long, target_year, avoided_mortality_ref_path, 
                           output_dir, fallback_ratio=1.0):
    """
    Build annual GDP ratio raster aligned to avoided_mortality grid.
    
    Joins zones to GDP by iso3 and target year, applies nearest-year fallback
    for missing country-year pairs, then rasterizes to match avoided_mortality
    grid exactly (same CRS, transform, dimensions).
    
    Parameters
    ----------
    gdf_zones : gpd.GeoDataFrame
        Zones with ee_r264_id, iso3, geometry
    gdp_long : pd.DataFrame
        Long-form GDP with iso3, year, gdp_per_capita_usd
    target_year : int
        Year to compute ratio for
    avoided_mortality_ref_path : str
        Path to avoided_mortality_{year}.tif (used as reference grid)
    output_dir : str
        Where to save output ratio raster
    fallback_ratio : float
        Default ratio when country-year missing (typically 1.0)
    
    Returns
    -------
    ratio_raster_path : str
        Path to output gdp_ratio_{year}.tif
    diagnostics : dict
        Summary: matched_zones, unmatched_iso3s, fallback_count, total_area_defaulted_pct
    """
    import numpy as np
    from osgeo import gdal, ogr
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get USA GDP for target year (or nearest available)
    usa_gdp_candidates = gdp_long[(gdp_long['iso3'] == 'USA')]
    if len(usa_gdp_candidates) == 0:
        hb.log(f'ERROR: USA GDP not found in database')
        raise ValueError('USA GDP not found')
    
    usa_gdp_for_year = usa_gdp_candidates[usa_gdp_candidates['year'] == target_year]
    if len(usa_gdp_for_year) == 0:
        # Fallback to nearest year
        year_diff = (usa_gdp_candidates['year'] - target_year).abs()
        nearest_year = usa_gdp_candidates.loc[year_diff.idxmin(), 'year']
        usa_gdp_for_year = usa_gdp_candidates[usa_gdp_candidates['year'] == nearest_year]
        hb.log(f'  USA GDP not available for {target_year}; using {nearest_year}: ${usa_gdp_for_year.iloc[0]["gdp_per_capita_usd"]:,.0f}')
    
    usa_gdp = float(usa_gdp_for_year.iloc[0]['gdp_per_capita_usd'])
    
    # Create zone-gdp join with fallback for missing years
    zone_gdp = []
    unmatched_iso3s = []
    matched_count = 0
    fallback_count = 0
    
    for zone_id, iso3 in zip(gdf_zones['ee_r264_id'], gdf_zones['iso3']):
        # Try exact year first
        gdp_record = gdp_long[(gdp_long['iso3'] == iso3) & (gdp_long['year'] == target_year)]
        
        if len(gdp_record) > 0:
            gdp_val = float(gdp_record.iloc[0]['gdp_per_capita_usd'])
            ratio = gdp_val / usa_gdp
            matched_count += 1
        else:
            # Fallback to nearest year for this country
            country_gdp = gdp_long[gdp_long['iso3'] == iso3]
            if len(country_gdp) > 0:
                year_diff = (country_gdp['year'] - target_year).abs()
                nearest_row = country_gdp.loc[year_diff.idxmin()]
                gdp_val = float(nearest_row['gdp_per_capita_usd'])
                ratio = gdp_val / usa_gdp
                fallback_count += 1
            else:
                # No GDP data for this country at all
                ratio = fallback_ratio
                unmatched_iso3s.append(iso3)
        
        zone_gdp.append({'ee_r264_id': zone_id, 'iso3': iso3, 'ratio': ratio})
    
    df_zone_gdp = pd.DataFrame(zone_gdp)
    gdf_zones_gdp = gdf_zones.merge(df_zone_gdp, on='ee_r264_id', suffixes=('', '_gdp'))
    
    # Read reference raster to get exact grid parameters
    with rasterio.open(avoided_mortality_ref_path) as src:
        ref_profile = src.profile
        ref_transform = src.transform
        ref_width = src.width
        ref_height = src.height
        ref_crs = src.crs
    
    # Rasterize using rasterio.features (no GDAL driver dependency)
    from rasterio.features import rasterize
    
    ratio_raster_path = os.path.join(output_dir, f'gdp_ratio_{target_year}.tif')
    
    hb.log(f'  Rasterizing GDP ratio for {target_year}...')
    
    # Prepare shapes: (geometry, value) tuples where value is encoded ratio
    shapes = []
    for idx, row in gdf_zones_gdp.iterrows():
        # Encode ratio as integer (ratio * 10000) to preserve precision
        ratio_val = int(row['ratio'] * 10000)
        shapes.append((row['geometry'], ratio_val))
    
    # Rasterize to match reference grid exactly
    rasterized = rasterize(
        shapes,
        out_shape=(ref_height, ref_width),
        transform=ref_transform,
        fill=0,
        dtype=rasterio.uint32
    )
    
    # Scale back from integers to actual ratios
    ratio_data = np.where(rasterized > 0, rasterized / 10000.0, 1.0).astype(np.float32)
    
    # Update profile for output
    output_profile = ref_profile.copy()
    output_profile.update(dtype=rasterio.float32, compress='lzw', nodata=0)
    
    # Write output raster
    with rasterio.open(ratio_raster_path, 'w', **output_profile) as dst:
        dst.write(ratio_data, 1)
    
    hb.log(f'  ✓ Created: {ratio_raster_path}')
    
    # Diagnostics
    total_areas_defaulted = sum(1 for r in zone_gdp if r['ratio'] == fallback_ratio)
    diagnostics = {
        'year': target_year,
        'matched_zones': matched_count,
        'fallback_zones': fallback_count,
        'unmatched_zones': len(unmatched_iso3s),
        'unmatched_iso3s': unmatched_iso3s,
        'total_zones': len(zone_gdp),
        'usa_gdp_usd': usa_gdp,
    }
    
    # Write diagnostics
    diag_path = ratio_raster_path.replace('.tif', '_diagnostics.json')
    with open(diag_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    hb.log(f'  Diagnostics: {matched_count} matched, {fallback_count} fallback, {len(unmatched_iso3s)} unmatched')
    
    return ratio_raster_path, diagnostics


def compute_avoided_mortality(p):
    """
    Compute avoided/lost mortality and economic value rasters.

    For each year:
    1. Load stitched hazard rasters and fitted mortality raster
     2. Compute scenario avoided mortality:
         (p_scenario_deforestation - p_no_deforestation) * fitted_mortality
     3. Compute lost mortality relative to no-deforestation:
         (p_observed_deforestation - p_no_deforestation) * fitted_mortality
    4. Convert each mortality raster to value raster with GDP-adjusted VSL

    Outputs
    -------
    avoided_mortality_{scenario}_{year}.tif
    avoided_mortality_value_{scenario}_{year}.tif
    lost_mortality_{year}.tif
    lost_value_{year}.tif

    Compatibility outputs (first configured scenario):
    avoided_mortality_{year}.tif
    avoided_mortality_value_{year}.tif
    """
    if not p.run_this:
        return p

    hb.log('=' * 60)
    hb.log('PHASE 1: Loading GDP and zone data...')
    hb.log('=' * 60)

    has_gdp_adjustment = False
    gdp_df = None
    gdf_zones = None
    gdp_cache_dir = os.path.join(p.cur_dir, '_gdp_ratio_cache')
    os.makedirs(gdp_cache_dir, exist_ok=True)

    gdp_candidates = [
        os.path.join(p.base_data_dir, 'socioeconomic', 'worldbank_gdp_per_capita.csv'),
        os.path.join(os.path.expanduser('~'), 'Files', 'base_data', 'socioeconomic', 'worldbank_gdp_per_capita.csv'),
    ]

    for gdp_path in gdp_candidates:
        if os.path.exists(gdp_path):
            try:
                gdp_df = load_world_bank_gdp(gdp_path)
                has_gdp_adjustment = True
                hb.log(f'✓ Loaded GDP data: {len(gdp_df)} country-year records')
                break
            except Exception as e:
                hb.log(f'Warning: Could not load GDP data from {gdp_path}: {e}')

    if has_gdp_adjustment:
        try:
            zones_path = get_ee_r264_path()
            gdf_zones = load_ee_r264_zones(zones_path)
            hb.log(f'✓ Loaded zones: {len(gdf_zones)} zones')
        except Exception as e:
            hb.log(f'Warning: Could not load zone vector: {e}')
            hb.log('  Falling back to constant USA VSL')
            has_gdp_adjustment = False
            gdf_zones = None

    if not has_gdp_adjustment:
        hb.log('WARNING: GDP adjustment data not fully available.')
        hb.log('  Using constant USA VSL for all regions.')

    scenario_specs = getattr(p, 'forest_value_scenarios', [])
    if not scenario_specs:
        hb.log('WARNING: p.forest_value_scenarios is empty; only lost_mortality/lost_value will be generated.')

    for year in p.prediction_years:
        hb.log('\n' + '=' * 60)
        hb.log(f'YEAR {year}')
        hb.log('=' * 60)

        expected_output_paths = []
        for scenario_name, _, _ in scenario_specs:
            expected_output_paths.append(os.path.join(p.cur_dir, f'avoided_mortality_{scenario_name}_{year}.tif'))
            expected_output_paths.append(os.path.join(p.cur_dir, f'avoided_mortality_value_{scenario_name}_{year}.tif'))
        expected_output_paths.extend([
            os.path.join(p.cur_dir, f'lost_mortality_{year}.tif'),
            os.path.join(p.cur_dir, f'lost_value_{year}.tif'),
            os.path.join(p.cur_dir, f'value_summary_{year}.json'),
        ])

        if (not getattr(p, 'force_run', False)) and expected_output_paths and all(os.path.exists(path) for path in expected_output_paths):
            hb.log(f'Skipping year {year}: all avoided mortality outputs already exist.')
            continue

        if year not in VSL_BY_YEAR:
            hb.log(f'ERROR: VSL not defined for year {year}')
            raise ValueError(f'VSL not defined for year {year}')
        vsl_usa = VSL_BY_YEAR[year]
        hb.log(f'Using VSL (USA): ${vsl_usa/1e6:.1f}M for year {year}')

        p_nodef_path = os.path.join(p.stitch_tiles_dir, f'no_deforestation_landslide_prob_{year}.tif')
        p_obs_path = os.path.join(p.stitch_tiles_dir, f'observed_landslide_prob_{year}.tif')
        fitted_mort_path = os.path.join(p.stitch_tiles_dir, f'expected_mortality_{year}.tif')

        required_inputs = [p_nodef_path, p_obs_path, fitted_mort_path]
        missing = [path for path in required_inputs if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(f'Missing required stitched rasters for {year}: {missing}')

        with rasterio.open(p_nodef_path) as src:
            p_nodef = src.read(1).astype(np.float32)
            profile = src.profile
            nodata = src.nodata

        with rasterio.open(p_obs_path) as src:
            p_obs = src.read(1).astype(np.float32)

        with rasterio.open(fitted_mort_path) as src:
            fitted_mort = src.read(1).astype(np.float32)

        if nodata is not None:
            valid_base = (
                (p_nodef != nodata)
                & (p_obs != nodata)
                & (fitted_mort != nodata)
                & np.isfinite(p_nodef)
                & np.isfinite(p_obs)
                & np.isfinite(fitted_mort)
            )
        else:
            valid_base = np.isfinite(p_nodef) & np.isfinite(p_obs) & np.isfinite(fitted_mort)

        ratio_raster_path = os.path.join(gdp_cache_dir, f'gdp_ratio_{year}.tif')
        ratio_raster = None
        gdp_metadata = {}

        if has_gdp_adjustment and gdf_zones is not None:
            if not os.path.exists(ratio_raster_path):
                hb.log('Phase 2: Building GDP ratio raster...')
                try:
                    ratio_raster_path, diagnostics = build_gdp_ratio_raster(
                        gdf_zones, gdp_df, year, p_nodef_path, gdp_cache_dir
                    )
                    gdp_metadata = diagnostics
                except Exception as e:
                    hb.log(f'ERROR building ratio raster: {e}')
                    hb.log('  Falling back to constant VSL')
                    has_gdp_adjustment = False
            else:
                hb.log(f'Using cached GDP ratio raster: {ratio_raster_path}')
                diag_path = ratio_raster_path.replace('.tif', '_diagnostics.json')
                if os.path.exists(diag_path):
                    with open(diag_path, 'r') as f:
                        gdp_metadata = json.load(f)

        if has_gdp_adjustment and os.path.exists(ratio_raster_path):
            with rasterio.open(ratio_raster_path) as src:
                ratio_raster = src.read(1).astype(np.float32)
            hb.log('Phase 3: Using GDP-adjusted VSL (ratio raster)')
        else:
            hb.log('Phase 3: Using constant USA VSL')

        profile.update(dtype=rasterio.float32, compress='lzw')

        first_scenario_avoided_path = None
        first_scenario_value_path = None

        for scenario_name, _, _ in scenario_specs:
            p_scen_path = os.path.join(p.stitch_tiles_dir, f'scenario_{scenario_name}_landslide_prob_{year}.tif')
            if not os.path.exists(p_scen_path):
                hb.log(f'WARNING: Missing scenario hazard raster, skipping {scenario_name}: {p_scen_path}')
                continue

            with rasterio.open(p_scen_path) as src:
                p_scen = src.read(1).astype(np.float32)

            if nodata is not None:
                valid_mask = valid_base & (p_scen != nodata) & np.isfinite(p_scen)
            else:
                valid_mask = valid_base & np.isfinite(p_scen)

            avoided_mort = (p_scen - p_nodef) * fitted_mort
            avoided_mort = np.where(valid_mask, avoided_mort, nodata if nodata is not None else np.nan).astype(np.float32)

            avoided_mort_path = os.path.join(p.cur_dir, f'avoided_mortality_{scenario_name}_{year}.tif')
            with rasterio.open(avoided_mort_path, 'w', **profile) as dst:
                dst.write(avoided_mort, 1)

            if ratio_raster is not None:
                value_raster = avoided_mort * vsl_usa * ratio_raster
            else:
                value_raster = avoided_mort * vsl_usa
            if nodata is not None:
                value_raster = np.where(valid_mask, value_raster, nodata)
            value_raster = value_raster.astype(np.float32)

            value_output_path = os.path.join(p.cur_dir, f'avoided_mortality_value_{scenario_name}_{year}.tif')
            with rasterio.open(value_output_path, 'w', **profile) as dst:
                dst.write(value_raster, 1)

            if first_scenario_avoided_path is None:
                first_scenario_avoided_path = avoided_mort_path
                first_scenario_value_path = value_output_path

            total_mortality = float(np.nansum(np.where(valid_mask, avoided_mort, np.nan)))
            total_value = float(np.nansum(np.where(valid_mask, value_raster, np.nan)))
            hb.log(f'✓ {scenario_name} totals for {year}: mortality={total_mortality:,.0f}, value=${total_value/1e9:,.2f}B')

        lost_valid_mask = valid_base
        lost_mortality = (p_obs - p_nodef) * fitted_mort
        lost_mortality = np.where(lost_valid_mask, lost_mortality, nodata if nodata is not None else np.nan).astype(np.float32)
        lost_mort_path = os.path.join(p.cur_dir, f'lost_mortality_{year}.tif')
        with rasterio.open(lost_mort_path, 'w', **profile) as dst:
            dst.write(lost_mortality, 1)

        if ratio_raster is not None:
            lost_value = lost_mortality * vsl_usa * ratio_raster
        else:
            lost_value = lost_mortality * vsl_usa
        if nodata is not None:
            lost_value = np.where(lost_valid_mask, lost_value, nodata)
        lost_value = lost_value.astype(np.float32)
        lost_value_path = os.path.join(p.cur_dir, f'lost_value_{year}.tif')
        with rasterio.open(lost_value_path, 'w', **profile) as dst:
            dst.write(lost_value, 1)

        lost_total_mort = float(np.nansum(np.where(lost_valid_mask, lost_mortality, np.nan)))
        lost_total_value = float(np.nansum(np.where(lost_valid_mask, lost_value, np.nan)))
        hb.log(f'✓ lost totals for {year}: mortality={lost_total_mort:,.0f}, value=${lost_total_value/1e9:,.2f}B')

        summary_path = os.path.join(p.cur_dir, f'value_summary_{year}.json')
        summary = {
            'year': year,
            'vsl_usa_usd': vsl_usa,
            'gdp_adjusted': has_gdp_adjustment,
            'scenarios': [spec[0] for spec in scenario_specs],
            'formula_avoided': '(p_scenario_deforestation - p_no_deforestation) * fitted_mortality',
            'formula_lost': '(p_observed_deforestation - p_no_deforestation) * fitted_mortality',
            'lost_total_mortality': lost_total_mort,
            'lost_total_value_usd': lost_total_value,
        }
        if gdp_metadata:
            summary['gdp_source'] = 'World Bank GDP per capita'
            summary['gdp_diagnostics'] = gdp_metadata
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        hb.log(f'✓ Saved summary: {summary_path}')

    hb.log('\n' + '=' * 60)
    hb.log('COMPUTE AVOIDED MORTALITY COMPLETE')
    hb.log('=' * 60)
    return p


def compute_value(p):
    """Backward-compatible wrapper for legacy task name."""
    return compute_avoided_mortality(p)


def _get_avoided_output_dir(p):
    """Resolve directory containing avoided_mortality/value rasters.
    
    Tries in order:
    1. p.compute_avoided_mortality_dir (explicit attribute)
    2. p.compute_value_dir (legacy name)
    3. {base_dir}/global_results/intermediate/compute_avoided_mortality/ (standard location)
    4. p.cur_dir (fallback)
    """
    # Try explicit attributes first
    if hasattr(p, 'compute_avoided_mortality_dir'):
        return p.compute_avoided_mortality_dir
    if hasattr(p, 'compute_value_dir'):
        return p.compute_value_dir
    
    # Try standard location in project structure
    base_dir = getattr(p, 'base_dir', None) or getattr(p, 'project_dir', None) or os.getcwd()
    standard_path = os.path.join(os.path.dirname(base_dir), 'global_results', 'intermediate', 'compute_avoided_mortality')
    if os.path.exists(standard_path):
        return standard_path
    
    # Final fallback
    return p.cur_dir


def _compute_zonal_stats_for_pair(p, year, scenario_name, avoided_mort_path, value_path,
                                  zones_vector_path, zones_id_column, zones_label_column,
                                  gdf_zones, zone_lookup, max_zone_id, gpkg_path, csv_path):
    """Compute zonal stats for one avoided-mortality/value raster pair."""
    hb.log(f'  Scenario {scenario_name}: {os.path.basename(avoided_mort_path)}')

    with rasterio.open(avoided_mort_path) as src:
        mort_nodata = src.nodata
        mort_height = src.height
        mort_width = src.width
        mort_transform = src.transform
        mort_crs = src.crs

    with rasterio.open(value_path) as src:
        value_nodata = src.nodata

    zones_raster_path = os.path.join(p.cur_dir, '_zones_id.tif')
    zones_raster_needs_update = True
    if os.path.exists(zones_raster_path):
        try:
            with rasterio.open(zones_raster_path) as src:
                zones_raster_needs_update = (
                    src.width != mort_width or
                    src.height != mort_height or
                    src.transform != mort_transform or
                    src.crs != mort_crs
                )
        except Exception:
            zones_raster_needs_update = True

    if zones_raster_needs_update:
        hb.log(f'  [1/3] Rasterizing zones...')
        cmd = [
            'gdal_rasterize',
            '-a', zones_id_column,
            '-of', 'GTiff',
            '-ot', 'UInt16',
            '-co', 'COMPRESS=LZW',
            '-te',
            f'{mort_transform.c}',
            f'{mort_transform.f + mort_height * mort_transform.e}',
            f'{mort_transform.c + mort_width * mort_transform.a}',
            f'{mort_transform.f}',
            '-ts', str(mort_width), str(mort_height),
            zones_vector_path,
            zones_raster_path
        ]
        gdal_rasterize_path = shutil.which('gdal_rasterize')
        if gdal_rasterize_path:
            hb.log(f'  Running: {gdal_rasterize_path}')
            subprocess.run(cmd, check=True)
        else:
            hb.log('  gdal_rasterize not found on PATH; using GDAL Python API fallback')
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(
                zones_raster_path,
                mort_width,
                mort_height,
                1,
                gdal.GDT_UInt16,
                options=['COMPRESS=LZW'],
            )
            dst_ds.SetGeoTransform((mort_transform.c, mort_transform.a, mort_transform.b, mort_transform.f, mort_transform.d, mort_transform.e))
            srs = osr.SpatialReference()
            if mort_crs is not None:
                srs.ImportFromWkt(mort_crs.to_wkt())
            else:
                srs.ImportFromEPSG(4326)
            dst_ds.SetProjection(srs.ExportToWkt())
            band = dst_ds.GetRasterBand(1)
            band.Fill(0)
            band.SetNoDataValue(0)

            vector_ds = ogr.Open(zones_vector_path)
            if vector_ds is None:
                raise FileNotFoundError(f'Could not open zones vector: {zones_vector_path}')
            layer = vector_ds.GetLayer()
            gdal.RasterizeLayer(dst_ds, [1], layer, options=[f'ATTRIBUTE={zones_id_column}'])
            band.FlushCache()
            dst_ds.FlushCache()
            dst_ds = None
            vector_ds = None
        hb.log(f'    ✓ Created: {os.path.basename(zones_raster_path)}')
    else:
        hb.log(f'  [1/3] Using existing zones raster')

    hb.log(f'  [2/3] Computing zonal statistics (block processing)...')
    stats_accum = {}
    for zone_id in range(1, max_zone_id):
        stats_accum[zone_id] = {
            'mort_sum': 0.0,
            'mort_sum_sq': 0.0,
            'mort_count': 0,
            'mort_min': np.inf,
            'mort_max': -np.inf,
            'mort_values': [],
            'value_sum': 0.0,
            'value_sum_sq': 0.0,
            'value_count': 0,
            'value_min': np.inf,
            'value_max': -np.inf,
            'value_values': []
        }

    total_blocks_processed = 0
    zones_with_data = set()
    block_size = 2000

    with rasterio.open(avoided_mort_path) as src:
        raster_height = src.height
        raster_width = src.width

    for y_start in tqdm(range(0, raster_height, block_size), desc=f'  Processing blocks', ncols=80):
        y_end = min(y_start + block_size, raster_height)
        for x_start in range(0, raster_width, block_size):
            x_end = min(x_start + block_size, raster_width)
            total_blocks_processed += 1

            with rasterio.open(avoided_mort_path) as src:
                mort_block = src.read(1, window=Window(x_start, y_start, x_end - x_start, y_end - y_start))
            with rasterio.open(value_path) as src:
                value_block = src.read(1, window=Window(x_start, y_start, x_end - x_start, y_end - y_start))
            with rasterio.open(zones_raster_path) as src:
                zones_block = src.read(1, window=Window(x_start, y_start, x_end - x_start, y_end - y_start))

            unique_zones = np.unique(zones_block)
            for zone_id in unique_zones:
                if zone_id == 0 or zone_id >= max_zone_id:
                    continue

                zone_mask = zones_block == zone_id
                mort_zone_values = mort_block[zone_mask]
                value_zone_values = value_block[zone_mask]

                mort_valid_mask = (mort_zone_values != mort_nodata) & np.isfinite(mort_zone_values) if mort_nodata is not None else np.isfinite(mort_zone_values)
                mort_valid_values = mort_zone_values[mort_valid_mask]
                if len(mort_valid_values) > 0:
                    zones_with_data.add(int(zone_id))
                    stats_accum[zone_id]['mort_sum'] += float(np.sum(mort_valid_values))
                    stats_accum[zone_id]['mort_sum_sq'] += float(np.sum(mort_valid_values ** 2))
                    stats_accum[zone_id]['mort_count'] += int(len(mort_valid_values))
                    stats_accum[zone_id]['mort_min'] = min(stats_accum[zone_id]['mort_min'], float(np.min(mort_valid_values)))
                    stats_accum[zone_id]['mort_max'] = max(stats_accum[zone_id]['mort_max'], float(np.max(mort_valid_values)))
                    if len(mort_valid_values) <= 10000:
                        stats_accum[zone_id]['mort_values'].extend(mort_valid_values.tolist())

                value_valid_mask = (value_zone_values != value_nodata) & np.isfinite(value_zone_values) if value_nodata is not None else np.isfinite(value_zone_values)
                value_valid_values = value_zone_values[value_valid_mask]
                if len(value_valid_values) > 0:
                    stats_accum[zone_id]['value_sum'] += float(np.sum(value_valid_values))
                    stats_accum[zone_id]['value_sum_sq'] += float(np.sum(value_valid_values ** 2))
                    stats_accum[zone_id]['value_count'] += int(len(value_valid_values))
                    stats_accum[zone_id]['value_min'] = min(stats_accum[zone_id]['value_min'], float(np.min(value_valid_values)))
                    stats_accum[zone_id]['value_max'] = max(stats_accum[zone_id]['value_max'], float(np.max(value_valid_values)))
                    if len(value_valid_values) <= 10000:
                        stats_accum[zone_id]['value_values'].extend(value_valid_values.tolist())

    hb.log(f'  [3/3] Finalizing statistics...')
    hb.log(f'  Block processing complete: {total_blocks_processed} blocks, {len(zones_with_data)} zones with data')

    stats_list = []
    for zone_id in range(1, max_zone_id):
        acc = stats_accum[zone_id]
        if acc['mort_count'] > 0 or acc['value_count'] > 0:
            row = {
                zones_id_column: int(zone_id),
                zones_label_column: zone_lookup.get(zone_id, f'Zone_{zone_id}'),
                'scenario': scenario_name,
            }
            if acc['mort_count'] > 0:
                mort_mean = acc['mort_sum'] / acc['mort_count']
                mort_var = (acc['mort_sum_sq'] / acc['mort_count']) - (mort_mean ** 2)
                row.update({
                    'avoided_mortality_sum': float(acc['mort_sum']),
                    'avoided_mortality_mean': float(mort_mean),
                    'avoided_mortality_median': float(np.median(acc['mort_values'])) if acc['mort_values'] else float(mort_mean),
                    'avoided_mortality_std': float(np.sqrt(max(0, mort_var))),
                    'avoided_mortality_min': float(acc['mort_min']),
                    'avoided_mortality_max': float(acc['mort_max']),
                    'avoided_mortality_count': int(acc['mort_count']),
                })
            if acc['value_count'] > 0:
                value_mean = acc['value_sum'] / acc['value_count']
                value_var = (acc['value_sum_sq'] / acc['value_count']) - (value_mean ** 2)
                row.update({
                    'value_sum': float(acc['value_sum']),
                    'value_mean': float(value_mean),
                    'value_median': float(np.median(acc['value_values'])) if acc['value_values'] else float(value_mean),
                    'value_std': float(np.sqrt(max(0, value_var))),
                    'value_min': float(acc['value_min']),
                    'value_max': float(acc['value_max']),
                    'value_count': int(acc['value_count']),
                })
            stats_list.append(row)

    df_stats = pd.DataFrame(stats_list)
    if len(df_stats) == 0:
        hb.log(f'WARNING: No zones with data found for {year} / {scenario_name}!')
        return False

    gdf_stats = gdf_zones.merge(df_stats, on=[zones_id_column, zones_label_column], how='left')
    gdf_stats.to_file(gpkg_path, driver='GPKG', index=False)
    hb.log(f'✓ Saved GPKG: {gpkg_path}')

    df_stats = pd.DataFrame(gdf_stats.drop(columns='geometry'))
    df_stats.to_csv(csv_path, index=False)
    hb.log(f'✓ Saved CSV: {csv_path}')

    valid_zones = df_stats[(df_stats['avoided_mortality_count'] > 0) | (df_stats['value_count'] > 0)]
    hb.log(f'Summary for {year} / {scenario_name}: {len(valid_zones)} zones with data')
    return True


def compute_zonal_statistics(p, zones_vector_path=None, zones_id_column='ee_r264_id', 
                             zones_label_column='ee_r264_label'):
    """
    Compute zonal statistics for avoided mortality by aggregating to regions.
    
    Processes both avoided_mortality and avoided_mortality_value rasters,
    computing mean, median, min, max, sum by zone using memory-efficient block processing.
    
    IMPORTANT: Pixel-level mortality predictions are ALREADY population-representative
    due to IPW adjustment in Stage 2 model. Do NOT apply intercept_correction at pixel
    level. The intercept_correction value stored in sampling_meta.csv is provided for
    reference and is used only if needed for specific aggregate-level adjustments, but
    for standard zonal summation, predictions are used as-is.
    
    Parameters
    ----------
    p : ProjectFlow
        Project object with stitch_tiles_dir, time_range attributes
    zones_vector_path : str, optional
        Path to zones vector file (GeoPackage or shapefile)
        If None, looks for ../base_data/cartographic/ee/ee_r264_correspondence.gpkg
    zones_id_column : str
        Name of column with zone IDs
    zones_label_column : str
        Name of column with zone labels/names
    
    Outputs
    -------
    {year}_zonal_stats_avoided_mortality.csv
    {year}_zonal_stats_value.gpkg (with geometry)
    """
    if not p.run_this:
        return p
    
    scenario_specs = getattr(p, 'forest_value_scenarios', [])
    if not scenario_specs:
        hb.log('WARNING: p.forest_value_scenarios is empty; zonal stats will not run for scenario outputs.')
        return p

    output_files = []
    for year in p.prediction_years:
        for scenario_name, _, _ in scenario_specs:
            gpkg_path = os.path.join(p.cur_dir, f'{year}_{scenario_name}_zonal_stats.gpkg')
            csv_path = os.path.join(p.cur_dir, f'{year}_{scenario_name}_zonal_stats.csv')
            output_files.extend([gpkg_path, csv_path])
    if all(os.path.exists(path) for path in output_files):
        hb.log(f'✓ Zonal statistics already exist for years={p.prediction_years}, scenarios={[spec[0] for spec in scenario_specs]}')
        return p
    else:
        hb.log(f'Computing zonal statistics for years: {p.prediction_years}, scenarios: {[spec[0] for spec in scenario_specs]}...')
    
    if zones_vector_path is None:
        try:
            zones_vector_path = get_ee_r264_path()
        except FileNotFoundError:
            hb.log(f'ERROR: Zones vector not found at standard locations')
            return p
    if not os.path.exists(zones_vector_path):
        hb.log(f'ERROR: Zones vector not found: {zones_vector_path}')
        hb.log('  Expected location: ~/Files/base_data/cartographic/ee/ee_r264_correspondence.gpkg')
        return p
        
    # Load zones vector
    hb.log(f'Loading zones from: {zones_vector_path}')
    gdf_zones = gpd.read_file(zones_vector_path)
    hb.log(f'  Loaded {len(gdf_zones)} zones')
    
    # Get max zone ID to set size of accumulator arrays
    max_zone_id = int(gdf_zones[zones_id_column].max()) + 1
    # Ensure zone ID column is integer (handles float64 from some GIS formats)
    gdf_zones[zones_id_column] = gdf_zones[zones_id_column].astype(int)
    
    hb.log(f'  Zone ID range: 1-{max_zone_id - 1}')
    
    # Create zone lookup tables
    zone_lookup = dict(zip(gdf_zones[zones_id_column], gdf_zones[zones_label_column]))
    
    # Compute scenario-specific zonal stats and return before the legacy single-output code below.
    avoided_dir = _get_avoided_output_dir(p)
    for year in p.prediction_years:
        hb.log(f'\nProcessing year {year}...')
        for scenario_name, _, _ in scenario_specs:
            avoided_mort_path = os.path.join(avoided_dir, f'avoided_mortality_{scenario_name}_{year}.tif')
            value_path = os.path.join(avoided_dir, f'avoided_mortality_value_{scenario_name}_{year}.tif')
            if not os.path.exists(avoided_mort_path):
                hb.log(f'WARNING: Missing avoided mortality raster for {scenario_name}: {avoided_mort_path}')
                continue
            if not os.path.exists(value_path):
                hb.log(f'WARNING: Missing avoided mortality value raster for {scenario_name}: {value_path}')
                continue

            gpkg_path = os.path.join(p.cur_dir, f'{year}_{scenario_name}_zonal_stats.gpkg')
            csv_path = os.path.join(p.cur_dir, f'{year}_{scenario_name}_zonal_stats.csv')
            if (not getattr(p, 'force_run', False)) and os.path.exists(gpkg_path) and os.path.exists(csv_path):
                hb.log(f'✓ Zonal stats already exist for {year} / {scenario_name}')
                continue

            _compute_zonal_stats_for_pair(
                p=p,
                year=year,
                scenario_name=scenario_name,
                avoided_mort_path=avoided_mort_path,
                value_path=value_path,
                zones_vector_path=zones_vector_path,
                zones_id_column=zones_id_column,
                zones_label_column=zones_label_column,
                gdf_zones=gdf_zones,
                zone_lookup=zone_lookup,
                max_zone_id=max_zone_id,
                gpkg_path=gpkg_path,
                csv_path=csv_path,
            )

    return p


def _save_table_multiple_formats(df, base_path, index=False, float_fmt='%.3f', md_text=None, tex_text=None):
    """Save one table as CSV, LaTeX, and Markdown for flexible downstream use.

    Parameters
    ----------
    df : pd.DataFrame
        Table data used for CSV export and as fallback for Markdown/LaTeX.
    base_path : str
        Output path without file extension.
    index : bool
        Whether to write index to outputs.
    float_fmt : str
        Float format used by default LaTeX fallback.
    md_text : str, optional
        If provided, write this markdown text instead of DataFrame markdown.
    tex_text : str, optional
        If provided, write this LaTeX text instead of DataFrame latex wrapper.
    """
    csv_path = f'{base_path}.csv'
    tex_path = f'{base_path}.tex'
    md_path = f'{base_path}.md'

    df.to_csv(csv_path, index=index)
    with open(md_path, 'w') as f:
        if md_text is not None:
            f.write(md_text)
        else:
            f.write(df.to_markdown(index=index))
    # Wrap LaTeX output in table environment
    with open(tex_path, 'w') as f:
        if tex_text is not None:
            f.write(tex_text)
        else:
            latex_table = df.to_latex(index=index, float_format=lambda x: float_fmt % x)
            f.write('\\begin{table}\n\\centering\n')
            f.write(latex_table)
            f.write('\n\\end{table}\n')

    hb.log(f'✓ Saved table: {csv_path}')
    hb.log(f'✓ Saved table: {tex_path}')
    hb.log(f'✓ Saved table: {md_path}')


def export_summary_stats(p):
    """Compute and export summary statistics for numeric study variables.
    Writes `summary_stats.csv` (and JSON) into the target assets folder so the
    manuscript can load it directly.
    """

    table_path = os.path.join(p.build_estimation_table_dir, 'estimation_table.parquet')
    hb.log(f'Loading sample for summary stats: {table_path}')
    try:
        df = pd.read_parquet(table_path)
    except Exception as e:
        hb.log(f'Error loading estimation table: {e}')
        return None

    variable_specs = [
        {
            'aliases': ['std_slope_degree', 'slope_degree'],
            'label': 'Slope',
            'note': 'slope in degrees, standardized within the estimation sample',
        },
        {
            'aliases': ['roughness'],
            'label': 'Roughness',
            'note': 'local terrain roughness index capturing fine-scale slope variability',
        },
        {
            'aliases': ['tpi'],
            'label': 'Topographic Position Index',
            'note': 'relative position in the surrounding terrain, with positive values indicating ridges and negative values indicating valleys',
        },
        {
            'aliases': ['elev_stdev'],
            'label': 'Elevation Variability',
            'note': 'within-neighborhood elevation standard deviation, used as a local terrain heterogeneity measure',
        },
        {
            'aliases': ['std_road_density', 'road_density'],
            'label': 'Road Density',
            'note': 'road length per square kilometer from the road network raster',
        },
        {
            'aliases': ['dist_to_fault_km'],
            'label': 'Distance to Fault',
            'note': 'distance to the nearest active fault trace in kilometers',
        },
        {
            'aliases': ['deforestation_forest_share_1yr'],
            'label': 'Deforestation (1yr)',
            'note': 'one-year trailing change in forest cover share used to approximate recent clearing pressure',
        },
        {
            'aliases': ['deforestation_forest_share_3yr'],
            'label': 'Deforestation (3yr)',
            'note': 'three-year trailing change in forest cover share used to smooth short-run clearing pressure',
        },
        {
            'aliases': ['std_population_log1p', 'population', 'std_population'],
            'label': 'Population (log scale)',
            'note': 'log1p population or population density, standardized within the estimation sample',
        },
        {
            'aliases': ['std_rain_max_daily', 'rain_max_daily'],
            'label': 'Extreme Precipitation',
            'note': 'annual maximum daily precipitation derived from ERA5-Land and standardized within the estimation sample',
        },
        {
            'aliases': ['std_forest_share', 'forest_share'],
            'label': 'Forest Cover Share',
            'note': 'share of the pixel classified as forest cover in the corresponding raster year',
        },
        {
            'aliases': ['std_othernat_share', 'othernat_share'],
            'label': 'Other Natural Veg. Share',
            'note': 'share of the pixel classified as other natural vegetation in the corresponding raster year',
        },
    ]

    def _fmt_number(value):
        if pd.isna(value):
            return ''
        absolute = abs(float(value))
        if absolute < 10:
            return f'{float(value):.3f}'
        return f'{float(value):.1f}'

    rows = []
    note_lines = []
    used_columns = []

    for spec in variable_specs:
        column_name = next((alias for alias in spec['aliases'] if alias in df.columns), None)
        if column_name is None:
            continue

        series = pd.to_numeric(df[column_name], errors='coerce')
        series = series.dropna()
        if series.empty:
            continue

        mean_value = float(series.mean())
        std_value = float(series.std(ddof=1)) if len(series) > 1 else np.nan
        rows.append({
            'Variable': spec['label'],
            'Mean (SD)': f'{_fmt_number(mean_value)} ({_fmt_number(std_value)})' if pd.notna(std_value) else _fmt_number(mean_value),
            'Range (Min, Max)': f'({_fmt_number(series.min())}, {_fmt_number(series.max())})',
        })
        used_columns.append(column_name)
        note_lines.append(f"{spec['label']} = {spec['note']}")

    if not rows:
        hb.log('No study variables found in sample to summarize.')
        return None

    stats = pd.DataFrame(rows)

    csv_path = os.path.join(p.visualizations_dir, 'summary_stats.csv')
    json_path = os.path.join(p.visualizations_dir, 'summary_stats.json')
    meta_path = os.path.join(p.visualizations_dir, 'summary_stats_metadata.json')

    stats.to_csv(csv_path, index=False)
    stats.to_json(json_path, orient='records', indent=2)
    # Prefer year information from the estimation table if present
    year_range_str = ''
    yrs = None
    if 'year' in df.columns:
        try:
            yrs = sorted({int(y) for y in df['year'].dropna().unique()})
        except Exception:
            yrs = None
    # Fallback to project-configured prediction_years
    if not yrs and hasattr(p, 'prediction_years') and getattr(p, 'prediction_years'):
        try:
            yrs = sorted({int(y) for y in getattr(p, 'prediction_years')})
        except Exception:
            yrs = None
    if yrs:
        # If contiguous years, show as start–end; otherwise list years
        if len(yrs) > 1 and all(yrs[i] + 1 == yrs[i + 1] for i in range(len(yrs) - 1)):
            year_range_str = f'; years = {yrs[0]}–{yrs[-1]}'
        else:
            year_range_str = '; years = ' + ','.join(str(y) for y in yrs)

    metadata = {
        'n_obs': int(len(df)),
        'n_variables': int(len(rows)),
        'used_columns': used_columns,
        'notes': (
            f'Summary statistics computed for the estimation sample (N = {len(df):,} pixel-years, 2010-2017). '
            'Statistics show raw variable values before standardization. '
            'Slope is measured in degrees; roughness and elevation variability are terrain indices from Geomorpho90m; '
            'road density is road length per km²; distance to fault is in kilometers; '
            'deforestation variables are one-year and three-year trailing changes in forest cover share; '
            'population is in persons per pixel; extreme precipitation is annual maximum daily precipitation in mm; '
            'forest and vegetation shares are proportions (0-1).'
        ),
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Also export HTML and LaTeX table versions (publication style)
    output_dir = getattr(p, 'visualizations_dir', None) or getattr(p, 'dissertation_dir', None) or os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Build HTML
    html_lines = [
        '<table class="table table-sm table-striped">',
        '  <thead>',
        '    <tr><th>Variable</th><th>Mean (SD)</th><th>Range (Min, Max)</th></tr>',
        '  </thead>',
        '  <tbody>',
    ]
    for _, row in stats.iterrows():
        html_lines.append(f'    <tr><td>{row["Variable"]}</td><td>{row["Mean (SD)"]}</td><td>{row["Range (Min, Max)"]}</td></tr>')
    html_lines.extend([
        '  </tbody>',
        '</table>',
        f'<p class="table-notes">Notes: {metadata["notes"]}</p>',
        '',
    ])
    html_text = '\n'.join(html_lines)

    html_path = os.path.join(output_dir, 'summary_stats.html')
    with open(html_path, 'w') as f:
        f.write(html_text)

    # Build LaTeX
    tex_lines = [
        '\\centering',
        '\\begin{tabular}{lrr}',
        '\\toprule',
        'Variable & Mean (SD) & Range (Min, Max) \\\\',
        '\\midrule',
    ]
    for _, row in stats.iterrows():
        var = row['Variable']
        mean_sd = row['Mean (SD)']
        rng = row['Range (Min, Max)']
        # Escape percent signs etc. Assume values are safe strings
        tex_lines.append(f'{var} & {mean_sd} & {rng} \\\\')
    tex_lines.extend([
        '\\bottomrule',
        '\\end{tabular}',
        '\\vspace{2pt}',
        '\\begin{minipage}{0.92\\linewidth}',
        f'\\footnotesize Notes: {metadata["notes"]}',
        '\\end{minipage}',
        '',
    ])
    tex_text = '\n'.join(tex_lines)

    tex_path = os.path.join(output_dir, 'summary_stats.tex')
    with open(tex_path, 'w') as f:
        f.write(tex_text)

    hb.log(f'✓ Saved summary stats CSV: {csv_path}')
    hb.log(f'✓ Saved summary stats JSON: {json_path}')
    hb.log(f'✓ Saved summary stats metadata: {meta_path}')
    hb.log(f'✓ Saved summary stats HTML: {html_path}')
    hb.log(f'✓ Saved summary stats LaTeX: {tex_path}')
    return {'csv': csv_path, 'json': json_path, 'html': html_path, 'tex': tex_path}


def _regression_stars(pval):
    if pd.isna(pval):
        return ''
    if pval < 0.001:
        return '***'
    if pval < 0.01:
        return '**'
    if pval < 0.05:
        return '*'
    return ''


def _format_regression_publication_table(model_label, coef_df, metadata, config):
    """Build a publication-style two-column regression table and md/tex strings.

    Returns
    -------
    table_df, md_text, tex_text
    """
    dep_label = config.get('dep_label', 'Estimate')
    table_title = config.get('title', f'{model_label.title()} Model')
    table_id = config.get('table_id', f'tbl-{model_label}-publication')
    term_labels = config.get('term_labels', {})
    include_terms = config.get('include_terms')
    variable_notes = config.get('variable_notes', {})

    if include_terms is None:
        include_terms = [t for t in coef_df['term'].tolist() if not str(t).startswith('gaez_')]

    rows = []
    available = set(coef_df['term'].tolist())
    for term in include_terms:
        if term not in available:
            continue
        row = coef_df.loc[coef_df['term'] == term].iloc[0]
        label = term_labels.get(term, term)
        coef_txt = f"{row['coefficient']:.3f}{_regression_stars(row['p_value'])}"
        se_txt = f"({row['std_error']:.3f})" if pd.notna(row['std_error']) else ''
        rows.append({' ': label, dep_label: coef_txt})
        rows.append({' ': '', dep_label: se_txt})

    if rows:
        rows.append({' ': '', dep_label: ''})

    # Bottom summary rows
    rows.append({' ': 'Observations', dep_label: f"{int(metadata['n_obs']):,}" if metadata.get('n_obs') is not None else ''})
    rows.append({' ': 'Events (nonzero outcome)', dep_label: f"{int(metadata['n_events']):,}" if metadata.get('n_events') is not None else ''})
    rows.append({' ': 'Dependent var. mean', dep_label: f"{metadata['dep_mean']:.3f}" if metadata.get('dep_mean') is not None else ''})
    rows.append({' ': 'Pseudo R² (McFadden)', dep_label: f"{metadata['pseudo_r2_mcfadden']:.3f}" if metadata.get('pseudo_r2_mcfadden') is not None else ''})
    rows.append({' ': 'Fixed effects', dep_label: 'GAEZ (17)' if metadata.get('has_fixed_effects') else 'No'})

    table_df = pd.DataFrame(rows)

    md_lines = [
        f'Table: {table_title} {{#{table_id}}}',
        '',
        table_df.to_markdown(index=False),
        '',
        '`\\footnotesize`{=latex}',
        '<div class="tablenotes">',
        '',
        'Notes: Complementary log-log model estimated on case-control sample with 50:1 control-to-case ratio, stratified by GAEZ zones. Standard errors in parentheses. * p < 0.05, ** p < 0.01, *** p < 0.001. Variable definitions: Slope = terrain slope in degrees; Roughness = local terrain roughness index; Topographic Position Index = relative elevation position (positive = ridges, negative = valleys); Elevation Variability = within-neighborhood elevation standard deviation; Road Density = road length per km²; Distance to Fault = distance to nearest active fault in km; Deforestation (1yr) and (3yr) = one-year and three-year trailing changes in forest cover share.',
        '',
        '</div>',
        '`\\normalsize`{=latex}',
        '',
    ]
    md_text = '\n'.join(md_lines)

    tex_table = table_df.to_latex(index=False, escape=True)
    notes_text = 'Logit model estimated on case-control sample with 50:1 control-to-case ratio, stratified by GAEZ zones. Standard errors in parentheses. * p < 0.05, ** p < 0.01, *** p < 0.001. Variable definitions: Slope = terrain slope in degrees; Roughness = local terrain roughness index; Topographic Position Index = relative position in the surrounding terrain, with positive values indicating ridges and negative values indicating valleys; Elevation Variability = within-neighborhood elevation standard deviation, used as a local terrain heterogeneity measure; Road Density = road length per square kilometer from the road network raster; Distance to Fault = distance to the nearest active fault trace in kilometers; Extreme Precipitation = annual maximum daily precipitation in mm; Deforestation (1yr) and (3yr) = one-year and three-year trailing changes in forest cover share.'
    tex_lines = [
        '\\begin{table}[htbp]',
        '\\centering',
        f'\\caption{{{table_title}}}',
        tex_table,
        '\\vspace{2pt}',
        '\\begin{minipage}{0.92\\linewidth}',
        f'\\footnotesize Notes: {notes_text}',
        '\\end{minipage}',
        '\\end{table}',
        '',
    ]
    tex_text = '\n'.join(tex_lines)

    return table_df, md_text, tex_text


def plot_glc_from_vector(p):
    """
    Plot UGLC buffered event geometry as a point map with fatality bins.

    Output
    ------
    {cur_dir}/uglc_events_fatality_bins.png
    """
    uglc_mort_path = os.path.join(p.base_data_dir, 'preprocess_data', 'uglc', 'uglc_buffered_points.gpkg')
    if not os.path.exists(uglc_mort_path):
        uglc_mort_path = os.path.join(p.visualizations_dir, 'uglc_buffered_points.gpkg')

    out_png = os.path.join(p.visualizations_dir, 'uglc_events_fatality_bins.png')
    if os.path.exists(out_png):
        hb.log(f'✓ UGLC events plot already exists: {out_png}')
        return p

    hb.log(f'Plotting UGLC events from: {uglc_mort_path}')
    gdf = gpd.read_file(uglc_mort_path)
    if gdf.empty:
        raise ValueError('WARNING: UGLC vector is empty, skipping.')

    gdf = gdf.to_crs('EPSG:3857')
    points = gdf.copy()
    points['geometry'] = points.geometry.representative_point()

    fatalities = points['fatality_count'].fillna(0).clip(lower=0)
    nonfatal = fatalities <= 0
    bins = {
        '1-5': (fatalities >= 1) & (fatalities < 5),
        '5-25': (fatalities >= 5) & (fatalities < 25),
        '25-100': (fatalities >= 25) & (fatalities < 100),
        '100+': fatalities >= 100,
    }

    size_map = {'1-5': 12, '5-25': 24, '25-100': 44, '100+': 80}
    color_map = {
        '1-5': '#f28e2b',
        '5-25': '#e15759',
        '25-100': '#b51d39',
        '100+': '#5b0f1f',
    }

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=220)
    if nonfatal.any():
        points.loc[nonfatal].plot(
            ax=ax,
            color='#6c757d',
            markersize=6,
            alpha=0.32,
            linewidth=0,
            zorder=2,
        )

    for label, mask in bins.items():
        if mask.any():
            points.loc[mask].plot(
                ax=ax,
                color=color_map[label],
                markersize=size_map[label],
                alpha=0.82,
                linewidth=0,
                zorder=3,
            )

    legend_handles = [
        Line2D(
            [0], [0], marker='o', color='none', label='Nonfatal landslide',
            markerfacecolor='#6c757d', markeredgecolor='none', markersize=6, alpha=0.4
        ),
        Line2D(
            [0], [0], marker='o', color='none', label='1-5 deaths',
            markerfacecolor=color_map['1-5'], markeredgecolor='none', markersize=5, alpha=0.9
        ),
        Line2D(
            [0], [0], marker='o', color='none', label='5-25 deaths',
            markerfacecolor=color_map['5-25'], markeredgecolor='none', markersize=6, alpha=0.9
        ),
        Line2D(
            [0], [0], marker='o', color='none', label='25-100 deaths',
            markerfacecolor=color_map['25-100'], markeredgecolor='none', markersize=7, alpha=0.9
        ),
        Line2D(
            [0], [0], marker='o', color='none', label='100+ deaths',
            markerfacecolor=color_map['100+'], markeredgecolor='none', markersize=9, alpha=0.9
        ),
    ]

    ax.legend(
        handles=legend_handles,
        title='Fatality count',
        loc='lower left',
        frameon=True,
        framealpha=0.9,
        facecolor='white',
        edgecolor='none',
        title_fontproperties= {'family': 'serif', 'size': 9},
        prop={'family': 'serif', 'size': 8},
    )
    

    if ctx is not None:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, crs=points.crs, attribution=False)

    ax.set_axis_off()
    # ax.set_title('Global Landslide Events by Fatality Bin', fontsize=11, fontname='serif')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    hb.log(f'✓ Saved figure: {out_png}')
    return p


def plot_global_rasters_png(p):
    os.makedirs(p.visualizations_dir, exist_ok=True)
    failures = []
    scenario_specs = getattr(p, 'forest_value_scenarios', [])

    def _plot_raster(raster_path, out_png, title, cmap, q_low, q_high, cbar_format):
        if os.path.exists(out_png):
            hb.log(f'✓ Plot already exists: {out_png}')
            return True
        if not os.path.exists(raster_path):
            msg = f'Raster not found: {raster_path}'
            hb.log(f'WARNING: {msg}')
            failures.append(msg)
            return False

        try:
            with rasterio.open(raster_path) as src:
                max_dim = int(getattr(p, 'plot_raster_max_dim', 4096))
                scale = max(src.height / max_dim, src.width / max_dim, 1.0)
                out_h = max(1, int(np.ceil(src.height / scale)))
                out_w = max(1, int(np.ceil(src.width / scale)))
                arr = src.read(
                    1,
                    out_shape=(out_h, out_w),
                    resampling=Resampling.nearest,
                ).astype(np.float32)
                ndv = src.nodata

            if ndv is not None:
                arr = np.where(arr == ndv, np.nan, arr)

            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                msg = f'Raster has no finite data: {raster_path}'
                hb.log(f'WARNING: {msg}')
                failures.append(msg)
                return False

            vmin, vmax = np.nanpercentile(finite, [q_low, q_high])
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                msg = f'Invalid plotting range: {raster_path}'
                hb.log(f'WARNING: {msg}')
                failures.append(msg)
                return False

            if vmin == vmax:
                eps = abs(vmin) * 1e-6 if vmin != 0 else 1e-6
                vmin -= eps
                vmax += eps

            fig, ax = plt.subplots(figsize=(9, 4.8), dpi=220)
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_axis_off()
            cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.4)
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.yaxis.set_major_formatter(mtick.StrMethodFormatter(cbar_format))
            plt.tight_layout()
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close(fig)
            hb.log(f'✓ Saved figure: {out_png}')
            return True
        except Exception as e:
            msg = f'Failed plotting {os.path.basename(raster_path)}: {e}'
            hb.log(f'WARNING: {msg}')
            failures.append(msg)
            plt.close('all')
            return False

    for year in p.prediction_years:
        hb.log(f'\nPlotting fitted_landslide_hazard for {year}...')
        _plot_raster(
            os.path.join(p.stitch_tiles_dir, f'fitted_landslide_hazard_{year}.tif'),
            os.path.join(p.visualizations_dir, f'fitted_landslide_hazard_{year}.png'),
            f'Fitted Landslide Hazard, {year}',
            'Reds',
            0,
            100,
            '{x:.3f}',
        )

        for scenario_name, _, _ in scenario_specs:
            raster_base = os.path.join(_get_avoided_output_dir(p), f'avoided_mortality_{scenario_name}_{year}.tif')
            value_base = os.path.join(_get_avoided_output_dir(p), f'avoided_mortality_value_{scenario_name}_{year}.tif')
            hb.log(f'\nPlotting avoided mortality for {year} / {scenario_name}...')
            _plot_raster(
                raster_base,
                os.path.join(p.visualizations_dir, f'avoided_mortality_{scenario_name}_{year}.png'),
                f'Avoided Landslide Mortality ({scenario_name}), {year}',
                'Purples',
                2,
                98,
                '{x:.3f}',
            )
            _plot_raster(
                value_base,
                os.path.join(p.visualizations_dir, f'avoided_mortality_value_{scenario_name}_{year}.png'),
                f'Economic Value of Avoided Mortality ({scenario_name}), {year}',
                'Greens',
                2,
                98,
                '${x:,.0f}',
            )

    if failures:
        hb.log(f'plot_global_rasters_png completed with {len(failures)} warnings.')

    return p


def export_results_tables(p):
    """Export core results tables from scenario-specific zonal statistics."""
    scenario_specs = getattr(p, 'forest_value_scenarios', [])
    if not scenario_specs:
        hb.log('WARNING: p.forest_value_scenarios is empty; skipping results table export.')
        return p

    for year in p.prediction_years:
        for scenario_name, _, _ in scenario_specs:
            zonal_csv = os.path.join(p.compute_zonal_statistics_dir, f'{year}_{scenario_name}_zonal_stats.csv')
            if not os.path.exists(zonal_csv):
                hb.log(f'WARNING: Missing zonal table for {year} / {scenario_name}, skipping table export: {zonal_csv}')
                continue

            df = pd.read_csv(zonal_csv)

            if 'type' in df.columns:
                df = df[df['type'].fillna('').str.lower() == 'sovereign country'].copy()

            base_suffix = f'{scenario_name}_{year}'

            # Global summary
            summary = pd.DataFrame([
                {
                    'year': year,
                    'scenario': scenario_name,
                    'avoided_mortality_sum': float(df['avoided_mortality_sum'].sum(skipna=True)),
                    'avoided_mortality_mean': float(df['avoided_mortality_mean'].mean(skipna=True)),
                    'avoided_mortality_median': float(df['avoided_mortality_median'].median(skipna=True)),
                    'value_sum': float(df['value_sum'].sum(skipna=True)),
                    'value_mean': float(df['value_mean'].mean(skipna=True)),
                    'value_median': float(df['value_median'].median(skipna=True)),
                }
            ])
            _save_table_multiple_formats(summary, os.path.join(p.visualizations_dir, f'global_summary_{base_suffix}'), index=False)

            # Top countries by mortality and value
            country_col = 'name_long' if 'name_long' in df.columns else ('ee_r264_label' if 'ee_r264_label' in df.columns else 'iso3')
            top_mort = (
                df[[country_col, 'iso3', 'region_wb', 'avoided_mortality_sum', 'value_sum']]
                .sort_values('avoided_mortality_sum', ascending=False)
                .head(15)
                .reset_index(drop=True)
            )
            _save_table_multiple_formats(top_mort, os.path.join(p.visualizations_dir, f'top_countries_mortality_{base_suffix}'), index=False)

            top_value = (
                df[[country_col, 'iso3', 'region_wb', 'avoided_mortality_sum', 'value_sum']]
                .sort_values('value_sum', ascending=False)
                .head(15)
                .reset_index(drop=True)
            )
            _save_table_multiple_formats(top_value, os.path.join(p.visualizations_dir, f'top_countries_value_{base_suffix}'), index=False)

            # Region WB summary
            if 'region_wb' in df.columns:
                region = (
                    df.groupby('region_wb', dropna=False, as_index=False)[['avoided_mortality_sum', 'value_sum']]
                    .sum()
                    .sort_values('avoided_mortality_sum', ascending=False)
                )
                _save_table_multiple_formats(region, os.path.join(p.visualizations_dir, f'region_wb_summary_{base_suffix}'), index=False)

    return p


def export_hazard_model_table(p):
    """Export the hazard model as a single markdown publication table."""

    candidates = []
    if hasattr(p, 'estimate_hazard_model_dir') and p.estimate_hazard_model_dir:
        candidates.append(p.estimate_hazard_model_dir)
    if hasattr(p, 'project_dir') and p.project_dir:
        candidates.append(os.path.join(p.project_dir, 'intermediate', 'estimate_hazard_model'))

    model_dir = None
    for candidate in candidates:
        if not candidate:
            continue
        hazard_path = os.path.join(candidate, 'hazard_model.json')
        if os.path.exists(hazard_path):
            model_dir = candidate
            break

    if model_dir is None:
        hb.log('WARNING: No estimate_hazard_model directory found for hazard table export.')
        return p

    hazard_model_path = os.path.join(model_dir, 'hazard_model.json')
    with open(hazard_model_path, 'r') as f:
        hazard_data = json.load(f)

    def _term_label(term):
        if term == 'const':
            return 'Intercept'
        pretty = {
            'slope_degree': 'Slope',
            'roughness': 'Roughness',
            'tpi': 'Topographic Position Index',
            'elev_stdev': 'Elevation Variability',
            'road_density': 'Road Density',
            'dist_to_fault_km': 'Distance to Fault',
            'deforestation_forest_share_1yr': 'Deforestation (1yr)',
            'deforestation_forest_share_3yr': 'Deforestation (3yr)',
        }
        return pretty.get(term, term.replace('_', ' ').title())

    def _star_text(p_value):
        if pd.isna(p_value):
            return ''
        if p_value < 0.001:
            return '***'
        if p_value < 0.01:
            return '**'
        if p_value < 0.05:
            return '*'
        return ''

    coefficients = hazard_data.get('coefficients', {})
    pvalues = hazard_data.get('pvalues', {})
    std_errors = hazard_data.get('std_errors', {})
    marginal_effects_data = hazard_data.get('marginal_effects', {})
    marginal_effects_at_mean = marginal_effects_data.get('at_mean', {})

    include_terms = [
        'const',
        'slope_degree',
        'roughness',
        'tpi',
        'elev_stdev',
        'road_density',
        'dist_to_fault_km',
        'rain_max_daily',
        'deforestation_forest_share_1yr',
        'deforestation_forest_share_3yr',
    ]

    variable_notes = {
        'slope_degree': 'terrain slope in degrees derived from the DEM and standardized within the estimation sample',
        'roughness': 'local terrain roughness index capturing fine-scale slope variability',
        'tpi': 'relative position in the surrounding terrain, with positive values indicating ridges and negative values indicating valleys',
        'elev_stdev': 'within-neighborhood elevation standard deviation, used as a local terrain heterogeneity measure',
        'road_density': 'road length per square kilometer from the road network raster',
        'dist_to_fault_km': 'distance to the nearest active fault trace in kilometers',
        'rain_max_daily': 'annual maximum daily precipitation from ERA5-Land in mm',
        'deforestation_forest_share_1yr': 'one-year trailing change in forest cover share used to approximate recent clearing pressure',
        'deforestation_forest_share_3yr': 'three-year trailing change in forest cover share used to smooth short-run clearing pressure',
    }

    rows = []
    for term in include_terms:
        if term not in coefficients:
            continue
        coefficient = coefficients[term]
        p_value = pvalues.get(term, np.nan)
        std_error = std_errors.get(term, np.nan)
        me = marginal_effects_at_mean.get(term, np.nan)

        me_scale = 10.0 if term.startswith('deforestation_') else 100.0
        me_text = f'{float(me) * me_scale:.4f}' if pd.notna(me) else ''
        rows.append({
            ' ': _term_label(term),
            'Coef.': f'{float(coefficient):.3f}{_star_text(p_value)}',
            'Marginal Effect (pp)': me_text,
        })

        se_text = f'({float(std_error):.3f})' if pd.notna(std_error) else ''
        rows.append({' ': '', 'Coef.': se_text, 'Marginal Effect (pp)': ''})

    rows.append({' ': 'Observations', 'Coef.': f"{int(hazard_data.get('n_obs', 0)):,}" if hazard_data.get('n_obs') is not None else '', 'Marginal Effect (pp)': ''})
    rows.append({' ': 'Events (nonzero outcome)', 'Coef.': f"{int(hazard_data.get('n_events', 0)):,}" if hazard_data.get('n_events') is not None else '', 'Marginal Effect (pp)': ''})
    rows.append({' ': 'Dependent var. mean', 'Coef.': f"{float(hazard_data.get('dep_mean')):.3f}" if hazard_data.get('dep_mean') is not None else '', 'Marginal Effect (pp)': ''})
    rows.append({' ': 'Pseudo R² (McFadden)', 'Coef.': f"{float(hazard_data.get('pseudo_r2_mcfadden')):.3f}" if hazard_data.get('pseudo_r2_mcfadden') is not None else '', 'Marginal Effect (pp)': ''})
    rows.append({' ': 'Fixed effects', 'Coef.': 'GAEZ (17)' if any(str(term).startswith('gaez_') for term in coefficients) else 'No', 'Marginal Effect (pp)': ''})

    p_at_mean = marginal_effects_data.get('p_at_mean', np.nan)
    if pd.notna(p_at_mean):
        rows.append({' ': 'Pred. Prob. at mean', 'Coef.': f'{float(p_at_mean):.4f}', 'Marginal Effect (pp)': ''})

    table_df = pd.DataFrame(rows)

    table_title = 'Hazard Model of Landslide Occurrence'
    note_lines = []
    for term in include_terms:
        if term == 'const':
            continue
        description = variable_notes.get(term)
        if description:
            note_lines.append(f'{_term_label(term)} = {description}')

    notes_text = ('Complementary log-log model estimated on case-control sample with 50:1 control-to-case ratio, '
                  'stratified by GAEZ zones. Coefficients are on the link scale; marginal effects are reported in '
                  'percentage points and show the change in predicted landslide probability for a one-unit increase '
                  'in the predictor, evaluated at the mean of all covariates. For deforestation share variables, the '
                  'reported marginal effect corresponds to a 10 percentage-point increase in forest loss. Marginal effects '
                  'for standardized variables reflect a one-SD increase. Standard errors are shown on the row beneath '
                  'coefficients. * p < 0.05, ** p < 0.01, *** p < 0.001. Variable definitions: Slope = terrain slope in degrees; Roughness = local '
                  'terrain roughness index; Topographic Position Index = relative elevation position (positive = ridges, '
                  'negative = valleys); Elevation Variability = within-neighborhood elevation standard deviation; Road Density = '
                  'road length per km²; Distance to Fault = distance to nearest active fault in km; Extreme Precipitation = '
                  'annual maximum daily precipitation in mm; Deforestation (1yr) and (3yr) = one-year and three-year trailing '
                  'changes in forest cover share.')

    html_lines = [
        table_df.to_html(index=False, escape=False, border=0, classes='table table-sm table-striped'),
        f'<p class="table-notes">Notes: {notes_text}</p>',
        '',
    ]
    html_text = '\n'.join(html_lines)

    tex_lines = [
        '\\centering',
        table_df.to_latex(index=False, escape=True),
        '\\vspace{2pt}',
        '\\begin{minipage}{0.92\\linewidth}',
        f'\\footnotesize Notes: {notes_text}',
        '\\end{minipage}',
        '',
    ]
    tex_text = '\n'.join(tex_lines)

    output_dir = getattr(p, 'visualizations_dir', None) or getattr(p, 'dissertation_dir', None)
    if output_dir is None:
        hb.log('WARNING: No output directory available for hazard table export.')
        return p

    output_html_path = os.path.join(output_dir, 'hazard_model_publication_table.html')
    with open(output_html_path, 'w') as f:
        f.write(html_text)

    output_tex_path = os.path.join(output_dir, 'hazard_model_publication_table.tex')
    with open(output_tex_path, 'w') as f:
        f.write(tex_text)

    hb.log(f'✓ Saved hazard table: {output_html_path}')
    hb.log(f'✓ Saved hazard table: {output_tex_path}')
    return p


def export_mortality_model_table(p):
    """Export mortality model as two separate panel tables (hurdle and severity)."""

    stage2_path = os.path.join(p.estimate_hazard_model_dir, 'severity_model.json')
    with open(stage2_path, 'r') as f:
        mortality_data = json.load(f)

    def _term_label(term):
        if term == 'const':
            return 'Intercept'
        if term.startswith('gaez_'):
            return f'GAEZ {term.split("_")[1]}'
        pretty = {
            'std_population_log1p': 'Population (log scale)',
            'std_rain_max_daily': 'Extreme Precipitation',
            'std_slope_degree': 'Slope',
            'std_road_density': 'Road Density',
            'std_forest_share': 'Forest Cover Share',
            'std_othernat_share': 'Other Natural Veg. Share',
            'population': 'Population (log scale)',
            'rain_max_daily': 'Extreme Precipitation',
            'slope_degree': 'Slope',
            'road_density': 'Road Density',
            'forest_share': 'Forest Cover Share',
            'othernat_share': 'Other Natural Veg. Share',
        }
        return pretty.get(term, term.replace('_', ' ').title())

    def _star_text(p_value):
        if pd.isna(p_value):
            return ''
        if p_value < 0.001:
            return '***'
        if p_value < 0.01:
            return '**'
        if p_value < 0.05:
            return '*'
        return ''

    def _build_panel(stage_key, dep_label):
        """Build a single panel table for hurdle or severity, using actual predictors from model."""
        coef_key = f'{stage_key}_coefficients'
        pval_key = f'{stage_key}_pvalues'
        se_key = f'{stage_key}_std_errors'
        pred_key = f'{stage_key}_predictors'

        coefficients = mortality_data.get(coef_key, {})
        pvalues = mortality_data.get(pval_key, {})
        std_errors = mortality_data.get(se_key, {})
        predictors = mortality_data.get(pred_key, [])

        if not coefficients:
            return None

        rows = []
        # Always add intercept first if present
        if 'const' in coefficients:
            coefficient = coefficients['const']
            p_value = pvalues.get('const', np.nan)
            std_error = std_errors.get('const', np.nan)
            rows.append({' ': _term_label('const'), dep_label: f'{float(coefficient):.3f}{_star_text(p_value)}'})
            rows.append({' ': '', dep_label: f'({float(std_error):.3f})' if pd.notna(std_error) else ''})

        # Add non-GAEZ predictors
        non_gaez_terms = [t for t in predictors if not t.startswith('gaez_')]
        for term in non_gaez_terms:
            if term not in coefficients:
                continue
            coefficient = coefficients[term]
            p_value = pvalues.get(term, np.nan)
            std_error = std_errors.get(term, np.nan)
            rows.append({' ': _term_label(term), dep_label: f'{float(coefficient):.3f}{_star_text(p_value)}'})
            rows.append({' ': '', dep_label: f'({float(std_error):.3f})' if pd.notna(std_error) else ''})

        # Add GAEZ fixed effects summary (with space before)
        gaez_terms = [t for t in predictors if t.startswith('gaez_')]
        if stage_key == 'severity':
            n_obs = mortality_data.get('severity_n_obs', mortality_data.get('n_obs'))
        else:
            n_obs = mortality_data.get('n_obs')
        n_events = mortality_data.get(f'{stage_key}_n_events')
        dep_mean = mortality_data.get(f'{stage_key}_dep_mean')
        pseudo_r2 = mortality_data.get(f'{stage_key}_pseudo_r2_mcfadden')

        rows.append({' ': 'Observations', dep_label: f"{int(n_obs):,}" if n_obs is not None else ''})
        if stage_key != 'severity':
            rows.append({' ': 'Events (nonzero outcome)', dep_label: f"{int(n_events):,}" if n_events is not None else ''})
        rows.append({' ': 'Dependent var. mean', dep_label: f"{float(dep_mean):.3f}" if dep_mean is not None else ''})
        rows.append({' ': 'Pseudo R² (McFadden)', dep_label: f"{float(pseudo_r2):.3f}" if pseudo_r2 is not None else ''})
        if gaez_terms:
            rows.append({' ': 'Fixed effects', dep_label: f'GAEZ ({len(gaez_terms)})'})

        table_df = pd.DataFrame(rows)
        return table_df

    def _latex_body(df):
        lines = df.to_latex(index=False, escape=True).split('\n')
        return lines[4:-3]  # Skip header AND final bottomrule/end

    def _html_rows(df):
        lines = df.to_html(index=False, escape=False, border=0).split('\n')
        body = []
        in_body = False
        for line in lines:
            if '<tbody>' in line:
                in_body = True
                continue
            if '</tbody>' in line:
                break
            if in_body:
                body.append(line)
        return body

    def _export_single_panel(panel_key, dep_label, include_notes=False):
        """Export a single panel as standalone HTML and LaTeX tables."""
        panel_df = _build_panel(panel_key, dep_label)
        if panel_df is None:
            return None

        notes_text = 'Two-part hurdle model with logit link for the hurdle stage (Panel a) and log link with Tweedie distribution for the severity stage (Panel b). Estimated on landslide events only (pixels with observed landslides). All covariates are standardized (mean 0, SD 1) within the estimation sample before modeling. The GAEZ fixed-effects count reports the number of indicator terms actually estimated in each panel after omitting the reference category and any unsupported zones in that panel-specific sample, so Panel a retains 13 and Panel b retains 16. Standard errors in parentheses. * p < 0.05, ** p < 0.01, *** p < 0.001. Variable definitions: Population (log scale) = log-transformed population density; Extreme Precipitation = annual maximum daily precipitation in mm; Slope = terrain slope in degrees; Road Density = road length per km².'
        
        # Add variable definitions only if include_notes is True
        if include_notes:
            predictors = mortality_data.get(f'{panel_key}_predictors', [])
            non_gaez = [t for t in predictors if not t.startswith('gaez_')]
            
        # LaTeX export
        latex_lines = []
        
        latex_lines.extend([
            '\\centering',
            '\\begin{tabular}{lr}',
            '\\toprule',
            panel_df.columns[0] + ' & ' + panel_df.columns[1] + ' \\\\',
            '\\midrule',
            *_latex_body(panel_df),
            '\\bottomrule',
            '\\end{tabular}',
        ])
        
        if include_notes:
            latex_lines.extend([
                '\\vspace{2pt}',
                '\\begin{minipage}{0.92\\linewidth}',
                f'\\footnotesize Notes: {notes_text}',
                '\\end{minipage}',
            ])
        
        latex_lines.append('')
        tex_text = '\n'.join(latex_lines)

        # HTML export
        html_lines = [
            '<table class="table table-sm table-striped">',
            '  <thead>',
            f'    <tr><th>{panel_df.columns[0]}</th><th>{panel_df.columns[1]}</th></tr>',
            '  </thead>',
            '  <tbody>',
            *[f'    {line.strip()}' for line in _html_rows(panel_df)],
            '  </tbody>',
            '</table>',
        ]
        
        if include_notes:
            html_lines.append(f'<p class="table-notes">Notes: {notes_text}</p>')
        
        html_lines.append('')
        html_text = '\n'.join(html_lines)

        return {'latex': tex_text, 'html': html_text}

    hurdle_output = _export_single_panel('hurdle', 'Pr(Mortality > 0)', include_notes=False)
    severity_output = _export_single_panel('severity', 'Mortality Count', include_notes=True)

    if hurdle_output is None or severity_output is None:
        hb.log('WARNING: Could not build mortality model tables.')
        return p

    output_dir = getattr(p, 'visualizations_dir', None) or getattr(p, 'dissertation_dir', None)
    if output_dir is None:
        hb.log('WARNING: No output directory available for mortality table export.')
        return p

    # Export Panel A (Hurdle)
    output_html_path_a = os.path.join(output_dir, 'mortality_model_publication_table_a.html')
    with open(output_html_path_a, 'w') as f:
        f.write(hurdle_output['html'])

    output_tex_path_a = os.path.join(output_dir, 'mortality_model_publication_table_a.tex')
    with open(output_tex_path_a, 'w') as f:
        f.write(hurdle_output['latex'])

    hb.log(f'✓ Saved mortality hurdle table: {output_html_path_a}')
    hb.log(f'✓ Saved mortality hurdle table: {output_tex_path_a}')

    # Export Panel B (Severity)
    output_html_path_b = os.path.join(output_dir, 'mortality_model_publication_table_b.html')
    with open(output_html_path_b, 'w') as f:
        f.write(severity_output['html'])

    output_tex_path_b = os.path.join(output_dir, 'mortality_model_publication_table_b.tex')
    with open(output_tex_path_b, 'w') as f:
        f.write(severity_output['latex'])

    hb.log(f'✓ Saved mortality severity table: {output_html_path_b}')
    hb.log(f'✓ Saved mortality severity table: {output_tex_path_b}')

    return p


def export_mortality_representative_cases_table(p):
    """Export a combined representative-case table for the mortality model.

    The table varies one predictor at a time across low/high values while holding
    the other predictors at their sample means and the GAEZ fixed effects at the
    reference category. It reports the hurdle probability, the positive-mortality
    severity prediction, and the combined conditional expected mortality.
    """

    candidates = []
    if hasattr(p, 'estimate_hazard_model_dir') and p.estimate_hazard_model_dir:
        candidates.append(p.estimate_hazard_model_dir)
    if hasattr(p, 'project_dir') and p.project_dir:
        candidates.append(os.path.join(p.project_dir, 'intermediate', 'estimate_hazard_model'))

    model_dir = None
    for candidate in candidates:
        if not candidate:
            continue
        stage2_path = os.path.join(candidate, 'severity_model.json')
        if os.path.exists(stage2_path):
            model_dir = candidate
            break

    if model_dir is None:
        hb.log('WARNING: No estimate_hazard_model directory found for mortality representative-case export.')
        return p

    stage2_path = os.path.join(model_dir, 'severity_model.json')
    with open(stage2_path, 'r') as f:
        mortality_data = json.load(f)

    table_candidates = []
    if hasattr(p, 'build_estimation_table_dir') and p.build_estimation_table_dir:
        table_candidates.append(os.path.join(p.build_estimation_table_dir, 'estimation_table.parquet'))
    if hasattr(p, 'project_dir') and p.project_dir:
        table_candidates.append(os.path.join(p.project_dir, 'intermediate', 'build_estimation_table', 'estimation_table.parquet'))

    table_path = next((path for path in table_candidates if os.path.exists(path)), None)
    if table_path is None:
        hb.log('WARNING: Could not find estimation table for representative-case mortality export.')
        return p

    df = pd.read_parquet(table_path)
    required_cols = ['landslide', 'mortality', 'population', 'rain_max_daily', 'slope_degree', 'road_density']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        hb.log(f'WARNING: Representative-case export skipped; missing columns: {missing_cols}')
        return p

    df_rep = df[(df['landslide'] == 1) & (df['mortality'] >= 0) & (df['population'] > 0)].copy()
    if df_rep.empty:
        hb.log('WARNING: Representative-case export skipped; no landslide-positive rows found.')
        return p

    df_rep['population_log1p'] = np.log1p(df_rep['population'].clip(lower=0))

    quantile_specs = [
        ('population_log1p', 'Population', 'persons/pixel', True),
        ('rain_max_daily', 'Extreme Precipitation', 'mm', False),
        ('slope_degree', 'Slope', 'degrees', False),
        ('road_density', 'Road Density', 'km/km²', False),
    ]

    scaler = mortality_data.get('scaler', {})
    hurdle_coefs = mortality_data.get('hurdle_coefficients', {})
    severity_coefs = mortality_data.get('severity_coefficients', {})

    def _zscore(raw_value, key):
        info = scaler.get(key, {})
        mean_value = float(info.get('mean', 0.0))
        std_value = float(info.get('std', 1.0)) if float(info.get('std', 1.0)) != 0 else 1.0
        return (float(raw_value) - mean_value) / std_value

    def _linear_predict(coefs, focal_key=None, focal_value=0.0):
        eta = float(coefs.get('const', 0.0))
        for term, coef in coefs.items():
            if term == 'const':
                continue
            if term.startswith('gaez_'):
                continue
            if term == focal_key:
                eta += float(coef) * float(focal_value)
            else:
                eta += 0.0
        return eta

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    rows = []
    for source_key, label, unit_label, is_log1p in quantile_specs:
        series = pd.to_numeric(df_rep[source_key], errors='coerce').dropna()
        low_raw = float(series.quantile(0.25))
        high_raw = float(series.quantile(0.75))

        if is_log1p:
            low_display = f'{np.expm1(low_raw):,.0f}'
            high_display = f'{np.expm1(high_raw):,.0f}'
            model_key = 'population_log1p'
        else:
            low_display = f'{low_raw:,.3f}'
            high_display = f'{high_raw:,.3f}'
            model_key = source_key

        low_std = _zscore(low_raw, model_key)
        high_std = _zscore(high_raw, model_key)

        hurdle_low = _sigmoid(_linear_predict(hurdle_coefs, focal_key=f'std_{model_key}', focal_value=low_std))
        hurdle_high = _sigmoid(_linear_predict(hurdle_coefs, focal_key=f'std_{model_key}', focal_value=high_std))
        sev_low = float(np.exp(_linear_predict(severity_coefs, focal_key=f'std_{model_key}', focal_value=low_std)))
        sev_high = float(np.exp(_linear_predict(severity_coefs, focal_key=f'std_{model_key}', focal_value=high_std)))
        exp_low = hurdle_low * sev_low
        exp_high = hurdle_high * sev_high

        rows.append({
            'Variable': label,
            'Units': unit_label,
            'Low value (25th pct.)': low_display,
            'High value (75th pct.)': high_display,
            'Pr(any fatalities) - low': f'{hurdle_low:.3f}',
            'Pr(any fatalities) - high': f'{hurdle_high:.3f}',
            'E[fatalities | positive] - low': f'{sev_low:.3f}',
            'E[fatalities | positive] - high': f'{sev_high:.3f}',
            'E[fatalities | landslide] - low': f'{exp_low:.3f}',
            'E[fatalities | landslide] - high': f'{exp_high:.3f}',
            'Change in expected mortality': f'{(exp_high - exp_low):.3f}',
        })

    table_df = pd.DataFrame(rows)
    notes_text = (
        'Representative-case predictions from the mortality model. Low/high values are the 25th and 75th percentiles '
        'of the landslide-positive estimation sample. Other covariates are held at their sample means and the GAEZ '
        'fixed effects are set to the reference category. Population is shown in persons per pixel, but the model uses '
        'log1p(population) internally. The combined conditional expected mortality equals the hurdle probability times '
        'the positive-mortality severity prediction.'
    )

    output_dir = getattr(p, 'visualizations_dir', None) or getattr(p, 'dissertation_dir', None)
    if output_dir is None:
        hb.log('WARNING: No output directory available for mortality representative-case export.')
        return p

    _save_table_multiple_formats(
        table_df,
        os.path.join(output_dir, 'mortality_model_representative_cases'),
        index=False,
        md_text='\n'.join([
            'Table: Mortality Model Representative-Case Predictions {#tbl-mortality-representative-cases}',
            '',
            table_df.to_markdown(index=False),
            '',
            '`\\footnotesize`{=latex}',
            '<div class="tablenotes">',
            f'Notes: {notes_text}',
            '</div>',
            '`\\normalsize`{=latex}',
            '',
        ]),
        tex_text='\n'.join([
            '\\begin{table}[htbp]',
            '\\centering',
            '\\caption{Mortality Model Representative-Case Predictions}',
            table_df.to_latex(index=False, escape=True),
            '\\vspace{2pt}',
            '\\begin{minipage}{0.92\\linewidth}',
            f'\\footnotesize Notes: {notes_text}',
            '\\end{minipage}',
            '\\end{table}',
            '',
        ]),
    )

    hb.log(f'✓ Saved mortality representative-case table: {os.path.join(output_dir, "mortality_model_representative_cases.html")}')
    hb.log(f'✓ Saved mortality representative-case table: {os.path.join(output_dir, "mortality_model_representative_cases.tex")}')
    return p


def export_regression_tables(p):
    """Export hazard, hurdle, and Tweedie regression tables from saved JSONs."""

    candidates = []
    if hasattr(p, 'estimate_hazard_model_dir') and p.estimate_hazard_model_dir:
        candidates.append(p.estimate_hazard_model_dir)
    if hasattr(p, 'project_dir') and p.project_dir:
        candidates.append(os.path.join(p.project_dir, 'intermediate', 'estimate_hazard_model'))

    model_dir = None
    for candidate in candidates:
        if not candidate:
            continue
        if any(os.path.exists(os.path.join(candidate, name)) for name in ['hazard_model.json', 'mortality_model.json']):
            model_dir = candidate
            break

    if model_dir is None:
        hb.log('WARNING: No estimate_hazard_model directory found for regression table export.')
        return p

    def _load_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    def _term_label(term):
        if term == 'const':
            return 'Intercept'
        pretty = {
            'slope_degree': 'Slope',
            'twi': 'Topographic Wetness Index',
            'road_density': 'Road Density',
            'dist_to_fault_km': 'Distance to Fault',
            'population': 'Population',
            'rain_max_daily': 'Extreme Precipitation',
            'forest_share': 'Forest Cover Share',
            'othernat_share': 'Other Natural Veg. Share',
        }
        if term.startswith('std_'):
            return _term_label(term[4:])
        if term.startswith('gaez_'):
            return f'GAEZ {term.split("_", 1)[1]}'
        return pretty.get(term, term.replace('_', ' ').title())

    def _export_stage(model_data, model_label, stage_key, table_title, dep_label, include_terms=None):
        coef_key = f'{stage_key}_coefficients' if stage_key else 'coefficients'
        pval_key = f'{stage_key}_pvalues' if stage_key else 'pvalues'
        se_key = f'{stage_key}_std_errors' if stage_key else 'std_errors'

        def _get_value(*keys):
            for key in keys:
                if key is None:
                    continue
                value = model_data.get(key)
                if value is not None:
                    return value
            return None

        coefs = model_data.get(coef_key) or model_data.get('coefficients', {})
        pvals = model_data.get(pval_key) or model_data.get('pvalues', {})
        ses = model_data.get(se_key) or model_data.get('std_errors', {})

        coef_rows = []
        for term, coef in coefs.items():
            pval = pvals.get(term, np.nan)
            stars = '***' if pd.notna(pval) and pval < 0.001 else ('**' if pd.notna(pval) and pval < 0.01 else ('*' if pd.notna(pval) and pval < 0.05 else ''))
            coef_rows.append({
                'model': model_label,
                'term': term,
                'term_label': _term_label(term),
                'coefficient': float(coef),
                'std_error': float(ses.get(term, np.nan)) if pd.notna(ses.get(term, np.nan)) else np.nan,
                'p_value': float(pval) if pd.notna(pval) else np.nan,
                'signif': stars,
            })

        if not coef_rows:
            return False

        df_coef = pd.DataFrame(coef_rows)
        df_coef['_sort'] = df_coef['term'].map({'const': -1}).fillna(0)
        df_coef = df_coef.sort_values(by=['_sort', 'term']).drop(columns=['_sort']).reset_index(drop=True)

        _save_table_multiple_formats(
            df_coef,
            os.path.join(p.visualizations_dir, f'{model_label}_model_coefficients'),
            index=False,
        )

        summary = {
            'model': model_label,
            'model_key': model_data.get('model_key'),
            'family': model_data.get('family'),
            'model_type': model_data.get('model_type'),
            'dep_var': model_data.get('dep_var'),
            'converged': _get_value(f'{stage_key}_converged', 'converged'),
            'n_obs': _get_value(f'{stage_key}_n_obs', 'n_obs'),
            'n_events': _get_value(f'{stage_key}_n_events', 'n_events'),
            'n_positive': _get_value(f'{stage_key}_n_positive', 'n_positive'),
            'n_mortality_gt0': _get_value(f'{stage_key}_n_mortality_gt0', 'n_mortality_gt0'),
            'llf': _get_value(f'{stage_key}_llf', 'llf'),
            'll_null': _get_value(f'{stage_key}_llnull', f'{stage_key}_ll_null', 'll_null', 'llnull'),
            'aic': _get_value(f'{stage_key}_aic', 'aic'),
            'gaez_ref_zone': model_data.get('gaez_ref_zone'),
            'ipw_cap': model_data.get('ipw_cap'),
            'severity_deviance': model_data.get('severity_deviance'),
        }
        summary['dep_mean'] = _get_value(
            f'{stage_key}_dep_mean',
            f'{stage_key}_dep_var_mean',
            'dep_mean',
            'dep_var_mean',
        )
        summary['pseudo_r2_mcfadden'] = _get_value(
            f'{stage_key}_pseudo_r2_mcfadden',
            f'{stage_key}_pseudo_r2',
            'pseudo_r2_mcfadden',
            'pseudo_r2',
        )
        df_summary = pd.DataFrame([summary])
        _save_table_multiple_formats(
            df_summary,
            os.path.join(p.visualizations_dir, f'{model_label}_model_summary'),
            index=False,
        )

        metadata = {
            'model': model_label,
            'dep_var': summary.get('dep_var'),
            'n_obs': int(summary['n_obs']) if pd.notna(summary.get('n_obs')) else None,
            'n_events': int(summary['n_events']) if pd.notna(summary.get('n_events')) else None,
            'n_positive': int(summary['n_positive']) if pd.notna(summary.get('n_positive')) else None,
            'llf': float(summary['llf']) if pd.notna(summary.get('llf')) else None,
            'aic': float(summary['aic']) if pd.notna(summary.get('aic')) else None,
            'ipw_cap': float(summary['ipw_cap']) if pd.notna(summary.get('ipw_cap')) else None,
            'has_fixed_effects': any(str(term).startswith('gaez_') for term in coefs),
        }
        if 'll_null' in model_data and model_data.get('ll_null') is not None and metadata['llf'] is not None:
            try:
                ll_null = float(model_data.get('ll_null'))
                metadata['pseudo_r2_mcfadden'] = 1.0 - (metadata['llf'] / ll_null) if ll_null != 0 else None
            except Exception:
                metadata['pseudo_r2_mcfadden'] = model_data.get('pseudo_r2_mcfadden')
        else:
            metadata['pseudo_r2_mcfadden'] = model_data.get('pseudo_r2_mcfadden')
        metadata['dep_mean'] = float(model_data.get('dep_mean')) if model_data.get('dep_mean') is not None else (
            float(model_data.get('dep_var_mean')) if model_data.get('dep_var_mean') is not None else None
        )
        metadata_path = os.path.join(p.visualizations_dir, f'{model_label}_model_metadata.json')
        with open(metadata_path, 'w') as mf:
            json.dump(metadata, mf, indent=2)

        fmt_cfg = {
            'title': table_title,
            'table_id': f'tbl-{model_label}-publication',
            'dep_label': dep_label,
            'include_terms': include_terms,
            'variable_notes': {
                'slope_degree': 'terrain slope in degrees derived from the DEM and standardized within the estimation sample',
                'roughness': 'local terrain roughness index capturing fine-scale slope variability',
                'tpi': 'relative position in the surrounding terrain, with positive values indicating ridges and negative values indicating valleys',
                'elev_stdev': 'within-neighborhood elevation standard deviation, used as a local terrain heterogeneity measure',
                'road_density': 'road length per square kilometer from the road network raster',
                'dist_to_fault_km': 'distance to the nearest active fault trace in kilometers',
                'population': 'log1p population or population density, standardized within the estimation sample',
                'rain_max_daily': 'annual maximum daily precipitation derived from ERA5-Land and standardized within the estimation sample',
                'forest_share': 'share of the pixel classified as forest cover in the corresponding raster year',
                'othernat_share': 'share of the pixel classified as other natural vegetation in the corresponding raster year',
                'deforestation_forest_share_1yr': 'one-year trailing change in forest cover share used to approximate recent clearing pressure',
                'deforestation_forest_share_3yr': 'three-year trailing change in forest cover share used to smooth short-run clearing pressure',
            },
            'term_labels': {
                'const': 'Intercept',
                'slope_degree': 'Slope',
                'roughness': 'Roughness',
                'tpi': 'Topographic Position Index',
                'elev_stdev': 'Elevation Variability',
                'road_density': 'Road Density',
                'dist_to_fault_km': 'Distance to Fault',
                'deforestation_forest_share_1yr': 'Deforestation (1yr)',
                'deforestation_forest_share_3yr': 'Deforestation (3yr)',
                'std_population_log1p': 'Population',
                'std_rain_max_daily': 'Extreme Precipitation',
                'std_slope_degree': 'Slope',
                'std_road_density': 'Road Density',
                'population': 'Population',
                'std_population': 'Population',
                'rain_max_daily': 'Extreme Precipitation',
                'std_rain_max_daily': 'Extreme Precipitation',
                'forest_share': 'Forest Cover Share',
                'std_forest_share': 'Forest Cover Share',
                'othernat_share': 'Other Natural Veg. Share',
                'std_othernat_share': 'Other Natural Veg. Share',
            },
        }
        pub_df, pub_md, pub_tex = _format_regression_publication_table(
            model_label=model_label,
            coef_df=df_coef,
            metadata=metadata,
            config=fmt_cfg,
        )
        _save_table_multiple_formats(
            pub_df,
            os.path.join(p.visualizations_dir, f'{model_label}_model_publication_table'),
            index=False,
            md_text=pub_md,
            tex_text=pub_tex,
        )
        return True

    hazard_model_path = os.path.join(model_dir, 'hazard_model.json')
    stage2_path = os.path.join(model_dir, 'severity_model.json')
    if not os.path.exists(stage2_path):
        stage2_path = os.path.join(model_dir, 'mortality_model.json')

    any_exported = False

    if os.path.exists(hazard_model_path):
        hazard_data = _load_json(hazard_model_path)
        any_exported = _export_stage(
            hazard_data,
            'hazard',
            stage_key='',
            table_title='Hazard Model of Landslide Occurrence',
            dep_label='Pr(Landslide)',
            include_terms=['const', 'slope_degree', 'roughness', 'tpi', 'elev_stdev', 'road_density', 'dist_to_fault_km', 'deforestation_forest_share_1yr', 'deforestation_forest_share_3yr'],
        ) or any_exported
    else:
        hb.log(f'WARNING: Missing model file, skipping: {hazard_model_path}')

    if os.path.exists(stage2_path):
        stage2_data = _load_json(stage2_path)
        if 'hurdle_coefficients' in stage2_data:
            any_exported = _export_stage(
                stage2_data,
                'mortality_hurdle',
                stage_key='hurdle',
                table_title='Mortality Hurdle Model (Pr[mortality > 0])',
                dep_label='Pr(Mortality > 0)',
                include_terms=['const', 'std_population', 'std_rain_max_daily', 'std_forest_share', 'std_othernat_share'],
            ) or any_exported
        if 'severity_coefficients' in stage2_data:
            any_exported = _export_stage(
                stage2_data,
                'mortality_severity',
                stage_key='severity',
                table_title='Mortality Severity Model (Tweedie)',
                dep_label='Mortality Count',
                include_terms=['const', 'std_population', 'std_rain_max_daily', 'std_forest_share', 'std_othernat_share'],
            ) or any_exported
        if 'coefficients' in stage2_data and 'severity_coefficients' not in stage2_data:
            any_exported = _export_stage(
                stage2_data,
                'mortality_model',
                stage_key='',
                table_title='Mortality Model of Landslide Deaths',
                dep_label='Mortality Count',
                include_terms=['const', 'population', 'rain_max_daily', 'forest_share', 'othernat_share'],
            ) or any_exported
        any_exported = export_mortality_representative_cases_table(p) or any_exported
    else:
        hb.log(f'WARNING: Missing model file, skipping: {stage2_path}')

    if not any_exported:
        hb.log('WARNING: No regression tables were exported.')

    return p

def sync_to_dissertation(p):
    source_dir = p.visualizations_dir
    destination_dir = p.dissertation_dir
    # Copy the entire directory
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)

def visualizations(p):
    """Run post-model plotting and table exports in one convenience task."""
    return p
