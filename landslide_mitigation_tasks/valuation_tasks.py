"""
valuation_tasks.py
"""
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
import pygeoprocessing as pygeo


def tile_zones(p):
    """Generate tile zones for parallel prediction, filtered to
    land-containing tiles only. Tiles are defined on the EASE-Grid
    reference's exact dimensions/transform via p.gaez_path 
    """
    if not p.run_this:
        return p

    blocks_list_path = os.path.join(p.cur_dir, 'blocks_list.csv')

    if os.path.exists(blocks_list_path):
        p.L.info('Blocks list already exists, loading from file...')
        blocks_df = pd.read_csv(blocks_list_path, header=None)
        blocks_df.columns = ['col_offset', 'row_offset', 'n_cols', 'n_rows']
        blocks_list = blocks_df.values.tolist()
        p.L.info(f'Loaded {len(blocks_list)} tiles from existing blocks_list.csv')
    else:
        p.L.info('Creating tile list from GAEZ zones raster (real land data, '
               'on the exact EASE-Grid reference grid)...')

        ds = gdal.Open(p.gaez_path)
        n_cols = ds.RasterXSize
        n_rows = ds.RasterYSize
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()

        p.tile_size = getattr(p, 'processing_resolution', 2000)

        blocks_list = []
        for row_offset in range(0, n_rows, p.tile_size):
            for col_offset in range(0, n_cols, p.tile_size):
                actual_n_cols = min(p.tile_size, n_cols - col_offset)
                actual_n_rows = min(p.tile_size, n_rows - row_offset)

                tile = band.ReadAsArray(col_offset, row_offset, actual_n_cols, actual_n_rows)
                land_mask = np.isfinite(tile)
                if nodata is not None:
                    land_mask &= (tile != nodata)

                if land_mask.sum() > 0:
                    blocks_list.append([col_offset, row_offset, actual_n_cols, actual_n_rows])

        ds = None

        blocks_df = pd.DataFrame(blocks_list, columns=['col_offset', 'row_offset', 'n_cols', 'n_rows'])
        blocks_df.to_csv(blocks_list_path, index=False, header=False)
        p.L.info(f'Created {len(blocks_list)} land tiles (filtered ocean tiles)')
        p.L.info(f'Blocks list saved to: {blocks_list_path}')

    p.iterator_replacements = {
        'tile_col_offset': [block[0] for block in blocks_list],
        'tile_row_offset': [block[1] for block in blocks_list],
        'tile_n_cols': [block[2] for block in blocks_list],
        'tile_n_rows': [block[3] for block in blocks_list],
        'cur_dir_parent_dir': [
            os.path.join(p.cur_dir, f'{block[1]}_{block[0]}')
            for block in blocks_list
        ]
    }
    p.L.info(f'Set up iterator replacements for {len(blocks_list)} tiles.')
    return p


def predict_landslides_scenarios(p):
    """Per tile, per scenario: apply the calibrated logistic
    (alpha_corrected + beta_si*SI + beta_rain*rain) to produce a hazard
    probability GeoTIFF. Written to disk per tile x scenario.
    """
    if not p.run_this:
        return p

    hazard_model_coefficients_path = os.path.join(p.modeling_dir, 'hazard_model_coefficients.json')
    with open(hazard_model_coefficients_path) as f:
        coef = json.load(f)
    alpha = coef['alpha_corrected']
    beta_si = coef['beta_si']
    beta_rain = coef['beta_rain']

    ref_info = pygeo.get_raster_info(p.ease_grid_reference_path)
    gt = ref_info['geotransform']
    proj = ref_info['projection_wkt']

    col_off, row_off = p.tile_col_offset, p.tile_row_offset
    n_cols, n_rows = p.tile_n_cols, p.tile_n_rows
    tile_gt = (
        gt[0] + col_off * gt[1], gt[1], 0,
        gt[3] + row_off * gt[5], 0, gt[5],
    )

    # Rain: NOT scenario-varying, same prediction-year raster for both.
    prediction_year = p.prediction_years[0]  # NOTE: assumes single prediction year
    rain_path = os.path.join(
        p.input_data_dir, 'era5_land', f'era5_max_daily_mm_{prediction_year}.tif'
    )
    rain_ds = gdal.Open(rain_path)
    rain_band = rain_ds.GetRasterBand(1)
    rain_nodata = rain_band.GetNoDataValue()
    rain_tile = rain_band.ReadAsArray(col_off, row_off, n_cols, n_rows)
    rain_ds = None

    for scenario_name, si_paths_by_year in p.si_paths.items():
        si_path = si_paths_by_year.get(prediction_year)
        if si_path is None:
            p.L.warning(f'{scenario_name}: no SI for prediction year {prediction_year}, skipping tile.')
            continue

        si_ds = gdal.Open(si_path)
        si_band = si_ds.GetRasterBand(1)
        si_nodata = si_band.GetNoDataValue()
        si_tile = si_band.ReadAsArray(col_off, row_off, n_cols, n_rows)
        si_ds = None

        valid = np.ones(si_tile.shape, dtype=bool)
        if si_nodata is not None:
            valid &= (si_tile != si_nodata)
        if rain_nodata is not None:
            valid &= (rain_tile != rain_nodata)

        logodds = alpha + beta_si * si_tile + beta_rain * rain_tile
        prob = 1 / (1 + np.exp(-logodds))
        prob_out = np.where(valid, prob, -9999.0).astype(np.float32)

        out_path = os.path.join(p.cur_dir, f'hazard_prob_{scenario_name}_{prediction_year}.tif')
        driver = gdal.GetDriverByName('GTiff')
        ds_out = driver.Create(
            out_path, n_cols, n_rows, 1, gdal.GDT_Float32,
            options=['TILED=YES', 'COMPRESS=LZW'],
        )
        ds_out.SetGeoTransform(tile_gt)
        ds_out.SetProjection(proj)
        band_out = ds_out.GetRasterBand(1)
        band_out.WriteArray(prob_out)
        band_out.SetNoDataValue(-9999.0)
        ds_out = None

        p.L.info(f'Tile ({row_off},{col_off}) {scenario_name}: {out_path}')

    return p



def predict_mortality_scenarios(p):
    """
    Per tile, per scenario: predict expected deaths per pixel by combining:
 
        P(landslide | hazard model)
        *
        P(fatality > 0 | landslide, severity covariates)
        *
        E(fatalities | fatal, severity covariates)
 
    Part B expectation uses Duan's smearing correction.
    Writes mortality GeoTIFFs per tile x scenario.
    """
    if not p.run_this:
        return p
 
    severity_path = os.path.join(p.modeling_dir, 'severity_model_coefficients.json')
    with open(severity_path) as f:
        severity = json.load(f)
 
    part_a = severity['part_a_params']
    part_b = severity['part_b_params']
    smearing = severity['smearing_factor']
 
    ref_info = pygeo.get_raster_info(p.ease_grid_reference_path)
    gt = ref_info['geotransform']
    proj = ref_info['projection_wkt']
 
    col_off = p.tile_col_offset
    row_off = p.tile_row_offset
    n_cols = p.tile_n_cols
    n_rows = p.tile_n_rows
 
    tile_gt = (
        gt[0] + col_off * gt[1], gt[1], 0,
        gt[3] + row_off * gt[5], 0, gt[5],
    )
 
    def read_tile(path, already_tiled=False):
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        xsize, ysize = band.XSize, band.YSize
 
        if already_tiled or (xsize == n_cols and ysize == n_rows):
            arr = band.ReadAsArray()
        else:
            arr = band.ReadAsArray(col_off, row_off, n_cols, n_rows)
 
        ds = None
        return arr, nodata
 
    prediction_year = p.prediction_years[0]
 
    population_path = os.path.join(
        p.input_data_dir, 'landscan_1km', f'landscan_{prediction_year}_1km.tif'
    )
    rain_path = os.path.join(
        p.input_data_dir, 'era5_land', f'era5_max_daily_mm_{prediction_year}.tif'
    )
    slope_path = p.slope_path
    road_path = p.road_density_path

    population, pop_nd = read_tile(population_path)
    rain, rain_nd = read_tile(rain_path)
    slope, slope_nd = read_tile(slope_path)
    road, road_nd = read_tile(road_path)
 
    population_log1p = np.log1p(np.maximum(population, 0))
 
    valid = np.ones(population.shape, dtype=bool)
    for arr, nd in [(population, pop_nd), (rain, rain_nd), (slope, slope_nd), (road, road_nd)]:
        if nd is not None:
            valid &= (arr != nd)
 
    logit_a = (
        part_a['Intercept']
        + part_a['population_log1p'] * population_log1p
        + part_a['rain_max_daily'] * rain
        + part_a['slope_degrees'] * slope
        + part_a['road_density'] * road
    )
    p_fatal = 1 / (1 + np.exp(-logit_a))
 
    log_fatalities = (
        part_b['Intercept']
        + part_b['population_log1p'] * population_log1p
        + part_b['rain_max_daily'] * rain
        + part_b['slope_degrees'] * slope
        + part_b['road_density'] * road
    )
    expected_fatalities_if_fatal = np.exp(log_fatalities) * smearing
 
    severity_expectation = p_fatal * expected_fatalities_if_fatal
 
    for scenario_name in p.si_paths.keys():
        hazard_dir = os.path.join(os.path.dirname(p.cur_dir), 'predict_landslides_scenarios')
        hazard_path = os.path.join(hazard_dir, f'hazard_prob_{scenario_name}_{prediction_year}.tif')
 
        if not os.path.exists(hazard_path):
            p.L.warning(f'Missing hazard raster: {hazard_path}')
            continue
 
        hazard, hazard_nd = read_tile(hazard_path, already_tiled=True)
 
        valid_hazard = valid.copy()
        if hazard_nd is not None:
            valid_hazard &= (hazard != hazard_nd)
 
        deaths = hazard * severity_expectation
        deaths_out = np.where(valid_hazard, deaths, -9999.0).astype(np.float32)
 
        out_path = os.path.join(p.cur_dir, f'expected_deaths_{scenario_name}_{prediction_year}.tif')
        driver = gdal.GetDriverByName('GTiff')
        ds_out = driver.Create(
            out_path, n_cols, n_rows, 1, gdal.GDT_Float32,
            options=['TILED=YES', 'COMPRESS=LZW'],
        )
        ds_out.SetGeoTransform(tile_gt)
        ds_out.SetProjection(proj)
        band = ds_out.GetRasterBand(1)
        band.WriteArray(deaths_out)
        band.SetNoDataValue(-9999.0)
        ds_out = None
 
        p.L.info(f'Tile ({row_off},{col_off}) {scenario_name}: {out_path}')
 
    return p
 
 
def stitch_tiles(p):
    """
    Stitch tile-level hazard and mortality predictions into global rasters.
    """
    if not p.run_this:
        return p
 
    blocks_list_path = os.path.join(p.tile_zones_dir, 'blocks_list.csv')
    blocks_df = pd.read_csv(blocks_list_path, header=None)
    blocks_df.columns = ['col_offset', 'row_offset', 'n_cols', 'n_rows']
    blocks_list = blocks_df.values.tolist()
 
    ref_ds = gdal.Open(p.gaez_path)
    n_cols_full = ref_ds.RasterXSize
    n_rows_full = ref_ds.RasterYSize
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    ref_ds = None
 
    driver = gdal.GetDriverByName('GTiff')
    NODATA = -9999.0
 
    outputs = []
    for scenario_name in p.si_paths.keys():
        outputs.extend([
            {
                'name': f'hazard_{scenario_name}',
                'tile_subdir': 'predict_landslides_scenarios',
                'tile_filename': f'hazard_prob_{scenario_name}_{{year}}.tif',
                'global_filename': f'hazard_prob_{scenario_name}_{{year}}.tif',
            },
            {
                'name': f'mortality_{scenario_name}',
                'tile_subdir': 'predict_mortality_scenarios',
                'tile_filename': f'expected_deaths_{scenario_name}_{{year}}.tif',
                'global_filename': f'expected_deaths_{scenario_name}_{{year}}.tif',
            },
        ])
 
    for year in p.prediction_years:
        for spec in outputs:
            out_path = os.path.join(p.cur_dir, spec['global_filename'].format(year=year))
 
            if os.path.exists(out_path) and not getattr(p, 'force_run', False):
                p.L.info(f'Skipping existing: {out_path}')
                continue
            if os.path.exists(out_path):
                os.remove(out_path)
 
            p.L.info(f'Stitching {spec["name"]} {year}')
 
            ds_out = driver.Create(
                out_path, n_cols_full, n_rows_full, 1, gdal.GDT_Float32,
                options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
            )
            ds_out.SetGeoTransform(gt)
            ds_out.SetProjection(proj)
            band_out = ds_out.GetRasterBand(1)
            band_out.SetNoDataValue(NODATA)
            band_out.Fill(NODATA)
 
            written, missing = 0, 0
            for block in blocks_list:
                col_off, row_off, n_c, n_r = [int(x) for x in block]
                tile_dir = os.path.join(p.tile_zones_dir, f'{row_off}_{col_off}')
                tile_path = os.path.join(
                    tile_dir, spec['tile_subdir'], spec['tile_filename'].format(year=year)
                )
 
                if not os.path.exists(tile_path):
                    missing += 1
                    continue
 
                ds_tile = gdal.Open(tile_path)
                arr = ds_tile.GetRasterBand(1).ReadAsArray().astype(np.float32)
                ds_tile = None
 
                arr = np.where(np.isnan(arr), NODATA, arr)
                band_out.WriteArray(arr, col_off, row_off)
                written += 1
 
            band_out.FlushCache()
            ds_out = None
 
            p.L.info(f'  Wrote {written} tiles')
            if missing:
                p.L.info(f'  Missing {missing} tiles')
 
    p.L.info('Stitching complete.')
    return p




EPA_VSL_BASE_2008USD = 7.9e6  # EPA's officially-adopted Guidelines VSL, in 2008$
EPA_VSL_INCOME_ELASTICITY = 0.4  # EPA's adopted elasticity for updating the VSL over time
                                  # (distinct from the cross-country benefit-transfer step below)

def _compute_epa_vsl_usa(p, target_year):
    """Update EPA's officially-adopted VSL guidance value ($7.9 million,
    2008$) to target_year dollars, following EPA's stated methodology:
    CPI inflation adjustment x real (inflation-adjusted) US GDP per capita
    growth, raised to EPA's income elasticity of 0.4. See EPA's 2016
    "Valuing mortality risk reductions for policy: a meta-analytic
    approach" white paper (Office of Policy, National Center for
    Environmental Economics) sec 2.1 and 6.3, and the 2010 Guidelines for
    Preparing Economic Analyses.

    This produced $9,876,237 for target_year=2019 when derived.
    """
    cpi_path = p.get_path(os.path.join('socioeconomic', 'fred_cpiaucsl', 'CPIAUCSL.csv'))
    cpi = pd.read_csv(cpi_path)
    cpi['observation_date'] = pd.to_datetime(cpi['observation_date'])
    cpi['year'] = cpi['observation_date'].dt.year
    cpi_annual = cpi.groupby('year')['CPIAUCSL'].mean()
    cpi_ratio = cpi_annual.loc[target_year] / cpi_annual.loc[2008]

    gdp_path = p.get_path(os.path.join(
        'socioeconomic', 'worldbank_gdp_pc_constant2015usd',
        'API_NY.GDP.PCAP.KD_DS2_en_csv_v2_329799.csv',
    ))
    gdp = pd.read_csv(gdp_path, skiprows=4)
    us_row = gdp[gdp['Country Code'] == 'USA'].iloc[0]
    gdp_ratio = us_row[str(target_year)] / us_row['2008']

    vsl_usa = EPA_VSL_BASE_2008USD * cpi_ratio * (gdp_ratio ** EPA_VSL_INCOME_ELASTICITY)
    p.L.info(
        f'EPA VSL anchor updated to {target_year}$: ${vsl_usa:,.0f} '
        f'(base ${EPA_VSL_BASE_2008USD:,.0f} [2008$] x CPI ratio {cpi_ratio:.4f} '
        f'x real GDP per capita ratio {gdp_ratio:.4f}^{EPA_VSL_INCOME_ELASTICITY})'
    )
    return vsl_usa


def build_vsl_raster(p):
    """Computes a GDP-adjusted VSL per country from scratch, following the
    life-years-lost benefit-transfer method (GEP-AQ project): 
    each country's VSL is the US anchor VSL scaled by the
    ratio of that country's (GDP per capita x life-years-lost) to the
    US's own (GDP per capita x life-years-lost), where life-years-lost is
    approximated as life expectancy at birth minus median age.

    All three raw inputs (life expectancy, median age, GDP per capita)
    and the US anchor VSL are sourced and cited in
    ~/Files/base_data/socioeconomic/.

    Zones without a direct estimate (no WPP and/or no GDP-PPP data for
    that ISO3 - e.g. Taiwan, Cuba, Venezuela, North Korea, several small
    territories) get a fallback value imputed from peer countries that DO
    have a direct estimate, in two widening tiers: (1) the median VSL
    among direct-estimate countries in the same World Bank region
    (`region_wb`), or (2) the global median across all direct-estimate
    countries if that region has none. No separate external table.
    Everything the fallback needs comes from this same computation.
    `vsl_source` on the exported CSV/GPKG records which tier produced
    each zone's value.
    """
    if p.run_this:
        target_year = p.prediction_years[0]  # NOTE: assumes single prediction year, matches predict_*_scenarios elsewhere
        out_path = os.path.join(p.valuation_dir, f'vsl_usd_{target_year}_1km.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.vsl_raster_path = out_path
            return p

        # ---- 1. Life expectancy + median age (UN World Population Prospects 2024) ----
        wpp_path = p.get_path(os.path.join(
            'socioeconomic', 'un_wpp_demographic_indicators',
            'WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_FULL.xlsx',
        ))
        wpp = pd.read_excel(wpp_path, sheet_name='Estimates', header=16)
        wpp = wpp[(wpp['Type'] == 'Country/Area') & (wpp['Year'] == target_year)]
        wpp = wpp.rename(columns={
            'ISO3 Alpha-code': 'iso3',
            'Life Expectancy at Birth, both sexes (years)': 'life_expectancy',
            'Median Age, as of 1 July (years)': 'median_age',
        })[['iso3', 'life_expectancy', 'median_age']]
        wpp['lll'] = wpp['life_expectancy'] - wpp['median_age']  # life-years-lost proxy
        p.L.info(f'UN WPP {target_year}: {len(wpp)} countries with life expectancy + median age.')

        # ---- 2. GDP per capita, PPP (World Bank) -- cross-country benefit-transfer scaling ----
        gdp_ppp_path = p.get_path(os.path.join(
            'socioeconomic', 'worldbank_gdp_pc_ppp',
            'API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_349731.csv',
        ))
        gdp_ppp = pd.read_csv(gdp_ppp_path, skiprows=4)
        gdp_ppp = gdp_ppp.rename(columns={'Country Code': 'iso3', str(target_year): 'gdp_pc_ppp'})[['iso3', 'gdp_pc_ppp']]

        # World Bank's "Country Code" list mixes in aggregate regions (AFE, ARB, ...)
        # alongside real ISO3 codes - filter using the metadata file's Region field,
        # which is blank for aggregates.
        gdp_meta_path = p.get_path(os.path.join(
            'socioeconomic', 'worldbank_gdp_pc_ppp',
            'Metadata_Country_API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_349731.csv',
        ))
        gdp_meta = pd.read_csv(gdp_meta_path)
        real_countries = set(gdp_meta.loc[gdp_meta['Region'].notna(), 'Country Code'])
        gdp_ppp = gdp_ppp[gdp_ppp['iso3'].isin(real_countries)]
        p.L.info(f'World Bank GDP per capita (PPP) {target_year}: {len(gdp_ppp)} countries.')

        # ---- 3. US anchor VSL, updated to target_year via EPA's own methodology ----
        vsl_usa = _compute_epa_vsl_usa(p, target_year)

        us_wpp = wpp[wpp['iso3'] == 'USA'].iloc[0]
        us_gdp_pc_ppp = gdp_ppp.loc[gdp_ppp['iso3'] == 'USA', 'gdp_pc_ppp'].iloc[0]
        vsl_lll_usa = vsl_usa / us_wpp['lll']
        vsl_lll_gdp_usa = vsl_lll_usa / us_gdp_pc_ppp

        # ---- 4. VSL per country: gdp_pc_ppp x lll x (US's own vsl-per-lll-per-gdp ratio) ----
        vsl_df = wpp.merge(gdp_ppp, on='iso3', how='inner')
        vsl_df['vsl'] = vsl_df['gdp_pc_ppp'] * vsl_df['lll'] * vsl_lll_gdp_usa
        vsl_by_iso3 = vsl_df.set_index('iso3')['vsl'].to_dict()
        p.L.info(f'VSL computed for {len(vsl_by_iso3)} countries (US anchor: ${vsl_usa:,.0f}).')

        # ---- 5. Join to correspondence GPKG via ISO3 (iso3_r250_label) ----
        correspondence_path = p.get_path(os.path.join('cartographic', 'ee', 'ee_r250_correspondence.gpkg'))
        corr = gpd.read_file(correspondence_path)
        if 'iso3_r250_label' not in corr.columns:
            raise KeyError(
                f'iso3_r250_label not found in correspondence GPKG - check '
                f'actual column names: {list(corr.columns)}'
            )

        corr['vsl_usd'] = corr['iso3_r250_label'].map(vsl_by_iso3)
        corr['vsl_source'] = np.where(corr['vsl_usd'].notna(), 'direct', None)

        n_matched = corr['vsl_usd'].notna().sum()
        n_total = len(corr)
        p.L.info(f'VSL match: {n_matched}/{n_total} zones matched via ISO3 (direct estimate).')

        # ---- 5b. Fallback for zones with no direct estimate (no WPP and/or no
        # GDP-PPP data for that ISO3): impute from peer countries' ALREADY-
        # COMPUTED direct VSL estimates:
        #   1. Region median (World Bank region, `region_wb` - already a
        #      column on this same correspondence file), among zones with a
        #      direct estimate in that region.
        #   2. Global median across all zones with a direct estimate, if a
        #      region somehow has zero direct estimates of its own.
        # `vsl_source` records which tier produced each zone's value, so it's
        # always clear in the exported CSV which numbers are estimated vs
        # imputed. ----
        direct = corr[corr['vsl_source'] == 'direct']
        region_medians = direct.groupby('region_wb')['vsl_usd'].median()
        global_median = direct['vsl_usd'].median()

        needs_fallback = corr['vsl_usd'].isna()
        region_fallback = corr.loc[needs_fallback, 'region_wb'].map(region_medians)
        corr.loc[needs_fallback, 'vsl_usd'] = region_fallback
        corr.loc[needs_fallback & corr['vsl_usd'].notna(), 'vsl_source'] = 'region_median_fallback'

        still_needs_fallback = corr['vsl_usd'].isna()
        corr.loc[still_needs_fallback, 'vsl_usd'] = global_median
        corr.loc[still_needs_fallback, 'vsl_source'] = 'global_median_fallback'

        n_region_fallback = (corr['vsl_source'] == 'region_median_fallback').sum()
        n_global_fallback = (corr['vsl_source'] == 'global_median_fallback').sum()
        p.L.info(
            f'VSL fallback: {n_region_fallback} zones via region median, '
            f'{n_global_fallback} via global median (${global_median:,.0f}). '
            f'All {n_total} zones now have a value.'
        )

        # ---- 5c. Export the full computation as a shareable CSV -- every
        # input column plus IDs, so the whole panel can be inspected or
        # shared without needing to re-run the pipeline. ----
        export_cols = [
            'iso3_r250_id', 'iso3_r250_label', 'name_long', 'income_grp', 'region_wb',
            'vsl_usd', 'vsl_source',
        ]
        vsl_export = corr[export_cols].merge(
            vsl_df[['iso3', 'life_expectancy', 'median_age', 'lll', 'gdp_pc_ppp']],
            left_on='iso3_r250_label', right_on='iso3', how='left',
        ).drop(columns='iso3')
        vsl_export = vsl_export[[
            'iso3_r250_id', 'iso3_r250_label', 'name_long', 'income_grp', 'region_wb',
            'life_expectancy', 'median_age', 'lll', 'gdp_pc_ppp',
            'vsl_usd', 'vsl_source',
        ]]
        vsl_csv_path = os.path.join(p.valuation_dir, f'vsl_by_country_{target_year}.csv')
        vsl_export.to_csv(vsl_csv_path, index=False)
        p.L.info(f'VSL panel (all inputs + IDs) exported: {vsl_csv_path}')

        # ---- 6. Reproject to EASE-Grid, rasterize ----
        corr_ease = corr.to_crs('EPSG:6933')
        work_dir = os.path.join(p.valuation_dir, 'vsl_work')
        os.makedirs(work_dir, exist_ok=True)
        temp_gpkg = os.path.join(work_dir, 'corr_ease_vsl.gpkg')
        corr_ease.to_file(temp_gpkg, driver='GPKG')

        ref_info = pygeo.get_raster_info(p.ease_grid_reference_path)
        gt = ref_info['geotransform']
        n_cols, n_rows = ref_info['raster_size']

        driver = gdal.GetDriverByName('GTiff')
        ds_out = driver.Create(
            out_path, n_cols, n_rows, 1, gdal.GDT_Float32,
            options=['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                     'BLOCKXSIZE=256', 'BLOCKYSIZE=256'],
        )
        ds_out.SetGeoTransform(gt)
        ds_out.SetProjection(ref_info['projection_wkt'])
        band_out = ds_out.GetRasterBand(1)
        band_out.SetNoDataValue(-9999.0)
        band_out.Fill(-9999.0)
        ds_out = None

        pygeo.rasterize(
            temp_gpkg, out_path,
            option_list=['ATTRIBUTE=vsl_usd'],
        )

        p.L.info(f'VSL raster: {out_path}')
        p.vsl_raster_path = out_path
    return p


def valuation(p):
    """
    Creates directory for valuation outputs. 
    """
    if p.run_this:
        return p


def compute_avoided_mortality(p):
    if p.run_this:
        for year in p.prediction_years:
            deaths_observed_path = os.path.join(
                p.stitch_tiles_dir, f'expected_deaths_observed_{year}.tif'
            )
            deaths_full_impacts_path = os.path.join(
                p.stitch_tiles_dir, f'expected_deaths_full_impacts_{year}.tif'
            )
            vsl_path = os.path.join(p.valuation_dir, f'vsl_usd_{year}_1km.tif')
 
            avoided_mortality_path = os.path.join(p.valuation_dir, f'avoided_mortality_{year}.tif')
            avoided_mortality_value_path = os.path.join(
                p.valuation_dir, f'avoided_mortality_value_{year}.tif'
            )
 
            if (os.path.exists(avoided_mortality_path)
                    and os.path.exists(avoided_mortality_value_path)
                    and not p.force_run):
                p.L.info(f'{year}: avoided mortality outputs already exist, skipping.')
                continue
 
            for path, label in [(deaths_observed_path, 'expected_deaths_observed'),
                                 (deaths_full_impacts_path, 'expected_deaths_full_impacts'),
                                 (vsl_path, 'vsl_usd')]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f'{label} missing for {year}: {path}')
 
            # ---- avoided_mortality = full_impacts - observed ----
            deaths_obs_nd = pygeo.get_raster_info(deaths_observed_path)['nodata'][0]
            deaths_fi_nd = pygeo.get_raster_info(deaths_full_impacts_path)['nodata'][0]
 
            negative_count = [0]  # mutable closure for the sanity check below
 
            def avoided_op(deaths_obs, deaths_fi):
                valid = np.ones(deaths_obs.shape, dtype=bool)
                if deaths_obs_nd is not None:
                    valid &= (deaths_obs != deaths_obs_nd)
                if deaths_fi_nd is not None:
                    valid &= (deaths_fi != deaths_fi_nd)
 
                avoided = deaths_fi - deaths_obs
 
                negative_count[0] += int(((avoided < 0) & valid).sum())
 
                return np.where(valid, avoided, -9999.0).astype(np.float32)
 
            pygeo.raster_calculator(
                [(deaths_observed_path, 1), (deaths_full_impacts_path, 1)],
                avoided_op, avoided_mortality_path, gdal.GDT_Float32, -9999.0,
                calc_raster_stats=True,
            )
 
            if negative_count[0] > 0:
                p.L.warning(
                    f'{year}: {negative_count[0]} pixels have NEGATIVE avoided '
                    f'mortality (full_impacts predicts FEWER deaths than observed '
                    f'forest cover) -- this should not happen physically. Worth '
                    f'investigating (e.g. severity-model covariates that happen to '
                    f'differ between scenarios, though they should not; or SI clip '
                    f'edge cases) before trusting results near these locations.'
                )
            p.L.info(f'Avoided mortality {year}: {avoided_mortality_path}')
 
            # ---- avoided_mortality_value = avoided_mortality x VSL ----
            avoided_nd = -9999.0
            vsl_nd = pygeo.get_raster_info(vsl_path)['nodata'][0]
 
            def value_op(avoided, vsl):
                valid = np.ones(avoided.shape, dtype=bool)
                if avoided_nd is not None:
                    valid &= (avoided != avoided_nd)
                if vsl_nd is not None:
                    valid &= (vsl != vsl_nd)
                value = avoided * vsl
                return np.where(valid, value, -9999.0).astype(np.float32)
 
            pygeo.raster_calculator(
                [(avoided_mortality_path, 1), (vsl_path, 1)],
                value_op, avoided_mortality_value_path, gdal.GDT_Float32, -9999.0,
                calc_raster_stats=True,
            )
            p.L.info(f'Avoided mortality value {year}: {avoided_mortality_value_path}')
    return p


