"""
submission_tasks.py

Converts the landslide_mitigation project's RAW input rasters (as read
from p.raw_input_data_dir in input_data_tasks.py, before any EASE-Grid
warping) into Pyramidal Cloud-Optimized Geotiffs (POGs), staged at
p.submission_local_dir - by default the local base_data cache's own
submissions/landslide_mitigation/ folder (~/Files/base_data/submissions/
landslide_mitigation), so it doubles as both the local get_path() cache
and the exact folder structure to upload to the TEEMs base_data
submissions drive. Filenames carry no "_pog" suffix - everything in
base_data is assumed to already be a POG (see e.g. ha_per_cell_10sec.tif),
so the format isn't encoded in the name. See
https://justinandrewjohnson.com/earth_economy_devstack/ for the
base_data contribution process and POG spec.

Everything here works on LOCAL COPIES ONLY. Raw sources live on a
GDrive-synced share (p.raw_input_data_dir); make_path_pog writes transient
"copy_*.tif" intermediates next to whatever file it's given, and GDrive
sync does not play well with that, so every source is copied to
p.submission_work_dir before any hazelbean call touches it. Nothing in
this file writes back to GDrive - copy the finished contents of
p.submission_local_dir to the TEEMs submissions folder by hand once
everything here passes.

Three sources (SoilGrids properties, HiHydroSoil Ksat, ERA5 daily
rain) don't land on a supported pyramid resolution and can't be
pogified directly - installed hazelbean's make_path_pog has no
resolution-override argument, so these are pre-resampled with
warp_to_reference against the canonical ha_per_cell_<res>sec.tif match
raster before pogifying. All are intensive quantities (rates,
percentages, densities), not counts, so plain average/mode resampling
is sum-safe - no proportion/multiply trick needed.
"""
import os
import shutil
import hazelbean as hb
from osgeo import gdal
import pygeoprocessing as pygeo

from landslide_mitigation_tasks.landslide_mitigation_utils import warp_to_reference
from landslide_mitigation_tasks.landslide_mitigation_functions import (
    thickness_weighted_combine,
    DEPTH_WEIGHTS_0_30CM,
)


# ==================================================================== #
# 0. Parent dir-creator
# ==================================================================== #

def submission(p):
    """Creates p.submission_local_dir and p.submission_work_dir."""
    if p.run_this:
        os.makedirs(p.submission_local_dir, exist_ok=True)
        os.makedirs(p.submission_work_dir, exist_ok=True)
        p.L.info(f'submission_local_dir ready: {p.submission_local_dir}')
    return p


# ==================================================================== #
# Shared helpers
# ==================================================================== #

def _copy_local(src_path, work_dir):
    """Copy a (likely GDrive-synced) source file into the local work_dir
    and return the local path. Never operate on the GDrive path directly.
    """
    os.makedirs(work_dir, exist_ok=True)
    local_path = os.path.join(work_dir, os.path.basename(src_path))
    if not os.path.exists(local_path):
        shutil.copy2(src_path, local_path)
    return local_path


def _ensure_supported_dtype(local_src_path, work_dir):
    """hazelbean's NDV lookup table (get_correct_ndv_from_dtype_flex) has no
    entry for GDAL type 14 (Int8, a newer GDAL addition), so
    is_path_global_pyramid crashes on any Int8 input before make_path_pog
    gets a chance to convert it. Pre-convert Int8 -> Int16 ourselves,
    preserving the nodata value, so make_path_pog never sees an Int8 input.
    """
    ds = gdal.Open(local_src_path)
    band = ds.GetRasterBand(1)
    dtype = band.DataType
    ndv = band.GetNoDataValue()
    ds = None

    if dtype != gdal.GDT_Int8:
        return local_src_path

    out_path = os.path.join(
        work_dir, os.path.splitext(os.path.basename(local_src_path))[0] + '_int16.tif'
    )
    if not os.path.exists(out_path):
        translate_options = gdal.TranslateOptions(
            outputType=gdal.GDT_Int16,
            noData=ndv,
            creationOptions=['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                              'BLOCKXSIZE=256', 'BLOCKYSIZE=256'],
        )
        result_ds = gdal.Translate(out_path, local_src_path, options=translate_options)
        if result_ds is None:
            raise RuntimeError(f'gdal.Translate (Int8->Int16) failed: {local_src_path}')
        result_ds = None
    return out_path


def _cleanup(*paths):
    """Delete local scratch files once they're no longer needed. Disk space
    is tight (see submission_tasks.py history) - never let more than one
    layer's worth of intermediates sit in submission_work_dir at a time.
    """
    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)


import glob

_HAZELBEAN_TEMP_PREFIXES = ('copy_', 'translate_', 'resample_', 'censor_', 'reclassify_')

def _sweep_hazelbean_temp_files(work_dir):
    """hazelbean's own make_path_pog creates its intermediates (copy_*,
    resample_*, etc, see hb.temp() calls in hazelbean/pog.py) with
    remove_at_exit=True - an atexit hook that only fires when this whole
    script's Python process exits, not after each make_path_pog call. Since
    we process many layers/years in one long-lived process, these pile up
    unless we sweep them ourselves after each layer's POG is verified.
    """
    for prefix in _HAZELBEAN_TEMP_PREFIXES:
        for path in glob.glob(os.path.join(work_dir, f'{prefix}*')):
            os.remove(path)


def _pogify(local_src_path, out_path, p, output_data_type=None, force_rewrite=False,
            extra_cleanup_paths=()):
    """Converts local_src_path -> out_path (a POG), verifies it, then
    deletes local_src_path (and any dtype-converted stand-in, plus
    whatever intermediate paths the caller passed in extra_cleanup_paths)
    since disk space is tight and out_path is all that's needed downstream.
    """
    if os.path.exists(out_path) and not force_rewrite:
        return out_path
    dtype_converted_path = _ensure_supported_dtype(local_src_path, p.submission_work_dir)
    hb.make_path_pog(
        dtype_converted_path,
        output_raster_path=out_path,
        output_data_type=output_data_type,
        verbose=True,
    )
    is_pog = hb.is_path_pog(out_path, verbose=True)
    if not is_pog:
        raise RuntimeError(f'make_path_pog produced an invalid POG: {out_path}')
    p.L.info(f'POG verified: {out_path}')

    cleanup_paths = [local_src_path, *extra_cleanup_paths]
    if dtype_converted_path != local_src_path:
        cleanup_paths.append(dtype_converted_path)
    _cleanup(*cleanup_paths)
    _sweep_hazelbean_temp_files(p.submission_work_dir)
    return out_path


# ==================================================================== #
# 1. GAEZ zones - 30 arcsec, exact match, categorical
# ==================================================================== #

def pog_gaez(p):
    if p.run_this:
        out_path = os.path.join(p.submission_local_dir, 'gaez_zones.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.pog_gaez_path = out_path
            return p

        src_path = os.path.join(p.raw_input_data_dir, 'fao_gaez', 'GAEZ-V5.AEZ57.tif')
        local_src = _copy_local(src_path, p.submission_work_dir)
        _pogify(local_src, out_path, p, output_data_type=gdal.GDT_Byte, force_rewrite=p.force_run)
        p.pog_gaez_path = out_path
    return p


# ==================================================================== #
# 2. WorldClim BIO12 - 30 arcsec, exact match, continuous
# ==================================================================== #

def _replace_negatives(data):
    """WorldClim BIO12 carries bad negative-value artifacts (confirmed in
    the original project pipeline's reproject_worldclim_bio12) - baked in
    here so the submitted base_data is analysis-ready for every consumer,
    not just this project. The source's own NDV sentinel (-3.4e+38) is
    itself negative, so it gets swept up into -9999 too, which is exactly
    the POG-standard float NDV.
    """
    data[data < 0] = -9999.0
    return data


def pog_worldclim_bio12(p):
    if p.run_this:
        out_path = os.path.join(p.submission_local_dir, 'worldclim_bio12.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.pog_worldclim_bio12_path = out_path
            return p

        src_path = os.path.join(p.raw_input_data_dir, 'worldclim', 'wc2.1_30s_bio_12.tif')
        local_src = _copy_local(src_path, p.submission_work_dir)

        cleaned_path = os.path.join(p.submission_work_dir, 'worldclim_bio12_cleaned.tif')
        pygeo.raster_calculator(
            base_raster_path_band_const_list=[(local_src, 1)],
            local_op=_replace_negatives,
            target_raster_path=cleaned_path,
            datatype_target=gdal.GDT_Float32,
            nodata_target=-9999.0,
            calc_raster_stats=True,
            raster_driver_creation_tuple=('GTIFF', ('TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                                                     'BLOCKXSIZE=256', 'BLOCKYSIZE=256')),
        )
        p.L.info(f'WorldClim BIO12 negative-value artifacts cleaned: {cleaned_path}')

        _pogify(cleaned_path, out_path, p, force_rewrite=p.force_run, extra_cleanup_paths=[local_src])
        p.pog_worldclim_bio12_path = out_path
    return p


# ==================================================================== #
# 3. Soil depth (Pelletier/ORNL) - 30 arcsec pixel size, but native
#    extent is truncated well short of the poles (18000 rows, not the
#    full-globe 21600), so make_path_pog's implicit resample-to-match
#    kicks in. That resample only relabels the header NDV tag to -9999
#    without reclassifying pixels still holding the source's real NDV
#    (-1), so every true NoData pixel silently becomes "-1 m of soil" -
#    confirmed by inspecting the resulting POG (min=-1, should be masked).
#    Fixed by explicitly warping onto the full pyramid grid ourselves,
#    with src_nodata/dst_nodata set so gdal.Warp does the reclassification
#    (and extent padding) correctly.
# ==================================================================== #

def pog_soil_depth(p):
    if p.run_this:
        out_path = os.path.join(p.submission_local_dir, 'soil_depth.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.pog_soil_depth_path = out_path
            return p

        src_path = os.path.join(
            p.raw_input_data_dir, 'Global_Soil_Regolith_Sediment_1304', 'data',
            'average_soil_and_sedimentary-deposit_thickness.tif'
        )
        local_src = _copy_local(src_path, p.submission_work_dir)

        resampled_path = os.path.join(p.submission_work_dir, 'soil_depth_30sec.tif')
        warp_to_reference(
            local_src, resampled_path, p.ha_per_cell_30sec_path,
            resample_method='near',  # same resolution - extent-pad + NDV fix only
            src_nodata=-1, dst_nodata=-9999.0,
            output_type=gdal.GDT_Int16,
        )
        p.L.info(f'Soil depth warped onto full pyramid extent (NDV fixed): {resampled_path}')

        _pogify(resampled_path, out_path, p, force_rewrite=p.force_run, extra_cleanup_paths=[local_src])
        p.pog_soil_depth_path = out_path
    return p


# ==================================================================== #
# 4. LandScan population - 30 arcsec pixel size, but native extent is
#    truncated (~84N, not the full-globe 90N), so make_path_pog's implicit
#    resample-to-match kicks in - same NDV-not-reclassified bug as soil
#    depth (confirmed: resulting POG had population = -2147483647, i.e.
#    the raw NDV sentinel, for what should have been NoData pixels).
#    Fixed the same way: explicit warp onto the full pyramid grid with
#    src_nodata/dst_nodata set. No aggregation happens (same resolution,
#    'near' resample), so per-cell counts are otherwise unchanged from
#    source - sums are preserved.
# ==================================================================== #

def pog_landscan(p):
    if p.run_this:
        out_dir = os.path.join(p.submission_local_dir, 'landscan')
        os.makedirs(out_dir, exist_ok=True)
        p.pog_landscan_paths = {}

        for year in p.data_processing_range:
            src_path = os.path.join(p.raw_input_data_dir, 'landscan', f'landscan-global-{year}.tif')
            if not os.path.exists(src_path):
                p.L.warning(f'LandScan {year} not found, skipping.')
                continue

            out_path = os.path.join(out_dir, f'landscan_{year}.tif')
            if os.path.exists(out_path) and not p.force_run:
                p.pog_landscan_paths[year] = out_path
                continue

            local_src = _copy_local(src_path, p.submission_work_dir)

            resampled_path = os.path.join(p.submission_work_dir, f'landscan_{year}_30sec.tif')
            warp_to_reference(
                local_src, resampled_path, p.ha_per_cell_30sec_path,
                resample_method='near',  # same resolution - extent-pad + NDV fix only
                src_nodata=-2147483647, dst_nodata=-9999.0,
                output_type=gdal.GDT_Int32,
            )
            p.L.info(f'LandScan {year} warped onto full pyramid extent (NDV fixed): {resampled_path}')

            _pogify(
                resampled_path, out_path, p, force_rewrite=p.force_run,
                extra_cleanup_paths=[local_src],
            )
            p.pog_landscan_paths[year] = out_path
    return p


# ==================================================================== #
# 5. GRIP4 road density - 300 arcsec (5 arcmin), exact match, .asc with
#    no embedded CRS - assign WGS84 before pogifying
# ==================================================================== #

def pog_grip_roads(p):
    if p.run_this:
        out_path = os.path.join(p.submission_local_dir, 'grip4_road_density.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.pog_grip_roads_path = out_path
            return p

        src_path = os.path.join(p.raw_input_data_dir, 'GRIP4_density_total', 'grip4_total_dens_m_km2.asc')
        local_src_asc = _copy_local(src_path, p.submission_work_dir)

        from osgeo import osr
        probe_ds = gdal.Open(local_src_asc)
        has_crs = probe_ds.GetProjection() not in (None, '')
        probe_ds = None

        local_src = local_src_asc
        extra_cleanup = []
        if not has_crs:
            vrt_path = os.path.join(p.submission_work_dir, 'grip4_wgs84.vrt')
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            gdal.Translate(vrt_path, local_src_asc, outputSRS=srs.ExportToWkt())
            local_src = vrt_path
            extra_cleanup.append(local_src_asc)  # local_src is now the vrt, not the .asc
            p.L.info('Assigned WGS84 CRS to GRIP4 .asc (confirmed via source ReadMe.txt).')

        _pogify(local_src, out_path, p, force_rewrite=p.force_run, extra_cleanup_paths=extra_cleanup)
        p.pog_grip_roads_path = out_path
    return p


# ==================================================================== #
# 6. HiHydroSoil Ksat - 250m (~8.2 arcsec), NOT a supported resolution.
#    Combine depth layers, then pre-resample to 10 arcsec before pogifying.
# ==================================================================== #

def pog_hihydrosoil_ksat(p):
    if p.run_this:
        out_path = os.path.join(p.submission_local_dir, 'hihydrosoil_ksat.tif')
        if os.path.exists(out_path) and not p.force_run:
            p.pog_hihydrosoil_ksat_path = out_path
            return p

        depth_paths_local = {}
        for depth in DEPTH_WEIGHTS_0_30CM:
            src_path = os.path.join(p.raw_input_data_dir, 'hihydrosoil', f'Ksat_{depth}_M_250m.tif')
            if not os.path.exists(src_path):
                raise FileNotFoundError(f'Missing HiHydroSoil file: {src_path}')
            depth_paths_local[depth] = _copy_local(src_path, p.submission_work_dir)

        combined_path = os.path.join(p.submission_work_dir, 'hihydrosoil_ksat_combined_native.tif')
        thickness_weighted_combine(depth_paths_local, combined_path)
        p.L.info(f'HiHydroSoil Ksat combined 0-30cm: {combined_path}')
        _cleanup(*depth_paths_local.values())  # no longer needed once combined

        resampled_path = os.path.join(p.submission_work_dir, 'hihydrosoil_ksat_10sec.tif')
        warp_to_reference(
            combined_path, resampled_path, p.ha_per_cell_10sec_path,
            resample_method='average',  # continuous rate, downsampling ~250m -> ~300m
            src_nodata=-9999.0, dst_nodata=-9999.0,
            output_type=gdal.GDT_Float32,
        )
        p.L.info(f'HiHydroSoil Ksat pre-resampled to 10 arcsec: {resampled_path}')
        _cleanup(combined_path)  # no longer needed once resampled

        _pogify(resampled_path, out_path, p, force_rewrite=p.force_run)
        p.pog_hihydrosoil_ksat_path = out_path
    return p


# ==================================================================== #
# 7. SoilGrids sand/clay/soc/bdod - 250m in a projected (Homolosine-like)
#    CRS, NOT a supported resolution. Combine depths, reproject +
#    resample to 10 arcsec before pogifying.
# ==================================================================== #

SOILGRIDS_PROPERTIES = {
    'sand_pct': ('sand', 10),
    'clay_pct': ('clay', 10),
    'org_carbon_pct': ('soc', 10),
    'bulk_density': ('bdod', 100),
}

def pog_soilgrids(p):
    if p.run_this:
        p.pog_soilgrids_paths = {}

        for out_name, (prop_code, conv_factor) in SOILGRIDS_PROPERTIES.items():
            out_path = os.path.join(p.submission_local_dir, f'soilgrids_{out_name}.tif')
            if os.path.exists(out_path) and not p.force_run:
                p.pog_soilgrids_paths[out_name] = out_path
                continue

            depth_paths_local = {}
            for depth in DEPTH_WEIGHTS_0_30CM:
                src_path = os.path.join(p.raw_input_data_dir, 'soilgrids', f'{prop_code}_{depth}_mean.tif')
                if not os.path.exists(src_path):
                    raise FileNotFoundError(f'Missing SoilGrids file: {src_path}')
                depth_paths_local[depth] = _copy_local(src_path, p.submission_work_dir)

            combined_path = os.path.join(p.submission_work_dir, f'soilgrids_{out_name}_combined_native.tif')
            thickness_weighted_combine(depth_paths_local, combined_path, conv_factor=conv_factor)
            p.L.info(f'SoilGrids {out_name} combined 0-30cm: {combined_path}')
            _cleanup(*depth_paths_local.values())  # no longer needed once combined

            resampled_path = os.path.join(p.submission_work_dir, f'soilgrids_{out_name}_10sec.tif')
            warp_to_reference(
                combined_path, resampled_path, p.ha_per_cell_10sec_path,
                resample_method='average',  # continuous %/density, reprojecting + downsampling
                src_nodata=-9999.0, dst_nodata=-9999.0,
                output_type=gdal.GDT_Float32,
            )
            p.L.info(f'SoilGrids {out_name} reprojected + resampled to 10 arcsec: {resampled_path}')
            _cleanup(combined_path)  # no longer needed once resampled

            _pogify(resampled_path, out_path, p, force_rewrite=p.force_run)
            p.pog_soilgrids_paths[out_name] = out_path
    return p


# ==================================================================== #
# 8. ERA5 daily-max rainfall - 0.1 deg (360 arcsec), NOT a supported
#    resolution (falls between 300 and 900 arcsec tolerance windows).
#    Pre-resample to 300 arcsec before pogifying.
# ==================================================================== #

def pog_era5_rain(p):
    if p.run_this:
        out_dir = os.path.join(p.submission_local_dir, 'era5_land')
        os.makedirs(out_dir, exist_ok=True)
        p.pog_era5_rain_paths = {}

        for year in p.data_processing_range:
            src_path = os.path.join(
                p.raw_input_data_dir, 'era5_land_precip_annual_tif', f'era5_max_daily_mm_{year}.tif'
            )
            if not os.path.exists(src_path):
                p.L.warning(f'ERA5 max daily rain {year} not found, skipping.')
                continue

            out_path = os.path.join(out_dir, f'era5_max_daily_mm_{year}.tif')
            if os.path.exists(out_path) and not p.force_run:
                p.pog_era5_rain_paths[year] = out_path
                continue

            local_src = _copy_local(src_path, p.submission_work_dir)

            resampled_path = os.path.join(p.submission_work_dir, f'era5_max_daily_mm_{year}_300sec.tif')
            warp_to_reference(
                local_src, resampled_path, p.ha_per_cell_300sec_path,
                resample_method='average',  # continuous rain intensity, ~0.1deg -> 300 arcsec
                src_nodata=-9999.0, dst_nodata=-9999.0,
                output_type=gdal.GDT_Float32,
            )
            p.L.info(f'ERA5 {year} pre-resampled to 300 arcsec: {resampled_path}')

            _pogify(
                resampled_path, out_path, p, force_rewrite=p.force_run,
                extra_cleanup_paths=[local_src],
            )
            p.pog_era5_rain_paths[year] = out_path
    return p


# ==================================================================== #
# 9. UGLC landslide event points - vector/tabular, not a raster, no
#    POG conversion applies. Still needs a Ref Path so it's submitted
#    as-is (copied straight from the GDrive raw source).
# ==================================================================== #

def copy_uglc_points(p):
    if p.run_this:
        out_dir = os.path.join(p.submission_local_dir, 'uglc')
        out_path = os.path.join(out_dir, 'UGLC_point.csv')
        if os.path.exists(out_path) and not p.force_run:
            p.uglc_submission_path = out_path
            return p

        src_path = os.path.join(p.raw_input_data_dir, 'uglc', 'UGLC_point.csv')
        if not os.path.exists(src_path):
            raise FileNotFoundError(f'Missing UGLC file: {src_path}')

        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(src_path, out_path)
        p.L.info(f'UGLC points copied as-is (no POG conversion, vector/tabular): {out_path}')
        p.uglc_submission_path = out_path
    return p
