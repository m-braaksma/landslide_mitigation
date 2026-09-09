"""
run_make_base_data_submissions.py

Standalone driver: converts landslide_mitigation's raw input rasters
into POGs staged locally for the TEEMs base_data submissions process.
Not wired into the main modeling pipeline (run_landslide_mitigation.py) -
this is a one-off / occasional contribution step.
"""
import os
import hazelbean as hb
from landslide_mitigation_tasks import submission_tasks


def build_submission_task_tree(p):
    p.submission_task = p.add_task(submission_tasks.submission, creates_dir=True)
    p.pog_gaez_task = p.add_task(submission_tasks.pog_gaez, creates_dir=False)
    p.pog_worldclim_bio12_task = p.add_task(submission_tasks.pog_worldclim_bio12, creates_dir=False)
    p.pog_soil_depth_task = p.add_task(submission_tasks.pog_soil_depth, creates_dir=False)
    p.pog_landscan_task = p.add_task(submission_tasks.pog_landscan, creates_dir=False)
    p.pog_grip_roads_task = p.add_task(submission_tasks.pog_grip_roads, creates_dir=False)
    p.pog_hihydrosoil_ksat_task = p.add_task(submission_tasks.pog_hihydrosoil_ksat, creates_dir=False)
    p.pog_soilgrids_task = p.add_task(submission_tasks.pog_soilgrids, creates_dir=False)
    p.pog_era5_rain_task = p.add_task(submission_tasks.pog_era5_rain, creates_dir=False)
    p.copy_uglc_points_task = p.add_task(submission_tasks.copy_uglc_points, creates_dir=False)
    p.write_submission_readme_task = p.add_task(submission_tasks.write_submission_readme, creates_dir=False)
    return p


if __name__ == '__main__':
    hb.log('Starting landslide_mitigation base_data submission build...')
    p = hb.ProjectFlow()
    p.force_run = False
    p.L = hb.get_logger('landslide_mitigation_submission')

    # Raw, untouched source data for this project (GDrive-synced - read only,
    # never written to; see submission_tasks.py module docstring). NOT the
    # same thing as p.base_data_dir below - that name is reserved by
    # hazelbean's get_path() convention for the local base_data cache.
    p.raw_input_data_dir = (
        '/Users/mbraaksma/Library/CloudStorage/GoogleDrive-braak014@umn.edu/'
        'Shared drives/NatCapTEEMs/Projects/Global GEP/Ecosystem Services '
        'SubFolders/Landslides/global_results/input_data_raw'
    )
    p.base_data_dir = os.path.join(os.path.expanduser('~'), 'Files', 'base_data')

    # Canonical pyramid match rasters. Used to pre-resample the three sources
    # that don't land on a supported resolution (pog_hihydrosoil_ksat,
    # pog_soilgrids, pog_era5_rain) AND to explicitly warp any source whose
    # native extent doesn't cover the full globe (pog_landscan, pog_soil_depth
    # - both truncated well short of the poles) onto the exact pyramid grid
    # ourselves, with correct NDV reclassification. Relying on make_path_pog's
    # own implicit resample-to-match for that turned out to silently corrupt
    # NoData handling: it relabels the header NDV tag to the POG-spec value
    # but never reclassifies pixels that held the source's original NDV, so
    # every true NoData pixel becomes bogus real data (e.g. LandScan's
    # -2147483647 sentinel counted as -2.1 billion people). See
    # pog_landscan/pog_soil_depth docstrings.
    pyramids_dir = os.path.join(p.base_data_dir, 'pyramids')
    p.ha_per_cell_30sec_path = os.path.join(pyramids_dir, 'ha_per_cell_30sec.tif')
    p.ha_per_cell_10sec_path = os.path.join(pyramids_dir, 'ha_per_cell_10sec.tif')
    p.ha_per_cell_300sec_path = os.path.join(pyramids_dir, 'ha_per_cell_300sec.tif')

    p.project_name = 'global_results_si_submission'
    p.user_dir = os.path.expanduser('~')
    p.project_dir = os.path.join(p.user_dir, 'Files', 'landslide_mitigation', p.project_name)
    p.set_project_dir(p.project_dir)

    # Output goes straight into the local base_data cache's submissions/
    # folder - this is both where get_path() will find it locally (ref_path
    # "submissions/landslide_mitigation/...") and the exact folder structure
    # to upload to the TEEMs GDrive submissions drive. Scratch work stays in
    # the project dir, well away from GDrive and from base_data itself; see
    # submission_tasks.py module docstring for why nothing here is written
    # directly to the GDrive-synced TEEMs share.
    p.submission_local_dir = os.path.join(p.base_data_dir, 'submissions', 'landslide_mitigation')
    p.submission_work_dir = os.path.join(p.project_dir, 'base_data_submission_work')

    p.data_processing_range = range(2007, 2020)

    build_submission_task_tree(p)
    p.execute()
