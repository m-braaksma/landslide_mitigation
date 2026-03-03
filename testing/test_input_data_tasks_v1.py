import os
import rasterio
import numpy as np

def test_glc_raster_properties(raster_path, expected_crs=None, expected_shape=None, expected_nodata=None, min_expected_pixels=None):
    """
    Test basic properties of a GLC raster:
    - File exists
    - CRS matches expected
    - Shape matches expected
    - Nodata value matches expected
    - At least min_expected_pixels are non-nodata
    """
    assert os.path.exists(raster_path), f"Raster file does not exist: {raster_path}"
    with rasterio.open(raster_path) as src:
        if expected_crs is not None:
            assert src.crs == expected_crs, f"CRS mismatch: {src.crs} != {expected_crs}"
        if expected_shape is not None:
            assert src.shape == expected_shape, f"Shape mismatch: {src.shape} != {expected_shape}"
        if expected_nodata is not None:
            assert src.nodata == expected_nodata, f"Nodata mismatch: {src.nodata} != {expected_nodata}"
        arr = src.read(1)
        non_nodata_count = np.count_nonzero(arr != src.nodata)
        if min_expected_pixels is not None:
            assert non_nodata_count >= min_expected_pixels, f"Too few non-nodata pixels: {non_nodata_count} < {min_expected_pixels}"
        # Optionally, print for debugging
        print(f"Raster: {raster_path}")
        print(f"  CRS: {src.crs}")
        print(f"  Shape: {src.shape}")
        print(f"  Nodata: {src.nodata}")
        print(f"  Non-nodata pixels: {non_nodata_count}")

# Example usage in a pytest test:
# def test_glc_raster():
#     test_glc_raster_properties(
#         raster_path="path/to/glc_raster.tif",
#         expected_crs="EPSG:4326",
#         expected_shape=(1000, 1000),
#         expected_nodata=0,
#         min_expected_pixels=10
#     )