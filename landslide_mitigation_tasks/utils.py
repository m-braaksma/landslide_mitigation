import os
import hazelbean as hb
from osgeo import gdal
import pygeoprocessing as pygeo
from decimal import Decimal
import numpy as np
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from contextlib import contextmanager


# ------------------------------
# Spatial utils: raster warping and reclassification using GDAL
# ------------------------------

def warp_raster_gdal(
    base_raster_path,
    target_pixel_size,
    target_raster_path,
    resample_method='bilinear',
    target_bb=None,
    base_sr_wkt=None,
    target_sr_wkt=None,
    gtiff_creation_options=None,
    output_data_type=None,
    src_ndv=None,
    dst_ndv=None,
    target_aligned_pixels=True
):
    """
    Resize/resample raster to desired pixel size, bbox and projection using GDAL only (no Hazelbean dependencies).
    """

    if isinstance(target_pixel_size, (float, int)):
        target_pixel_size = (target_pixel_size, -target_pixel_size)

    # Open base raster
    base_raster = gdal.Open(base_raster_path)
    if base_raster is None:
        raise RuntimeError(f"Could not open base raster: {base_raster_path}")

    # Get info from base raster if needed
    if target_sr_wkt is None:
        target_sr_wkt = base_raster.GetProjection()
    if target_bb is None:
        gt = base_raster.GetGeoTransform()
        minx = gt[0]
        maxy = gt[3]
        maxx = minx + (base_raster.RasterXSize * gt[1])
        miny = maxy + (base_raster.RasterYSize * gt[5])
        target_bb = [minx, miny, maxx, maxy]
    if output_data_type is None:
        output_data_type = base_raster.GetRasterBand(1).DataType
    if src_ndv is None:
        src_ndv = base_raster.GetRasterBand(1).GetNoDataValue()
    if dst_ndv is None:
        dst_ndv = src_ndv
    if gtiff_creation_options is None:
        gtiff_creation_options = ['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS']

    # Set GDAL cache to 1GB for faster processing
    gdal.SetCacheMax(1024 * 1024 * 1024)

    print(f"Starting GDAL warp: {base_raster_path} -> {target_raster_path}")

    # Calculate raster size
    target_x_size = int(abs(float(target_bb[2] - target_bb[0]) / target_pixel_size[0]))
    target_y_size = int(abs(float(target_bb[3] - target_bb[1]) / target_pixel_size[1]))
    x_residual = (abs(target_x_size * target_pixel_size[0]) - (target_bb[2] - target_bb[0]))
    if not np.isclose(x_residual, 0.0):
        target_x_size += 1
    y_residual = (abs(target_y_size * target_pixel_size[1]) - (target_bb[3] - target_bb[1]))
    if not np.isclose(y_residual, 0.0):
        target_y_size += 1
    if target_x_size == 0:
        target_x_size = 1
    if target_y_size == 0:
        target_y_size = 1
    target_bb[2] = float(Decimal(str(target_bb[0])) + abs(Decimal(str(target_pixel_size[0])) * Decimal(str(target_x_size))))
    target_bb[3] = float(Decimal(str(target_bb[1])) + abs(Decimal(str(target_pixel_size[1])) * Decimal(str(target_y_size))))

    reproject_callback = hb.make_gdal_callback("Warp %.1f%% complete %s for %s")

    # Run GDAL Warp
    gdal.Warp(
        target_raster_path, base_raster,
        outputBounds=target_bb,
        xRes=abs(target_pixel_size[0]),
        yRes=abs(target_pixel_size[1]),
        resampleAlg=resample_method,
        outputBoundsSRS=target_sr_wkt,
        srcSRS=base_sr_wkt,
        dstSRS=target_sr_wkt,
        creationOptions=gtiff_creation_options,
        callback=reproject_callback,
        callback_data=[target_raster_path],
        outputType=output_data_type,
        srcNodata=src_ndv,
        dstNodata=dst_ndv,
        targetAlignedPixels=target_aligned_pixels,
    )
    print(f"Finished GDAL warp: {target_raster_path}")
    base_raster = None
    return None

def reclassify_raster_gdal(input_raster_path, rules, output_raster_path, output_data_type=gdal.GDT_Float32, output_ndv=None):
    ds = None
    ds = gdal.Open(input_raster_path)
    band = ds.GetRasterBand(1)
    if output_ndv is None:
        output_ndv = band.GetNoDataValue()
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_raster_path, ds.RasterXSize, ds.RasterYSize, 1, output_data_type)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(output_ndv)

    # Block-wise processing
    block_xsize, block_ysize = band.GetBlockSize()
    if block_xsize == 0 or block_ysize == 0:
        # Fallback to reasonable default if not set
        block_xsize, block_ysize = 256, 256
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    total_tiles = ((rows + block_ysize - 1) // block_ysize) * ((cols + block_xsize - 1) // block_xsize)
    tile_count = 0
    last_percent = -1
    print("Reclassifying raster (block-wise):")
    for y in range(0, rows, block_ysize):
        ysize = min(block_ysize, rows - y)
        for x in range(0, cols, block_xsize):
            xsize = min(block_xsize, cols - x)
            arr = band.ReadAsArray(x, y, xsize, ysize)
            if arr is None:
                arr = np.full((ysize, xsize), output_ndv, dtype=np.float32)
            # Reclassify block
            reclass_arr = np.full(arr.shape, output_ndv, dtype=np.float32)
            for k, v in rules.items():
                reclass_arr[arr == k] = v
            reclass_arr[arr == output_ndv] = output_ndv
            out_band.WriteArray(reclass_arr, xoff=x, yoff=y)
            tile_count += 1
            percent = int(100 * tile_count / total_tiles)
            if percent % 10 == 0 and percent != last_percent:
                print(f"Progress: {percent}% ({tile_count}/{total_tiles} tiles)")
                last_percent = percent

    # Explicitly flush and close before setting to None
    out_band.FlushCache()
    out_band = None  # Release band first
    out_ds.FlushCache()  # Flush the dataset
    # For S3, explicitly close the dataset
    del out_ds
    out_ds = None
    # Close input
    ds = None

# ------------------------------
# S3Handler: S3 and VSICURL utilities
# ------------------------------
BUCKET_NAME = os.environ.get('S3_BUCKET', 'jajohns-tier2')
S3_ENDPOINT = os.environ.get('AWS_S3_ENDPOINT', 'https://s3.msi.umn.edu')

class S3Handler:
    def __init__(self, bucket_name=BUCKET_NAME, endpoint_url=S3_ENDPOINT):
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        self.temp_dir = None
        self.s3cmd_path = shutil.which("s3cmd") or '/Users/mbraaksma/mambaforge/envs/teems_hb_dev/bin/s3cmd'
        self._configure_gdal_s3()

    def _normalize_s3_key(self, s3_path):
        """Convert s3:// or /vsis3/ paths into a bucket-relative key."""
        if s3_path.startswith('/vsis3/'):
            body = s3_path[len('/vsis3/'):]
            parts = body.split('/', 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid /vsis3 path: {s3_path}")
            bucket, key = parts
            if bucket != self.bucket_name:
                raise ValueError(f"Expected bucket {self.bucket_name}, got {bucket}")
            return key

        if s3_path.startswith('s3://'):
            body = s3_path[len('s3://'):]
            parts = body.split('/', 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid s3 path: {s3_path}")
            bucket, key = parts
            if bucket != self.bucket_name:
                raise ValueError(f"Expected bucket {self.bucket_name}, got {bucket}")
            return key

        return s3_path.lstrip('/')

    def _to_s3_url(self, s3_path):
        key = self._normalize_s3_key(s3_path)
        return f"s3://{self.bucket_name}/{key}"

    def _run_s3cmd(self, args, max_retries=3):
        """Run s3cmd with retries and surface a useful error on failure."""
        import time

        cmd = [self.s3cmd_path] + args
        last_result = None
        for attempt in range(max_retries):
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return result
            last_result = result
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        stderr = (last_result.stderr or '').strip() if last_result else ''
        stdout = (last_result.stdout or '').strip() if last_result else ''
        raise RuntimeError(
            f"s3cmd failed after {max_retries} attempts: {' '.join(cmd)}\n"
            f"stdout: {stdout}\nstderr: {stderr}"
        )

    def download_file(self, s3_path, local_path, overwrite=False, max_retries=3, require_nonempty=True):
        """Download one S3 object to local disk with retries and integrity checks."""
        s3_url = self._to_s3_url(s3_path)
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists() and not overwrite:
            if not require_nonempty or local_path.stat().st_size > 0:
                return local_path
            local_path.unlink(missing_ok=True)

        self._run_s3cmd(["get", "--force", s3_url, str(local_path)], max_retries=max_retries)

        if require_nonempty and (not local_path.exists() or local_path.stat().st_size == 0):
            local_path.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded file missing or empty: {local_path}")

        return local_path

    def get_or_cache_local_path(self, s3_path, cache_dir, overwrite=False, max_retries=3):
        """Return a local cached path for an S3 object key or s3:// path."""
        key = self._normalize_s3_key(s3_path)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / Path(key).name
        return str(self.download_file(key, local_path, overwrite=overwrite, max_retries=max_retries))
    
    @contextmanager
    def temp_workspace(self, workspace_name=None):
        """
        Context manager for temporary workspace (for backwards compatibility).
        Creates a temporary directory that gets cleaned up automatically.
        """
        try:
            self.temp_dir = Path(tempfile.mkdtemp(prefix=f"{workspace_name}_" if workspace_name else "temp_"))
            print(f"Created temp workspace: {self.temp_dir}")
            yield self.temp_dir
        finally:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temp workspace: {self.temp_dir}")
                self.temp_dir = None

    def download_to_temp(self, s3_path, filename=None):
        """Download file or directory from S3 to temporary workspace using s3cmd."""
        if not self.temp_dir:
            raise ValueError("No temp workspace created. Use temp_workspace() context manager")
        # If .gdb directory, use sync
        if s3_path.endswith('.gdb') or s3_path.endswith('.gdb/'):
            local_path = self.temp_dir / Path(s3_path).name
            print(f"Syncing {s3_path} directory from S3 with s3cmd...")
            s3_url = self._to_s3_url(s3_path.rstrip('/'))
            self._run_s3cmd(["sync", f"{s3_url}/", str(local_path) + "/"])
            return local_path
        # Otherwise, treat as file
        if filename is None:
            filename = Path(s3_path).name
        local_path = self.temp_dir / filename
        print(f"Downloading {s3_path} from S3 with s3cmd...")
        return self.download_file(s3_path, local_path, overwrite=True)
 
    def upload_from_temp(self, s3_path, filename=None):
        """
        Upload a file or directory from the temp workspace to S3 using s3cmd.
        """
        if not self.temp_dir:
            raise ValueError("No temp workspace created. Use temp_workspace() context manager")
        if filename is None:
            filename = Path(s3_path).name
        local_path = self.temp_dir / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found for upload: {local_path}")
        # If uploading a .gdb directory, use s3cmd sync
        if local_path.is_dir() and local_path.suffix == '.gdb':
            print(f"Syncing directory {local_path} to s3://{self.bucket_name}/{s3_path} with s3cmd...")
            s3_url = self._to_s3_url(s3_path.rstrip('/'))
            self._run_s3cmd(["sync", str(local_path) + "/", f"{s3_url}/"])
            print(f"Upload successful: {local_path} -> s3://{self.bucket_name}/{s3_path}/")
        else:
            print(f"Uploading {local_path} to s3://{self.bucket_name}/{s3_path} with s3cmd...")
            s3_url = self._to_s3_url(s3_path)
            self._run_s3cmd(["put", str(local_path), s3_url])
            print(f"Upload successful: {local_path} -> s3://{self.bucket_name}/{s3_path}")

    # def download_to_temp(self, s3_path, filename=None):
    #     """Download file from S3 to temporary workspace using s3cmd."""
    #     if not self.temp_dir:
    #         raise ValueError("No temp workspace created. Use temp_workspace() context manager")
    #     if filename is None:
    #         filename = Path(s3_path).name
    #     local_path = self.temp_dir / filename
    #     local_path.parent.mkdir(parents=True, exist_ok=True)
    #     print(f"Downloading {s3_path} from S3 with s3cmd...")
    #     cmd = [
    #         "s3cmd", "get",
    #         f"s3://{self.bucket_name}/{s3_path}",
    #         str(local_path)
    #     ]
    #     subprocess.run(cmd, check=True)
    #     return local_path
    
    # def upload_from_temp(self, s3_path, filename=None):
    #     """
    #     Upload a file from the temp workspace to S3 using s3cmd.
    #     """
    #     if not self.temp_dir:
    #         raise ValueError("No temp workspace created. Use temp_workspace() context manager")
    #     if filename is None:
    #         filename = Path(s3_path).name
    #     local_path = self.temp_dir / filename
    #     local_path.parent.mkdir(parents=True, exist_ok=True)
    #     if not local_path.exists():
    #         raise FileNotFoundError(f"File not found for upload: {local_path}")
    #     print(f"Uploading {local_path} to s3://{self.bucket_name}/{s3_path} with s3cmd...")
    #     cmd = [
    #         "s3cmd", "put",
    #         str(local_path),
    #         f"s3://{self.bucket_name}/{s3_path}"
    #     ]
    #     subprocess.run(cmd, check=True)
    #     print(f"Upload successful: {local_path} -> s3://{self.bucket_name}/{s3_path}")
    
    def get_vsicurl_path(self, s3_path):
        """
        Get VSICURL path for direct S3 access without downloading.
        This allows GDAL to read rasters directly from S3.
        """
        return f"/vsicurl/{self.endpoint_url}/{self.bucket_name}/{s3_path}"
    
    def get_vsis3_path(self, s3_path):
        """
        Get VSIS3 path for direct S3 access without downloading.
        This allows GDAL to read rasters directly from S3.
        """
        return f"/vsis3/{self.bucket_name}/{s3_path}"
    
    def _configure_gdal_s3(self):
        """Configure GDAL environment variables for S3 access."""
        # Set GDAL configuration for S3 access
        if not hasattr(self, '_gdal_configured'):
            os.environ['AWS_S3_ENDPOINT'] = self.endpoint_url
            os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
            os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = '.tif,.tiff,.vrt'
            os.environ['CPL_VSIL_CURL_USE_HEAD'] = 'NO'
            os.environ['GDAL_HTTP_TIMEOUT'] = '30'
            os.environ['GDAL_HTTP_CONNECTTIMEOUT'] = '10'
            self._gdal_configured = True
    
    def file_exists(self, s3_path):
        """
        Check if an object exists in S3 bucket.
        """
        try:
            s3_url = self._to_s3_url(s3_path)
            result = subprocess.run([self.s3cmd_path, "ls", s3_url], capture_output=True, text=True)
            return result.returncode == 0 and len(result.stdout.strip()) > 0
        except Exception as e:
            print(f"Error occurred while checking file existence: {e}")
            return False
    
    def get_raster_info_from_s3(self, s3_path):
        """
        Get raster information directly from S3 using VSICURL.
        Returns basic raster info without downloading the file.
        """
        import pygeoprocessing
        vsicurl_path = self.get_vsicurl_path(s3_path)
        return pygeoprocessing.get_raster_info(vsicurl_path)
    
    def read_raster_array_from_s3(self, s3_path, band=1, window=None):
        """
        Read raster array directly from S3 using VSICURL.
        Args:
            s3_path (str): S3 path to raster
            band (int): Band number to read (1-based)
            window (tuple): ((row_start, row_stop), (col_start, col_stop)) for partial reads
        Returns:
            numpy.ndarray: Raster array
        """
        import rasterio
        import numpy as np
        vsicurl_path = self.get_vsicurl_path(s3_path)
        with rasterio.open(vsicurl_path) as src:
            if window:
                row_start, row_stop = window[0]
                col_start, col_stop = window[1]
                rasterio_window = rasterio.windows.Window(
                    col_start, row_start, 
                    col_stop - col_start, row_stop - row_start
                )
                return src.read(band, window=rasterio_window)
            else:
                return src.read(band)
            
    def list_files(self, s3_dir, suffix=None):
        """
        List all files in an S3 directory (prefix) using s3cmd.
        Args:
            s3_dir (str): Directory/prefix in the bucket (can be full s3://... or just key prefix)
            suffix (str, optional): Only return files ending with this suffix (e.g. '.tif')
        Returns:
            List[str]: List of S3 keys (relative to bucket)
        """
        # Normalize s3_dir to key prefix (remove s3://bucket/ if present)
        prefix = self._normalize_s3_key(s3_dir) if s3_dir else ''
        # s3cmd ls s3://bucket/prefix --recursive
        cmd = [
            self.s3cmd_path, 'ls', '--recursive', f's3://{self.bucket_name}/{prefix}'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"s3cmd failed: {result.stderr}")
        files = []
        for line in result.stdout.splitlines():
            # s3cmd output: 2024-01-01 00:00   1234567   s3://bucket/key
            parts = line.split()
            if len(parts) >= 4:
                s3_url = parts[-1]
                # Extract key relative to bucket
                if s3_url.startswith(f's3://{self.bucket_name}/'):
                    key = s3_url[len(f's3://{self.bucket_name}/'):]
                    if (suffix is None or key.endswith(suffix)):
                        files.append(key)
        return files

# Global instance
s3_handler = S3Handler()