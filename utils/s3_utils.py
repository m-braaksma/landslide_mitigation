import boto3
from botocore.exceptions import ClientError
from contextlib import contextmanager
import tempfile
import shutil
import os
import time
from pathlib import Path
import sys

# Import config
sys.path.append(str(Path(__file__).parent.parent))
from config.config import S3_ENDPOINT, BUCKET_NAME

class S3Handler:
    def __init__(self, bucket_name=BUCKET_NAME, endpoint_url=S3_ENDPOINT):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', endpoint_url=endpoint_url)
        self.temp_dir = None
    
    def create_temp_workspace(self, workspace_name=None):
        # Use /scratch.global instead of default temp location
        scratch_base = Path(f"/scratch.global/{os.environ.get('USER', 'unknown')}")
        scratch_base.mkdir(parents=True, exist_ok=True)
        
        if workspace_name:
            self.temp_dir = scratch_base / f"{workspace_name}_{int(time.time())}"
        else:
            self.temp_dir = scratch_base / f"temp_{int(time.time())}"
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created temp workspace: {self.temp_dir}")
        return self.temp_dir
    
    def cleanup_temp_workspace(self):
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temp workspace: {self.temp_dir}")
            self.temp_dir = None
    
    @contextmanager
    def temp_workspace(self, workspace_name=None):
        try:
            workspace = self.create_temp_workspace(workspace_name)
            yield workspace
        finally:
            self.cleanup_temp_workspace()
    
    def download_to_temp(self, s3_path, filename=None):
        if not self.temp_dir:
            raise ValueError("No temp workspace created. Use create_temp_workspace() or temp_workspace() context manager")
        
        if filename is None:
            filename = Path(s3_path).name
        
        local_path = self.temp_dir / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading {s3_path} from S3...")
        self.s3_client.download_file(self.bucket_name, s3_path, str(local_path))
        return local_path
    
    def upload_from_temp(self, local_filename, s3_path):
        if not self.temp_dir:
            raise ValueError("No temp workspace created. Use create_temp_workspace() or temp_workspace() context manager")
        
        local_path = self.temp_dir / local_filename
        if not local_path.exists():
            raise FileNotFoundError(f"File not found in temp workspace: {local_path}")
        
        print(f"Uploading to {s3_path}...")
        self.s3_client.upload_file(str(local_path), self.bucket_name, s3_path)
    
    def get_temp_path(self, filename):
        if not self.temp_dir:
            raise ValueError("No temp workspace created. Use create_temp_workspace() or temp_workspace() context manager")
        
        return self.temp_dir / filename
    
    # Method to check if a file exists on S3
    def file_exists(self, s3_path):
        """
        Check if an object exists in S3 bucket.
        s3_path is the key relative to the bucket root.
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_path)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise

    # Method to check if a file exists on S3
    def dir_exists(self, s3_path):
        """
        Check if a directory (prefix) exists in S3 bucket.
        s3_path is the directory path relative to the bucket root.
        Should end with '/' for proper directory checking.
        """
        try:
            # Ensure the path ends with '/' for directory checking
            if not s3_path.endswith('/'):
                s3_path += '/'

            # List objects with the prefix to see if any exist
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, 
                Prefix=s3_path, 
                MaxKeys=1
            )

            # If there are any objects with this prefix, the directory exists
            return response.get('KeyCount', 0) > 0

        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise
    
    # Legacy methods for backwards compatibility
    def download(self, s3_path, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {s3_path} from S3...")
        self.s3_client.download_file(self.bucket_name, s3_path, local_path)
    
    def upload(self, local_path, s3_path):
        print(f"Uploading to {s3_path}...")
        self.s3_client.upload_file(local_path, self.bucket_name, s3_path)
    
    def download_directory(self, s3_prefix, local_dir):
        import subprocess
        cmd = ["aws", "s3", "sync", f"s3://{self.bucket_name}/{s3_prefix}", local_dir, 
               "--endpoint-url", S3_ENDPOINT]
        subprocess.run(cmd, check=True)
    
    def upload_directory(self, local_dir, s3_prefix):
        import subprocess
        cmd = ["aws", "s3", "sync", local_dir, f"s3://{self.bucket_name}/{s3_prefix}",
               "--endpoint-url", S3_ENDPOINT]
        subprocess.run(cmd, check=True)

# Global instance
s3_handler = S3Handler()


# import boto3
# from botocore.exceptions import ClientError
# import os
# import tempfile
# import shutil
# from pathlib import Path
# import sys
# from contextlib import contextmanager

# # Import config
# sys.path.append(str(Path(__file__).parent.parent))
# from config.config import S3_ENDPOINT, BUCKET_NAME

# class S3Handler:
#     def __init__(self, bucket_name=BUCKET_NAME, endpoint_url=S3_ENDPOINT):
#         self.bucket_name = bucket_name
#         self.s3_client = boto3.client('s3', endpoint_url=endpoint_url)
#         self.temp_dir = None
    
#     def create_temp_workspace(self, workspace_name=None):
#         """Create a temporary workspace directory"""
#         if workspace_name:
#             self.temp_dir = Path(tempfile.mkdtemp(prefix=f"{workspace_name}_"))
#         else:
#             self.temp_dir = Path(tempfile.mkdtemp())
        
#         print(f"Created temp workspace: {self.temp_dir}")
#         return self.temp_dir
    
#     def cleanup_temp_workspace(self):
#         """Clean up the temporary workspace"""
#         if self.temp_dir and self.temp_dir.exists():
#             shutil.rmtree(self.temp_dir)
#             print(f"Cleaned up temp workspace: {self.temp_dir}")
#             self.temp_dir = None
    
#     @contextmanager
#     def temp_workspace(self, workspace_name=None):
#         """Context manager for temporary workspace"""
#         try:
#             workspace = self.create_temp_workspace(workspace_name)
#             yield workspace
#         finally:
#             self.cleanup_temp_workspace()
    
#     def download_to_temp(self, s3_path, filename=None):
#         """Download file from S3 to temporary workspace"""
#         if not self.temp_dir:
#             raise ValueError("No temp workspace created. Use create_temp_workspace() or temp_workspace() context manager")
        
#         if filename is None:
#             filename = Path(s3_path).name
        
#         local_path = self.temp_dir / filename
#         local_path.parent.mkdir(parents=True, exist_ok=True)
        
#         print(f"Downloading {s3_path} from S3...")
#         self.s3_client.download_file(self.bucket_name, s3_path, str(local_path))
#         return local_path
    
#     def upload_from_temp(self, local_filename, s3_path):
#         """Upload file from temporary workspace to S3"""
#         if not self.temp_dir:
#             raise ValueError("No temp workspace created. Use create_temp_workspace() or temp_workspace() context manager")
        
#         local_path = self.temp_dir / local_filename
#         if not local_path.exists():
#             raise FileNotFoundError(f"File not found in temp workspace: {local_path}")
        
#         print(f"Uploading to {s3_path}...")
#         self.s3_client.upload_file(str(local_path), self.bucket_name, s3_path)
    
#     def get_temp_path(self, filename):
#         """Get path to file in temporary workspace"""
#         if not self.temp_dir:
#             raise ValueError("No temp workspace created. Use create_temp_workspace() or temp_workspace() context manager")
        
#         return self.temp_dir / filename
    
#     # Legacy methods for backwards compatibility
#     def download(self, s3_path, local_path):
#         """Download file from S3 to specified local path"""
#         os.makedirs(os.path.dirname(local_path), exist_ok=True)
#         print(f"Downloading {s3_path} from S3...")
#         self.s3_client.download_file(self.bucket_name, s3_path, local_path)
    
#     def upload(self, local_path, s3_path):
#         """Upload file to S3 from specified local path"""
#         print(f"Uploading to {s3_path}...")
#         self.s3_client.upload_file(local_path, self.bucket_name, s3_path)
    
#     def download_directory(self, s3_prefix, local_dir):
#         """Download all files with given S3 prefix"""
#         import subprocess
#         cmd = ["aws", "s3", "sync", f"s3://{self.bucket_name}/{s3_prefix}", local_dir, 
#                "--endpoint-url", S3_ENDPOINT]
#         subprocess.run(cmd, check=True)
    
#     def upload_directory(self, local_dir, s3_prefix):
#         """Upload directory to S3"""
#         import subprocess
#         cmd = ["aws", "s3", "sync", local_dir, f"s3://{self.bucket_name}/{s3_prefix}",
#                "--endpoint-url", S3_ENDPOINT]
#         subprocess.run(cmd, check=True)

# # Global instance
# s3_handler = S3Handler()
