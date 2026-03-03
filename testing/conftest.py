"""
conftest.py - Shared pytest fixtures and configuration

This file is automatically discovered by pytest and makes fixtures available
to all test files in the project.
"""

import os
import sys
import tempfile
import shutil
import pytest
import numpy as np
from unittest.mock import Mock
import rasterio
from rasterio.transform import from_bounds

# Add project root to path so tests can import modules
# Adjust this path based on your project structure
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)


@pytest.fixture(scope='session')
def test_data_dir():
    """
    Create a temporary directory for test data that persists for the entire test session.
    """
    temp_dir = tempfile.mkdtemp(prefix='landslide_test_')
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_project_flow(test_data_dir):
    """
    Create a mock ProjectFlow object with all necessary attributes.
    This is the main fixture that most tests will use.
    """
    p = Mock()
    
    # Set up paths
    p.base_data_dir = os.path.join(test_data_dir, 'base_data')
    p.preprocess_data_dir = os.path.join(test_data_dir, 'preprocessed')
    p.project_dir = test_data_dir
    
    # Create directories
    os.makedirs(p.base_data_dir, exist_ok=True)
    os.makedirs(p.preprocess_data_dir, exist_ok=True)
    
    # Set parameters
    p.time_range = range(2000, 2003)
    p.max_location_accuracy_m = 1000
    p.run_this = True
    
    # Mock logger
    p.L = Mock()
    
    # Reference raster path
    p.reference_raster_path = os.path.join(test_data_dir, 'reference.tif')
    
    return p


@pytest.fixture
def small_reference_raster(mock_project_flow):
    """
    Create a small reference raster for testing (50x50 pixels).
    """
    width, height = 50, 50
    bounds = (-180, -90, -170, -80)
    transform = from_bounds(*bounds, width, height)
    
    with rasterio.open(
        mock_project_flow.reference_raster_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rasterio.uint8,
        crs='EPSG:4326',
        transform=transform,
        nodata=255
    ) as dst:
        dst.write(np.zeros((height, width), dtype=np.uint8), 1)
    
    return mock_project_flow.reference_raster_path


# Pytest configuration
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow", 
        action="store_true", 
        default=False, 
        help="run slow tests"
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests"
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Skip slow/integration tests unless explicitly requested."""
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
        if "integration" in item.keywords and not config.getoption("--run-integration"):
            item.add_marker(skip_integration)
