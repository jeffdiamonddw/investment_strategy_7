import hashlib
import numpy as np
import sys
import contextlib
import os
import logging

def get_dna_hash(dna_array, precision=8, length=12):
    """
    Generates a short, hardware-independent hex hash for pymoo DNA.
    
    Args:
        dna_array: The DNA (1D or 2D array of floats).
        precision: Rounding to handle floating-point jitter.
        length: The length of the hex string to return (e.g., 12 chars).
    """
    arr = np.round(np.array(dna_array, dtype=np.float64), precision)
    arr_bytes = arr.tobytes()
    
    # Generate SHA-256 and take the first 'length' characters
    hasher = hashlib.sha256()
    hasher.update(arr_bytes)
    
    return hasher.hexdigest()[:length]


def write_to_s3(df, s3_path):
    @contextlib.contextmanager
    def silence_everything():
        save_stdout = sys.stdout
        save_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        try:
            yield
        finally:
            sys.stdout = save_stdout
            sys.stderr = save_stderr

    # In your evaluation loop:
    silence_aws_logs()
    with silence_everything():
        df.to_csv(s3_path)


def silence_aws_logs():
    """
    Finds and removes logging handlers that are printing AWS 
    credential messages to the console.
    """
    for logger_name in ['botocore', 'boto3', 's3fs', 'fsspec', 's3fs.core']:
        logger = logging.getLogger(logger_name)
        # 1. Set the level to WARNING to ignore INFO logs
        logger.setLevel(logging.WARNING)
        # 2. Disable propagation so it doesn't bubble up to the root logger
        logger.propagate = False
        # 3. Clear existing handlers
        logger.handlers = []

import s3fs
import zarr

import xarray as xr
import s3fs

import xarray as xr
import s3fs

def save_to_zarr(da, store_path, append_dim='sim_id'):
    if da.name is None:
        da.name = "data"
        
    fs = s3fs.S3FileSystem()
    
    # 1. Use 'compressors' instead of 'compressor'
    # 2. Use 'None' or a specific zarr.storage.Blosc() object
    encoding = {da.name: {'compressors': None}} 
    
    # 3. Add zarr_format=2 and consolidated=True
    # These explicitly trigger V2 behavior, eliminating V3 warnings
    if fs.exists(store_path):
        da.to_zarr(store_path, append_dim=append_dim, mode='a', 
                   encoding=encoding, zarr_format=2, consolidated=True)
    else:
        da.to_zarr(store_path, mode='w', encoding=encoding, 
                   zarr_format=2, consolidated=True)