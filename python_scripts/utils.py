import hashlib
import numpy as np
import sys
import contextlib
import os
import logging

import boto3
from urllib.parse import urlparse
from botocore.exceptions import ClientError
from paretoset import paretoset


def get_pareto_layers(df, sense,  num_layers):
    df_out = df.copy()
    df_out['layer'] = num_layers 
    df_current = df.copy()
    
    
    for layer_idx in range(num_layers):
        # paretoset returns a boolean mask for the current non-dominated front
        mask = paretoset(df_current.values, sense=sense)
        layer_ids = df_current.index[mask]
        other_ids = df_current.index[~mask] # Define sense (max/min) for your objectives
        
        # Store the current layer
        df_out.loc[layer_ids, 'layer'] = layer_idx
        
        # Remove these points and move to the next layer
        df_current = df_current.loc[other_ids]
        
    return df_out  

def s3_list_files(s3_full_path):
    """
    Returns a list of full s3:// paths for all files in the given S3 folder.
    """
    parsed = urlparse(s3_full_path)
    if parsed.scheme != 's3':
        raise ValueError("Path must start with s3://")
    
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip('/')
    
    # Ensure prefix ends with a slash if it's meant to be a folder
    if prefix and not prefix.endswith('/'):
        prefix += '/'
        
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    
    file_list = []
    
    # Paginator handles the "next token" logic automatically
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                # Construct the full s3:// path for each object
                file_path = f"s3://{bucket_name}/{obj['Key']}"
                file_list.append(file_path)
                
    return file_list



def s3_folder_exists(s3_full_path):
    """
    Checks if an S3 folder exists given a full 's3://bucket/path/to/folder' string.
    """
    # Parse the S3 URL
    parsed = urlparse(s3_full_path)
    if parsed.scheme != 's3':
        raise ValueError("Path must start with s3://")
    
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip('/')
    
    # Ensure prefix ends with a slash to avoid partial matches
    if prefix and not prefix.endswith('/'):
        prefix += '/'
        
    s3 = boto3.client('s3')
    
    # List objects with the prefix, limit to 1 for performance
    response = s3.list_objects_v2(
        Bucket=bucket_name,
        Prefix=prefix,
        MaxKeys=1
    )
    
    return 'Contents' in response

def s3_file_exists(s3_path: str) -> bool:
    """
    Checks if a file exists at the given S3 path.
    
    :param s3_path: The full S3 path (e.g., 's3://my-bucket/path/to/file.txt')
    :return: True if exists, False otherwise.
    """
    # Parse the S3 URI
    parsed = urlparse(s3_path)
    bucket_name = parsed.netloc
    key = parsed.path.lstrip('/')
    
    s3_client = boto3.client('s3')
    
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        return True
    except ClientError as e:
        # If the error code is 404, the file does not exist
        if e.response['Error']['Code'] == "404":
            return False
        else:
            # If it's a different error (e.g., 403 Forbidden), re-raise it
            raise e
        



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