"""
API Caching Module - Provides consistent caching for API calls.

This module implements a decorator-based approach to caching API responses,
with support for different cache strategies and timeout handling.
"""

import os
import json
import time
import hashlib
import pandas as pd
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, Tuple, List

from .config import CACHE_DIR

# Ensure cache directory exists
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Default cache timeout in seconds (7 days)
DEFAULT_CACHE_TIMEOUT = 7 * 24 * 60 * 60


class APICache:
    """
    Handles caching of API responses with support for different strategies.
    """
    @staticmethod
    def generate_cache_key(func_name: str, args: Tuple, kwargs: Dict) -> str:
        """
        Generate a cache key based on function name and arguments.
        
        Args:
            func_name: Name of the function being cached
            args: Function positional arguments
            kwargs: Function keyword arguments
            
        Returns:
            A unique hash string to use as cache key
        """
        # Create a string representation of the function and its arguments
        args_str = str(args) + str(sorted(kwargs.items()))
        key_data = f"{func_name}_{args_str}"
        
        # Create a hash of this string
        hash_obj = hashlib.md5(key_data.encode())
        return hash_obj.hexdigest()

    @staticmethod
    def save_to_cache(data: Any, cache_path: Path) -> None:
        """
        Save data to cache file with timestamp.
        
        Args:
            data: Data to cache (DataFrame or JSON-serializable object)
            cache_path: Path to the cache file
        """
        # Create metadata with timestamp
        metadata = {
            'timestamp': time.time(),
            'cached_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Ensure parent directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle pandas DataFrame
        if isinstance(data, pd.DataFrame):
            # Save metadata in a separate file
            with open(f"{cache_path}.meta", 'w') as f:
                json.dump(metadata, f)
            # Save the DataFrame
            data.to_csv(cache_path, index=False)
        else:
            # For JSON-serializable data, include metadata in the same file
            if isinstance(data, dict):
                # If it's a dict, add metadata directly
                data_with_meta = {
                    '_metadata': metadata,
                    'data': data
                }
            else:
                # For other types (lists, etc.), wrap it
                data_with_meta = {
                    '_metadata': metadata,
                    'data': data
                }
            
            # Save to file
            with open(cache_path, 'w') as f:
                json.dump(data_with_meta, f)

    @staticmethod
    def load_from_cache(cache_path: Path, as_dataframe: bool = True, timeout: int = DEFAULT_CACHE_TIMEOUT) -> Optional[Any]:
        """
        Load data from cache if it exists and is not older than timeout.
        
        Args:
            cache_path: Path to the cache file
            as_dataframe: Whether to load as a pandas DataFrame
            timeout: Cache timeout in seconds
            
        Returns:
            Cached data or None if cache is invalid/expired
        """
        if not cache_path.exists():
            return None
            
        # Check if cache is expired
        current_time = time.time()
        
        try:
            if as_dataframe:
                # For DataFrames, metadata is in a separate file
                meta_path = Path(f"{cache_path}.meta")
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check if cache is expired
                    if current_time - metadata['timestamp'] > timeout:
                        return None
                        
                    # Load the DataFrame
                    return pd.read_csv(cache_path)
                else:
                    # No metadata file, consider cache invalid
                    return None
            else:
                # For JSON data, metadata is included in the file
                with open(cache_path, 'r') as f:
                    data_with_meta = json.load(f)
                
                # Ensure it has the expected structure
                if not isinstance(data_with_meta, dict) or '_metadata' not in data_with_meta:
                    return None
                    
                # Check if cache is expired
                if current_time - data_with_meta['_metadata']['timestamp'] > timeout:
                    return None
                    
                # Return the actual data
                return data_with_meta['data']
                
        except Exception as e:
            print(f"Error loading from cache {cache_path}: {e}")
            return None


def cache_api_response(cache_timeout: int = DEFAULT_CACHE_TIMEOUT, as_dataframe: bool = True, cache_subdir: str = ""):
    """
    Decorator for caching API responses.
    
    Args:
        cache_timeout: Cache timeout in seconds
        as_dataframe: Whether the response should be loaded as a DataFrame
        cache_subdir: Subdirectory within the cache directory for this cache
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check if offline mode is enabled
            offline_mode = os.environ.get("NBA_OFFLINE_MODE", "0").lower() in ("1", "true", "yes")
            
            # Check if force_refresh is explicitly provided
            force_refresh = kwargs.pop('force_refresh', False)
            
            # Generate a cache key
            cache_key = APICache.generate_cache_key(func.__name__, args, kwargs)
            
            # Determine cache path
            if cache_subdir:
                subdir_path = Path(CACHE_DIR) / cache_subdir
                subdir_path.mkdir(parents=True, exist_ok=True)
                cache_path = subdir_path / f"{cache_key}.{'csv' if as_dataframe else 'json'}"
            else:
                cache_path = Path(CACHE_DIR) / f"{cache_key}.{'csv' if as_dataframe else 'json'}"
            
            # Try to load from cache first
            if not force_refresh:
                cached_data = APICache.load_from_cache(cache_path, as_dataframe, cache_timeout)
                if cached_data is not None:
                    return cached_data
                    
            # If in offline mode and no valid cache, we can't proceed
            if offline_mode:
                print(f"No valid cache found for {func.__name__} and offline mode is enabled.")
                # Attempt to load expired cache as a fallback
                if cache_path.exists():
                    print("Loading expired cache as fallback in offline mode.")
                    if as_dataframe:
                        return pd.read_csv(cache_path)
                    else:
                        try:
                            with open(cache_path, 'r') as f:
                                data = json.load(f)
                            return data.get('data', data)  # Handle both formats
                        except Exception:
                            pass
                # Return empty response appropriate for the expected type
                return pd.DataFrame() if as_dataframe else {}
            
            # Make the actual API call
            try:
                response = func(*args, **kwargs)
                
                # Cache the response
                if response is not None:
                    APICache.save_to_cache(response, cache_path)
                
                return response
            except Exception as e:
                print(f"Error in API call {func.__name__}: {e}")
                
                # Fallback to cache even if expired
                if cache_path.exists():
                    print(f"API call failed. Using expired cache for {func.__name__} as fallback.")
                    if as_dataframe:
                        return pd.read_csv(cache_path)
                    else:
                        try:
                            with open(cache_path, 'r') as f:
                                data = json.load(f)
                            return data.get('data', data)  # Handle both formats
                        except Exception:
                            pass
                
                # Return empty response
                return pd.DataFrame() if as_dataframe else {}
                
        return wrapper
    return decorator 