import xarray as xr
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import hashlib
import logging
from datetime import datetime
import os

from database.schema import ARGO_PARAMETER_MAPPING, validate_measurement_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetCDFProcessor:
    """
    Process ARGO NetCDF files and extract structured data
    """
    
    def __init__(self):
        self.supported_formats = ['.nc', '.netcdf']
        self.required_variables = ['PRES', 'TEMP', 'PSAL']
        
    def validate_file(self, file_path: str) -> bool:
        """Validate if file is a valid NetCDF file"""
        try:
            if not os.path.exists(file_path):
                return False
                
            # Check file extension
            _, ext = os.path.splitext(file_path)
            if ext.lower() not in self.supported_formats:
                return False
            
            # Try to open with xarray
            with xr.open_dataset(file_path) as ds:
                # Check for basic ARGO structure
                if not any(var in ds.variables for var in self.required_variables):
                    logger.warning(f"File {file_path} does not contain required ARGO variables")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"File validation failed for {file_path}: {str(e)}")
            return False
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of the file for duplicate detection"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate file hash: {str(e)}")
            return ""
    
    # Replace these specific methods in your NetCDFProcessor class:

    def extract_profile_metadata(self, ds: xr.Dataset) -> Dict[str, Any]:
        """Extract profile-level metadata from NetCDF dataset"""
        try:
            metadata = {}
            
            # Safe extraction helper
            def safe_scalar_extract(var_name: str, default=None):
                """Safely extract scalar value from potentially multi-dimensional array"""
                if var_name not in ds.variables:
                    return default
                
                try:
                    data = ds[var_name].values
                    # Handle different array shapes
                    if np.isscalar(data):
                        return data
                    elif hasattr(data, 'item') and data.size == 1:
                        return data.item()
                    elif hasattr(data, '__len__') and len(data) > 0:
                        # Flatten and get first valid value
                        flat_data = np.array(data).flatten()
                        valid_data = flat_data[~pd.isna(flat_data)]
                        if len(valid_data) > 0:
                            return valid_data[0]
                    return default
                except Exception as e:
                    logger.warning(f"Error extracting {var_name}: {e}")
                    return default
            
            # Platform information
            platform_num = safe_scalar_extract('PLATFORM_NUMBER', '')
            if platform_num:
                metadata['platform_number'] = str(platform_num).strip()
                metadata['float_id'] = str(platform_num).strip()
            else:
                metadata['platform_number'] = 'unknown'
                metadata['float_id'] = 'unknown'
            
            # Cycle number
            cycle_num = safe_scalar_extract('CYCLE_NUMBER', 0)
            metadata['cycle_number'] = int(cycle_num) if cycle_num is not None else 0
            
            # Location - handle coordinate arrays properly
            lat = safe_scalar_extract('LATITUDE')
            lon = safe_scalar_extract('LONGITUDE')
            
            if lat is not None and lon is not None:
                metadata['latitude'] = float(lat)
                metadata['longitude'] = float(lon)
            else:
                logger.warning("Missing coordinates, using defaults")
                metadata['latitude'] = 0.0
                metadata['longitude'] = 0.0
            
            # Date/Time handling
            juld = safe_scalar_extract('JULD')
            if juld is not None and not np.isnan(juld):
                try:
                    # Convert Julian day to datetime
                    base_date = pd.Timestamp('1950-01-01')
                    measurement_date = base_date + pd.Timedelta(days=float(juld))
                    metadata['measurement_date'] = measurement_date
                except Exception as e:
                    logger.warning(f"Error converting JULD date: {e}")
                    metadata['measurement_date'] = datetime.now()
            else:
                metadata['measurement_date'] = datetime.now()
            
            # Data center
            data_center = (safe_scalar_extract('DATA_CENTRE') or 
                        safe_scalar_extract('DATA_CENTER') or 
                        'unknown')
            metadata['data_center'] = str(data_center).strip()
            
            logger.info(f"Extracted metadata keys: {list(metadata.keys())}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract profile metadata: {str(e)}")
            return {
                'platform_number': 'unknown',
                'float_id': 'unknown',
                'cycle_number': 0,
                'latitude': 0.0,
                'longitude': 0.0,
                'measurement_date': datetime.now(),
                'data_center': 'unknown'
            }

    def extract_measurements(self, ds: xr.Dataset) -> List[Dict[str, Any]]:
            """Extract measurement data from NetCDF dataset"""
            try:
                measurements = []
                
                # Get the number of levels - FIX THE DIMS WARNING
                n_levels = None
                if 'N_LEVELS' in ds.sizes:  # Use .sizes instead of .dims
                    n_levels = ds.sizes['N_LEVELS']
                elif 'N_PROF' in ds.sizes:
                    n_levels = ds.sizes['N_PROF']
                elif 'PRES' in ds.variables:
                    pres_data = ds['PRES'].values
                    n_levels = len(pres_data) if hasattr(pres_data, '__len__') else 1
                else:
                    logger.error("Cannot determine number of measurement levels")
                    return []
                
                logger.info(f"Processing {n_levels} measurement levels")
                
                # Safe array extraction
                def safe_array_extract(var_name: str):
                    """Safely extract array data"""
                    if var_name not in ds.variables:
                        return None
                    try:
                        data = ds[var_name].values
                        # Ensure it's a 1D array
                        if data.ndim > 1:
                            data = data.flatten()
                        return data
                    except Exception as e:
                        logger.warning(f"Error extracting array {var_name}: {e}")
                        return None
                
                # Extract measurement arrays
                variables = {}
                
                # Core variables
                core_vars = {
                    'pressure': ['PRES', 'PRES_ADJUSTED'],
                    'temperature': ['TEMP', 'TEMP_ADJUSTED'], 
                    'salinity': ['PSAL', 'PSAL_ADJUSTED']
                }
                
                # BGC variables
                bgc_vars = {
                    'oxygen': ['DOXY', 'DOXY_ADJUSTED'],
                    'nitrate': ['NITRATE', 'NITRATE_ADJUSTED'],
                    'ph': ['PH_IN_SITU_TOTAL', 'PH_IN_SITU_TOTAL_ADJUSTED'],
                    'chlorophyll': ['CHLA', 'CHLA_ADJUSTED']
                }
                
                all_vars = {**core_vars, **bgc_vars}
                
                # Extract available variables
                for param_name, var_candidates in all_vars.items():
                    for var_name in var_candidates:
                        data = safe_array_extract(var_name)
                        if data is not None:
                            variables[param_name] = data
                            logger.info(f"Found {var_name} -> {param_name}")
                            break
                
                # Check if we have minimum required variables
                if not all(param in variables for param in ['pressure', 'temperature', 'salinity']):
                    logger.error("Missing required core variables (pressure, temperature, salinity)")
                    return []
                
                # Build measurements list
                max_length = max(len(v) for v in variables.values())
                actual_levels = min(n_levels, max_length)
                
                for i in range(actual_levels):
                    measurement = {}
                    
                    # Extract values for this level
                    for param_name, param_data in variables.items():
                        if i < len(param_data):
                            value = param_data[i]
                            # Check for valid data
                            if (not np.isnan(value) and not np.isinf(value) and 
                                value != -999 and value != 99999):  # Common missing value flags
                                measurement[param_name] = float(value)
                            else:
                                measurement[param_name] = None
                        else:
                            measurement[param_name] = None
                    
                    # Calculate depth from pressure if available
                    if measurement.get('pressure') is not None:
                        measurement['depth'] = measurement['pressure']  # Simple approximation
                    else:
                        measurement['depth'] = None
                    
                    # Set quality flag
                    measurement['quality_flag'] = 1  # Default to good data
                    
                    # Only add if measurement has at least one valid parameter
                    valid_params = [v for v in measurement.values() 
                                if isinstance(v, (int, float)) and v is not None]
                    if valid_params:
                        measurements.append(measurement)
                
                logger.info(f"Extracted {len(measurements)} valid measurements")
                return measurements
                
            except Exception as e:
                logger.error(f"Failed to extract measurements: {str(e)}")
                logger.error(f"Dataset variables: {list(ds.variables.keys())}")
                return []
    
    def process_file(self, file_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process a NetCDF file and return profile metadata and measurements
        
        Returns:
            Tuple of (profile_metadata, measurements_list)
        """
        try:
            # Validate file
            if not self.validate_file(file_path):
                raise ValueError(f"Invalid NetCDF file: {file_path}")
            
            # Calculate file hash for duplicate detection
            file_hash = self.calculate_file_hash(file_path)
            
            # Open dataset
            with xr.open_dataset(file_path) as ds:
                # Extract profile metadata
                profile_metadata = self.extract_profile_metadata(ds)
                profile_metadata['file_hash'] = file_hash
                
                # Extract measurements
                measurements = self.extract_measurements(ds)
                
                logger.info(f"Successfully processed file: {file_path}")
                logger.info(f"Profile: {profile_metadata.get('float_id')} - Cycle: {profile_metadata.get('cycle_number')}")
                logger.info(f"Measurements: {len(measurements)}")
                
                return profile_metadata, measurements
                
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {str(e)}")
            raise
    
    def process_multiple_files(self, file_paths: List[str]) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
        """Process multiple NetCDF files"""
        results = []
        
        for file_path in file_paths:
            try:
                profile_data, measurements = self.process_file(file_path)
                results.append((profile_data, measurements))
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                continue
        
        return results
    
    def get_file_summary(self, file_path: str) -> Dict[str, Any]:
        """Get a quick summary of a NetCDF file without full processing"""
        try:
            if not self.validate_file(file_path):
                return {'error': 'Invalid NetCDF file'}
            
            with xr.open_dataset(file_path) as ds:
                summary = {
                    'file_path': file_path,
                    'file_size': os.path.getsize(file_path),
                    'dimensions': dict(ds.dims),
                    'variables': list(ds.variables.keys()),
                    'global_attributes': dict(ds.attrs),
                }
                
                # Quick metadata extraction
                metadata = self.extract_profile_metadata(ds)
                summary.update(metadata)
                
                return summary
                
        except Exception as e:
            return {'error': f'Failed to read file: {str(e)}'}
