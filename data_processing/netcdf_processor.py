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
    
    def extract_profile_metadata(self, ds: xr.Dataset) -> Dict[str, Any]:
        """Extract profile-level metadata from NetCDF dataset"""
        try:
            metadata = {}
            
            # Platform information
            if 'PLATFORM_NUMBER' in ds.variables:
                platform_num = ds['PLATFORM_NUMBER'].values
                if hasattr(platform_num, 'item'):
                    metadata['platform_number'] = str(platform_num.item())
                else:
                    metadata['platform_number'] = str(platform_num[0]) if len(platform_num) > 0 else ''
            
            # Float ID (usually same as platform number)
            metadata['float_id'] = metadata.get('platform_number', 'unknown')
            
            # Cycle number
            if 'CYCLE_NUMBER' in ds.variables:
                cycle_num = ds['CYCLE_NUMBER'].values
                if hasattr(cycle_num, 'item'):
                    metadata['cycle_number'] = int(cycle_num.item())
                else:
                    metadata['cycle_number'] = int(cycle_num[0]) if len(cycle_num) > 0 else 0
            else:
                metadata['cycle_number'] = 0
            
            # Location
            if 'LATITUDE' in ds.variables:
                lat = ds['LATITUDE'].values
                metadata['latitude'] = float(lat.item()) if hasattr(lat, 'item') else float(lat[0])
            
            if 'LONGITUDE' in ds.variables:
                lon = ds['LONGITUDE'].values
                metadata['longitude'] = float(lon.item()) if hasattr(lon, 'item') else float(lon[0])
            
            # Date/Time
            if 'JULD' in ds.variables:
                # ARGO uses Julian days since 1950-01-01
                juld = ds['JULD'].values
                if hasattr(juld, 'item'):
                    julian_day = juld.item()
                else:
                    julian_day = juld[0] if len(juld) > 0 else 0
                
                if not np.isnan(julian_day):
                    # Convert Julian day to datetime
                    base_date = pd.Timestamp('1950-01-01')
                    measurement_date = base_date + pd.Timedelta(days=julian_day)
                    metadata['measurement_date'] = measurement_date
                else:
                    metadata['measurement_date'] = datetime.now()
            else:
                metadata['measurement_date'] = datetime.now()
            
            # Data center
            if 'DATA_CENTRE' in ds.variables or 'DATA_CENTER' in ds.variables:
                dc_var = 'DATA_CENTRE' if 'DATA_CENTRE' in ds.variables else 'DATA_CENTER'
                dc = ds[dc_var].values
                if hasattr(dc, 'item'):
                    metadata['data_center'] = str(dc.item())
                else:
                    metadata['data_center'] = str(dc[0]) if len(dc) > 0 else 'unknown'
            else:
                metadata['data_center'] = 'unknown'
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract profile metadata: {str(e)}")
            return {}
    
    def extract_measurements(self, ds: xr.Dataset) -> List[Dict[str, Any]]:
        """Extract measurement data from NetCDF dataset"""
        try:
            measurements = []
            
            # Get the number of levels
            if 'N_LEVELS' in ds.dims:
                n_levels = ds.dims['N_LEVELS']
            elif 'PRES' in ds.variables:
                n_levels = len(ds['PRES'].values)
            else:
                logger.error("Cannot determine number of measurement levels")
                return []
            
            # Extract measurement arrays
            variables = {}
            
            # Pressure (required)
            if 'PRES' in ds.variables:
                variables['pressure'] = ds['PRES'].values
            else:
                logger.error("PRES variable not found")
                return []
            
            # Temperature (required)
            if 'TEMP' in ds.variables:
                variables['temperature'] = ds['TEMP'].values
            else:
                logger.error("TEMP variable not found")
                return []
            
            # Salinity (required)
            if 'PSAL' in ds.variables:
                variables['salinity'] = ds['PSAL'].values
            else:
                logger.error("PSAL variable not found")
                return []
            
            # Optional BGC parameters
            if 'DOXY' in ds.variables:
                variables['oxygen'] = ds['DOXY'].values
            
            if 'NITRATE' in ds.variables:
                variables['nitrate'] = ds['NITRATE'].values
            
            if 'PH_IN_SITU_TOTAL' in ds.variables:
                variables['ph'] = ds['PH_IN_SITU_TOTAL'].values
            
            if 'CHLA' in ds.variables:
                variables['chlorophyll'] = ds['CHLA'].values
            
            # Quality flags
            quality_flags = {}
            for param in ['PRES', 'TEMP', 'PSAL', 'DOXY', 'NITRATE', 'PH_IN_SITU_TOTAL', 'CHLA']:
                qc_var = f"{param}_QC"
                if qc_var in ds.variables:
                    quality_flags[param] = ds[qc_var].values
            
            # Build measurements list
            for i in range(n_levels):
                measurement = {}
                
                # Extract values for this level
                for var_name, var_data in variables.items():
                    if i < len(var_data):
                        value = var_data[i]
                        if not np.isnan(value) and not np.isinf(value):
                            measurement[var_name] = float(value)
                        else:
                            measurement[var_name] = None
                    else:
                        measurement[var_name] = None
                
                # Calculate depth from pressure (approximation)
                if measurement.get('pressure') is not None:
                    # Simple approximation: depth ≈ pressure (in meters)
                    measurement['depth'] = measurement['pressure']
                
                # Set quality flag (use pressure QC as default)
                qc_flag = 1  # Default to good data
                if 'PRES' in quality_flags and i < len(quality_flags['PRES']):
                    try:
                        qc_flag = int(quality_flags['PRES'][i])
                    except (ValueError, TypeError):
                        qc_flag = 1
                
                measurement['quality_flag'] = qc_flag
                
                # Validate measurement
                if validate_measurement_data(measurement):
                    measurements.append(measurement)
            
            logger.info(f"Extracted {len(measurements)} valid measurements")
            return measurements
            
        except Exception as e:
            logger.error(f"Failed to extract measurements: {str(e)}")
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
