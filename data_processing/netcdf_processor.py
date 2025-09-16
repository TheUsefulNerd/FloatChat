import xarray as xr
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import hashlib
import logging
from datetime import datetime
import os

# Conditional import for validation function
try:
    from database.schema import validate_measurement_data
except ImportError:
    def validate_measurement_data(measurement):
        """Fallback validation function"""
        return True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetCDFProcessor:
    """
    Process any NetCDF files and extract structured data
    Enhanced to handle various NetCDF formats, not just ARGO
    """
    
    def __init__(self, mode: str = "flexible"):
        self.supported_formats = ['.nc', '.netcdf', '.nc4']
        self.mode = mode  # "argo", "flexible", or "auto"
        
        # ARGO-specific configuration
        self.argo_required_variables = ['PRES', 'TEMP', 'PSAL']
        self.argo_optional_variables = ['DOXY', 'NITRATE', 'PH_IN_SITU_TOTAL', 'CHLA']
        
        # Common variable name mappings for different data sources
        self.variable_mappings = {
            # Temperature variants
            'temperature': ['TEMP', 'TEMP_ADJUSTED', 'temp_adjusted', 'temp', 'temperature', 'T', 'sea_water_temperature', 'TEMPERATURE'],
            # Pressure/Depth variants
            'pressure': ['PRES', 'PRES_ADJUSTED', 'pres_adjusted', 'pres', 'pressure', 'P', 'sea_water_pressure', 'PRESSURE'],
            'depth': ['DEPTH', 'depth', 'z', 'Z', 'level', 'LEVEL'],
            # Salinity variants
            'salinity': ['PSAL', 'PSAL_ADJUSTED', 'psal_adjusted', 'psal', 'salinity', 'salt', 'S', 'sea_water_salinity', 'SALINITY'],
            # Oxygen variants
            'oxygen': ['DOXY', 'DOXY_ADJUSTED', 'doxy_adjusted', 'oxygen', 'o2', 'O2', 'dissolved_oxygen', 'OXYGEN'],
            # Coordinate variants
            'latitude': ['LATITUDE', 'latitude', 'lat', 'LAT', 'y', 'Y'],
            'longitude': ['LONGITUDE', 'longitude', 'lon', 'LON', 'x', 'X'],
            'time': ['JULD', 'time', 'TIME', 't', 'T', 'date', 'DATE']
        }
        
    def detect_file_type(self, ds: xr.Dataset) -> str:
        """Detect the type of NetCDF file (ARGO, general oceanographic, etc.)"""
        try:
            # Check for ARGO-specific variables
            argo_vars = sum(1 for var in self.argo_required_variables if var in ds.variables)
            if argo_vars >= 2:  # At least 2 out of 3 ARGO variables
                return "argo"
            
            # Check for common oceanographic variables
            ocean_indicators = ['sea_water', 'ocean', 'marine', 'float', 'profile']
            has_ocean_vars = any(
                any(indicator in str(var).lower() for indicator in ocean_indicators)
                or any(indicator in str(ds[var].attrs.get('long_name', '')).lower() for indicator in ocean_indicators)
                for var in ds.variables
            )
            
            if has_ocean_vars:
                return "oceanographic"
            
            return "general"
            
        except Exception as e:
            logger.warning(f"Could not detect file type: {str(e)}")
            return "general"
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if file is a valid NetCDF file (more flexible validation)"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return False
                
            # Check file extension
            _, ext = os.path.splitext(file_path)
            if ext.lower() not in self.supported_formats:
                logger.warning(f"Unusual file extension: {ext}")
                # Don't fail here, try to open anyway
            
            # Try to open with xarray
            with xr.open_dataset(file_path) as ds:
                # Basic checks
                if len(ds.variables) == 0:
                    logger.error(f"File {file_path} contains no variables")
                    return False
                
                # Detect file type
                file_type = self.detect_file_type(ds)
                logger.info(f"Detected file type: {file_type}")
                
                # Mode-specific validation
                if self.mode == "argo" and file_type != "argo":
                    if not any(var in ds.variables for var in self.argo_required_variables):
                        logger.warning(f"File {file_path} does not contain ARGO variables in ARGO mode")
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"File validation failed for {file_path}: {str(e)}")
            return False
    
    def find_variable(self, ds: xr.Dataset, var_type: str) -> Optional[str]:
        """Find a variable in the dataset using multiple possible names"""
        if var_type not in self.variable_mappings:
            return None
            
        possible_names = self.variable_mappings[var_type]
        
        # Direct name match
        for name in possible_names:
            if name in ds.variables:
                return name
        
        # Case-insensitive match
        for name in possible_names:
            for var_name in ds.variables:
                if name.lower() == var_name.lower():
                    return var_name
        
        # Partial match in variable names
        for name in possible_names:
            for var_name in ds.variables:
                if name.lower() in var_name.lower():
                    return var_name
        
        # Check long_name and standard_name attributes
        for var_name in ds.variables:
            var = ds[var_name]
            long_name = str(var.attrs.get('long_name', '')).lower()
            standard_name = str(var.attrs.get('standard_name', '')).lower()
            
            for name in possible_names:
                if name.lower() in long_name or name.lower() in standard_name:
                    return var_name
        
        return None
    
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
    
    def safe_extract_value(self, var_data, index: int = 0, default=None):
        """Safely extract a value from numpy array/scalar"""
        try:
            if hasattr(var_data, 'item'):
                return var_data.item()
            elif hasattr(var_data, '__len__') and len(var_data) > index:
                return var_data[index]
            elif np.isscalar(var_data):
                return var_data
            else:
                return default
        except (IndexError, TypeError, ValueError):
            return default
    
    def convert_time_to_datetime(self, time_var, time_value) -> datetime:
        """Convert various time formats to datetime"""
        try:
            # Check for time units in attributes
            units = time_var.attrs.get('units', '')
            
            if 'days since 1950' in units.lower():
                # ARGO format
                base_date = pd.Timestamp('1950-01-01')
                return (base_date + pd.Timedelta(days=float(time_value))).to_pydatetime()
            elif 'seconds since' in units.lower():
                # Parse the base date from units
                import re
                match = re.search(r'since\s+(\d{4}-\d{2}-\d{2})', units)
                if match:
                    base_date = pd.Timestamp(match.group(1))
                    return (base_date + pd.Timedelta(seconds=float(time_value))).to_pydatetime()
            elif 'hours since' in units.lower():
                match = re.search(r'since\s+(\d{4}-\d{2}-\d{2})', units)
                if match:
                    base_date = pd.Timestamp(match.group(1))
                    return (base_date + pd.Timedelta(hours=float(time_value))).to_pydatetime()
            
            # Try to parse as pandas timestamp
            return pd.Timestamp(time_value).to_pydatetime()
            
        except Exception as e:
            logger.warning(f"Could not convert time value {time_value}: {str(e)}")
            return datetime.now()
    
    def extract_profile_metadata(self, ds: xr.Dataset) -> Dict[str, Any]:
        """Extract profile-level metadata from any NetCDF dataset"""
        try:
            metadata = {}
            file_type = self.detect_file_type(ds)
            metadata['file_type'] = file_type
            
            # Platform/Station information
            platform_vars = ['PLATFORM_NUMBER', 'platform_number', 'station', 'STATION', 'id', 'ID']
            for var_name in platform_vars:
                if var_name in ds.variables:
                    value = self.safe_extract_value(ds[var_name].values)
                    if value is not None:
                        metadata['platform_number'] = str(value)
                        break
            
            # Try global attributes for platform info
            if 'platform_number' not in metadata:
                for attr_name in ['platform_number', 'station_id', 'id']:
                    if attr_name in ds.attrs:
                        metadata['platform_number'] = str(ds.attrs[attr_name])
                        break
            
            metadata['float_id'] = metadata.get('platform_number', 'unknown')
            
            # Cycle number
            cycle_vars = ['CYCLE_NUMBER', 'cycle_number', 'profile', 'PROFILE']
            for var_name in cycle_vars:
                if var_name in ds.variables:
                    value = self.safe_extract_value(ds[var_name].values)
                    if value is not None:
                        try:
                            metadata['cycle_number'] = int(float(value))
                            break
                        except (ValueError, TypeError):
                            continue
            
            if 'cycle_number' not in metadata:
                metadata['cycle_number'] = 0
            
            # Location - try multiple approaches
            lat_var = self.find_variable(ds, 'latitude')
            if lat_var:
                lat_value = self.safe_extract_value(ds[lat_var].values)
                if lat_value is not None and not np.isnan(float(lat_value)):
                    metadata['latitude'] = float(lat_value)
            
            # Try global attributes if variable not found
            if 'latitude' not in metadata:
                for attr_name in ['latitude', 'geospatial_lat_min', 'lat']:
                    if attr_name in ds.attrs:
                        try:
                            metadata['latitude'] = float(ds.attrs[attr_name])
                            break
                        except (ValueError, TypeError):
                            continue
            
            # Set default if still not found
            if 'latitude' not in metadata:
                logger.warning("Latitude not found, using default value 0.0")
                metadata['latitude'] = 0.0
            
            lon_var = self.find_variable(ds, 'longitude')
            if lon_var:
                lon_value = self.safe_extract_value(ds[lon_var].values)
                if lon_value is not None and not np.isnan(float(lon_value)):
                    metadata['longitude'] = float(lon_value)
            
            # Try global attributes if variable not found
            if 'longitude' not in metadata:
                for attr_name in ['longitude', 'geospatial_lon_min', 'lon']:
                    if attr_name in ds.attrs:
                        try:
                            metadata['longitude'] = float(ds.attrs[attr_name])
                            break
                        except (ValueError, TypeError):
                            continue
            
            # Set default if still not found
            if 'longitude' not in metadata:
                logger.warning("Longitude not found, using default value 0.0")
                metadata['longitude'] = 0.0
            
            # Time
            time_var = self.find_variable(ds, 'time')
            if time_var:
                time_value = self.safe_extract_value(ds[time_var].values)
                if time_value is not None and not np.isnan(float(time_value)):
                    metadata['measurement_date'] = self.convert_time_to_datetime(ds[time_var], time_value)
                else:
                    metadata['measurement_date'] = datetime.now()
            else:
                metadata['measurement_date'] = datetime.now()
            
            # Data center/source
            dc_vars = ['DATA_CENTRE', 'DATA_CENTER', 'data_center', 'source', 'SOURCE']
            for var_name in dc_vars:
                if var_name in ds.variables:
                    value = self.safe_extract_value(ds[var_name].values)
                    if value is not None:
                        metadata['data_center'] = str(value)
                        break
            
            # Try global attributes
            if 'data_center' not in metadata:
                for attr_name in ['data_center', 'source', 'institution']:
                    if attr_name in ds.attrs:
                        metadata['data_center'] = str(ds.attrs[attr_name])
                        break
            
            if 'data_center' not in metadata:
                metadata['data_center'] = 'unknown'
            
            # Add dataset dimensions info - use ds.sizes instead of ds.dims
            metadata['dimensions'] = dict(ds.sizes)
            metadata['variables'] = list(ds.variables.keys())
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract profile metadata: {str(e)}")
            return {
                'file_type': 'unknown', 
                'data_center': 'unknown', 
                'cycle_number': 0,
                'latitude': 0.0,
                'longitude': 0.0,
                'measurement_date': datetime.now()
            }
    
    def extract_measurements(self, ds: xr.Dataset) -> List[Dict[str, Any]]:
        """Extract measurement data from any NetCDF dataset"""
        try:
            measurements = []
            
            # Find the main data dimension (levels, time, depth, etc.)
            main_dims = ['N_LEVELS', 'n_levels', 'depth', 'time', 'level', 'z']
            main_dim = None
            n_levels = 0
            
            for dim_name in main_dims:
                if dim_name in ds.sizes:
                    main_dim = dim_name
                    n_levels = ds.sizes[dim_name]
                    break
            
            # If no standard dimension found, use the largest dimension
            if main_dim is None:
                dims_by_size = sorted(ds.sizes.items(), key=lambda x: x[1], reverse=True)
                if dims_by_size:
                    main_dim, n_levels = dims_by_size[0]
                    logger.info(f"Using dimension '{main_dim}' with {n_levels} points")
            
            if n_levels == 0:
                logger.error("Cannot determine measurement dimension")
                return []
            
            # Find available variables
            variables = {}
            
            # Core oceanographic variables
            var_mappings = {
                'pressure': self.find_variable(ds, 'pressure'),
                'depth': self.find_variable(ds, 'depth'),
                'temperature': self.find_variable(ds, 'temperature'),
                'salinity': self.find_variable(ds, 'salinity'),
                'oxygen': self.find_variable(ds, 'oxygen')
            }
            
            # Add any numeric variables that vary along the main dimension
            for var_name in ds.variables:
                var = ds[var_name]
                if (main_dim in var.dims and 
                    var.dtype.kind in 'fcui' and  # numeric types
                    var_name not in var_mappings.values()):
                    # Use original variable name
                    variables[var_name] = var_name
            
            # Add mapped variables
            for mapped_name, var_name in var_mappings.items():
                if var_name:
                    variables[mapped_name] = var_name
            
            if not variables:
                logger.error("No suitable measurement variables found")
                return []
            
            logger.info(f"Found variables: {list(variables.keys())}")
            
            # Extract data
            for i in range(n_levels):
                measurement = {}
                
                for var_name, nc_var_name in variables.items():
                    try:
                        if nc_var_name in ds.variables:
                            var_data = ds[nc_var_name].values
                            
                            # Handle different array structures
                            if var_data.ndim == 0:  # scalar
                                value = float(var_data)
                            elif var_data.ndim == 1 and i < len(var_data):
                                value = var_data[i]
                            elif var_data.ndim > 1:
                                # For multidimensional arrays, try to extract along main dimension
                                if i < var_data.shape[0]:
                                    value = var_data[i] if var_data.ndim == 1 else var_data[i, 0]
                                else:
                                    continue
                            else:
                                continue
                            
                            # Validate value
                            if not np.isnan(value) and not np.isinf(value):
                                measurement[var_name] = float(value)
                            
                    except Exception as e:
                        logger.debug(f"Could not extract {var_name}: {str(e)}")
                        continue
                
                # Calculate depth from pressure if depth not available
                if 'depth' not in measurement and 'pressure' in measurement:
                    measurement['depth'] = measurement['pressure']
                
                # Set default quality flag
                measurement['quality_flag'] = 1
                
                # Only keep measurements with at least one valid value (fixed the variable reference bug)
                if any(v is not None for key, v in measurement.items() if key != 'quality_flag'):
                    # Validate against schema if available
                    try:
                        if validate_measurement_data(measurement):
                            measurements.append(measurement)
                    except:
                        # If validation function doesn't exist or fails, add anyway
                        measurements.append(measurement)
            
            logger.info(f"Extracted {len(measurements)} measurements")
            return measurements
            
        except Exception as e:
            logger.error(f"Failed to extract measurements: {str(e)}")
            return []
    
    def process_file(self, file_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process any NetCDF file and return profile metadata and measurements
        
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
                profile_metadata['file_path'] = file_path
                
                # Extract measurements
                measurements = self.extract_measurements(ds)
                
                logger.info(f"Successfully processed file: {file_path}")
                logger.info(f"File type: {profile_metadata.get('file_type', 'unknown')}")
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
        """Get a comprehensive summary of any NetCDF file"""
        try:
            if not self.validate_file(file_path):
                return {'error': 'Invalid NetCDF file'}
            
            with xr.open_dataset(file_path) as ds:
                summary = {
                    'file_path': file_path,
                    'file_size': os.path.getsize(file_path),
                    'dimensions': dict(ds.sizes),
                    'variables': {
                        var_name: {
                            'shape': ds[var_name].shape,
                            'dtype': str(ds[var_name].dtype),
                            'attributes': dict(ds[var_name].attrs)
                        }
                        for var_name in ds.variables
                    },
                    'global_attributes': dict(ds.attrs),
                }
                
                # Quick metadata extraction
                metadata = self.extract_profile_metadata(ds)
                summary.update(metadata)
                
                return summary
                
        except Exception as e:
            return {'error': f'Failed to read file: {str(e)}'}
    
    def set_mode(self, mode: str):
        """Change processing mode: 'argo', 'flexible', or 'auto'"""
        if mode in ['argo', 'flexible', 'auto']:
            self.mode = mode
            logger.info(f"Processing mode changed to: {mode}")
        else:
            logger.warning(f"Invalid mode: {mode}. Using 'flexible'")
            self.mode = 'flexible'