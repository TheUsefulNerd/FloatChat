import pandas as pd
import numpy as np
import base64
import io
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, date
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_data_for_display(df: pd.DataFrame, show_coordinates: bool = True, 
                          show_metadata: bool = False) -> pd.DataFrame:
    """
    Format DataFrame for display in Streamlit with proper column names and formatting
    """
    try:
        if df.empty:
            return df
        
        display_df = df.copy()
        
        # Rename columns to be more user-friendly
        column_mapping = {
            'id': 'ID',
            'float_id': 'Float ID',
            'cycle_number': 'Cycle',
            'latitude': 'Latitude',
            'longitude': 'Longitude',
            'measurement_date': 'Date',
            'platform_number': 'Platform',
            'data_center': 'Data Center',
            'created_at': 'Created'
        }
        
        display_df = display_df.rename(columns=column_mapping)
        
        # Format numeric columns
        if 'Latitude' in display_df.columns:
            display_df['Latitude'] = display_df['Latitude'].round(4)
        
        if 'Longitude' in display_df.columns:
            display_df['Longitude'] = display_df['Longitude'].round(4)
        
        # Format date columns
        date_columns = ['Date', 'Created']
        for col in date_columns:
            if col in display_df.columns:
                display_df[col] = pd.to_datetime(display_df[col]).dt.strftime('%Y-%m-%d %H:%M')
        
        # Select columns to display
        essential_columns = ['ID', 'Float ID', 'Cycle', 'Date']
        
        if show_coordinates:
            essential_columns.extend(['Latitude', 'Longitude'])
        
        if show_metadata:
            metadata_columns = ['Platform', 'Data Center', 'Created']
            essential_columns.extend([col for col in metadata_columns if col in display_df.columns])
        
        # Filter to available columns
        display_columns = [col for col in essential_columns if col in display_df.columns]
        
        if display_columns:
            display_df = display_df[display_columns]
        
        return display_df
        
    except Exception as e:
        logger.error(f"Failed to format data for display: {str(e)}")
        return df

def create_download_link(df: pd.DataFrame, file_format: str, filename: str = None) -> str:
    """
    Create a download link for DataFrame in specified format
    """
    try:
        if df.empty:
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"argo_data_{timestamp}"
        
        # Remove file extension if provided
        if '.' in filename:
            filename = filename.split('.')[0]
        
        if file_format.upper() == 'CSV':
            output = io.StringIO()
            df.to_csv(output, index=False)
            data = output.getvalue()
            b64 = base64.b64encode(data.encode()).decode()
            href = f'<a href="data:text/csv;base64,{b64}" download="{filename}.csv">Download CSV file</a>'
            
        elif file_format.upper() == 'PARQUET':
            output = io.BytesIO()
            df.to_parquet(output, index=False)
            data = output.getvalue()
            b64 = base64.b64encode(data).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}.parquet">Download Parquet file</a>'
            
        elif file_format.upper() == 'JSON':
            data = df.to_json(orient='records', date_format='iso')
            b64 = base64.b64encode(data.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="{filename}.json">Download JSON file</a>'
            
        elif file_format.upper() == 'EXCEL':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='ARGO_Data')
            data = output.getvalue()
            b64 = base64.b64encode(data).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download Excel file</a>'
            
        else:
            logger.error(f"Unsupported file format: {file_format}")
            return None
        
        return href
        
    except Exception as e:
        logger.error(f"Failed to create download link: {str(e)}")
        return None

def validate_coordinates(latitude: float, longitude: float) -> Tuple[bool, str]:
    """
    Validate latitude and longitude coordinates
    """
    try:
        # Check latitude range
        if not -90 <= latitude <= 90:
            return False, f"Latitude {latitude} is out of range [-90, 90]"
        
        # Check longitude range
        if not -180 <= longitude <= 180:
            return False, f"Longitude {longitude} is out of range [-180, 180]"
        
        return True, "Valid coordinates"
        
    except Exception as e:
        return False, f"Error validating coordinates: {str(e)}"

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using Haversine formula
    Returns distance in kilometers
    """
    try:
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return c * r
        
    except Exception as e:
        logger.error(f"Failed to calculate distance: {str(e)}")
        return np.nan

def format_parameter_value(value: Any, parameter: str, precision: int = 2) -> str:
    """
    Format parameter values with appropriate units and precision
    """
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        
        # Parameter units mapping
        units = {
            'temperature': '°C',
            'salinity': 'PSU',
            'pressure': 'dbar',
            'depth': 'm',
            'oxygen': 'μmol/kg',
            'nitrate': 'μmol/kg',
            'ph': '',
            'chlorophyll': 'mg/m³'
        }
        
        unit = units.get(parameter.lower(), '')
        
        # Format based on parameter type
        if parameter.lower() == 'ph':
            return f"{float(value):.2f}"
        else:
            formatted_value = f"{float(value):.{precision}f}"
            return f"{formatted_value} {unit}".strip()
            
    except Exception as e:
        logger.error(f"Failed to format parameter value: {str(e)}")
        return str(value)

def get_parameter_info(parameter: str) -> Dict[str, str]:
    """
    Get detailed information about an oceanographic parameter
    """
    parameter_info = {
        'temperature': {
            'name': 'Temperature',
            'units': '°C',
            'description': 'Sea water temperature',
            'typical_range': '-2 to 30°C',
            'measurement_method': 'CTD sensor or thermistor'
        },
        'salinity': {
            'name': 'Salinity',
            'units': 'PSU',
            'description': 'Practical salinity of seawater',
            'typical_range': '32 to 37 PSU',
            'measurement_method': 'Conductivity sensor'
        },
        'pressure': {
            'name': 'Pressure',
            'units': 'dbar',
            'description': 'Water pressure (approximately equal to depth in meters)',
            'typical_range': '0 to 6000 dbar',
            'measurement_method': 'Pressure sensor'
        },
        'depth': {
            'name': 'Depth',
            'units': 'm',
            'description': 'Depth below sea surface',
            'typical_range': '0 to 6000 m',
            'measurement_method': 'Calculated from pressure'
        },
        'oxygen': {
            'name': 'Dissolved Oxygen',
            'units': 'μmol/kg',
            'description': 'Concentration of dissolved oxygen in seawater',
            'typical_range': '0 to 400 μmol/kg',
            'measurement_method': 'Optical oxygen sensor'
        },
        'nitrate': {
            'name': 'Nitrate',
            'units': 'μmol/kg',
            'description': 'Nitrate concentration in seawater',
            'typical_range': '0 to 50 μmol/kg',
            'measurement_method': 'Chemical analysis or optical sensor'
        },
        'ph': {
            'name': 'pH',
            'units': '',
            'description': 'Acidity/alkalinity of seawater',
            'typical_range': '7.5 to 8.3',
            'measurement_method': 'pH sensor'
        },
        'chlorophyll': {
            'name': 'Chlorophyll-a',
            'units': 'mg/m³',
            'description': 'Chlorophyll-a concentration indicating phytoplankton biomass',
            'typical_range': '0 to 20 mg/m³',
            'measurement_method': 'Fluorescence sensor'
        }
    }
    
    return parameter_info.get(parameter.lower(), {
        'name': parameter.title(),
        'units': 'Unknown',
        'description': 'No description available',
        'typical_range': 'Unknown',
        'measurement_method': 'Unknown'
    })

def convert_julian_day(julian_day: float, reference_date: str = '1950-01-01') -> datetime:
    """
    Convert ARGO Julian day to datetime
    ARGO uses days since 1950-01-01
    """
    try:
        if pd.isna(julian_day):
            return None
        
        reference = pd.to_datetime(reference_date)
        converted_date = reference + pd.Timedelta(days=julian_day)
        
        return converted_date
        
    except Exception as e:
        logger.error(f"Failed to convert Julian day: {str(e)}")
        return None

def create_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a comprehensive summary of a DataFrame
    """
    try:
        if df.empty:
            return {'error': 'Empty DataFrame'}
        
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'column_info': {},
            'missing_data': {},
            'data_types': df.dtypes.to_dict()
        }
        
        # Column information
        for col in df.columns:
            col_info = {
                'type': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique()
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                })
            
            summary['column_info'][col] = col_info
        
        # Missing data summary
        missing_counts = df.isnull().sum()
        summary['missing_data'] = {
            col: {
                'count': int(count),
                'percentage': float(count / len(df) * 100)
            }
            for col, count in missing_counts.items() if count > 0
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to create data summary: {str(e)}")
        return {'error': str(e)}

def validate_netcdf_structure(file_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate NetCDF file structure for ARGO compatibility
    """
    try:
        import xarray as xr
        
        with xr.open_dataset(file_path) as ds:
            validation_result = {
                'dimensions': dict(ds.dims),
                'variables': list(ds.variables.keys()),
                'global_attributes': dict(ds.attrs),
                'coordinate_variables': [],
                'data_variables': [],
                'required_variables': [],
                'missing_variables': []
            }
            
            # Check for coordinate variables
            for var in ds.variables:
                if var in ds.dims:
                    validation_result['coordinate_variables'].append(var)
                else:
                    validation_result['data_variables'].append(var)
            
            # Check for required ARGO variables
            required_vars = ['PRES', 'TEMP', 'PSAL', 'LATITUDE', 'LONGITUDE', 'JULD']
            for var in required_vars:
                if var in ds.variables:
                    validation_result['required_variables'].append(var)
                else:
                    validation_result['missing_variables'].append(var)
            
            # Determine if valid
            is_valid = len(validation_result['missing_variables']) == 0
            
            if is_valid:
                message = "Valid ARGO NetCDF file"
            else:
                message = f"Missing required variables: {', '.join(validation_result['missing_variables'])}"
            
            return is_valid, message, validation_result
            
    except Exception as e:
        logger.error(f"Failed to validate NetCDF structure: {str(e)}")
        return False, f"Validation error: {str(e)}", {}

def create_quality_control_report(measurements_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a quality control report for measurement data
    """
    try:
        if measurements_df.empty:
            return {'error': 'Empty measurements DataFrame'}
        
        report = {
            'total_measurements': len(measurements_df),
            'parameter_quality': {},
            'depth_coverage': {},
            'outlier_analysis': {},
            'completeness': {}
        }
        
        # Parameter quality analysis
        parameters = ['temperature', 'salinity', 'pressure', 'oxygen', 'nitrate', 'ph', 'chlorophyll']
        
        for param in parameters:
            if param in measurements_df.columns:
                param_data = measurements_df[param].dropna()
                
                if not param_data.empty:
                    # Basic statistics
                    param_quality = {
                        'total_values': len(measurements_df),
                        'valid_values': len(param_data),
                        'missing_values': len(measurements_df) - len(param_data),
                        'completeness_percent': (len(param_data) / len(measurements_df)) * 100,
                        'min_value': param_data.min(),
                        'max_value': param_data.max(),
                        'mean_value': param_data.mean(),
                        'std_value': param_data.std()
                    }
                    
                    # Outlier detection using IQR
                    q1 = param_data.quantile(0.25)
                    q3 = param_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = param_data[(param_data < lower_bound) | (param_data > upper_bound)]
                    param_quality['outlier_count'] = len(outliers)
                    param_quality['outlier_percent'] = (len(outliers) / len(param_data)) * 100
                    
                    report['parameter_quality'][param] = param_quality
        
        # Depth coverage analysis
        if 'depth' in measurements_df.columns:
            depth_data = measurements_df['depth'].dropna()
            if not depth_data.empty:
                report['depth_coverage'] = {
                    'min_depth': depth_data.min(),
                    'max_depth': depth_data.max(),
                    'depth_range': depth_data.max() - depth_data.min(),
                    'measurements_per_100m': len(depth_data) / ((depth_data.max() - depth_data.min()) / 100)
                }
        
        # Quality flag analysis
        if 'quality_flag' in measurements_df.columns:
            quality_flags = measurements_df['quality_flag'].value_counts()
            report['quality_flags'] = {
                'flag_distribution': quality_flags.to_dict(),
                'good_quality_percent': (quality_flags.get(1, 0) / len(measurements_df)) * 100,
                'questionable_quality_percent': (quality_flags.get(2, 0) / len(measurements_df)) * 100,
                'bad_quality_percent': (quality_flags.get(4, 0) / len(measurements_df)) * 100
            }
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to create quality control report: {str(e)}")
        return {'error': str(e)}

def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human readable format
    """
    try:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"
    except:
        return "Unknown size"

def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default value if division by zero
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate string to specified length with suffix
    """
    try:
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    except:
        return str(text)
