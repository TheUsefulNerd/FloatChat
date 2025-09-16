import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransformer:
    """
    Transform and clean ARGO data for analysis and visualization
    """
    
    def __init__(self):
        self.parameter_ranges = {
            'temperature': (-5, 50),  # Celsius
            'salinity': (0, 50),      # PSU
            'pressure': (0, 10000),   # dbar
            'depth': (0, 10000),      # meters
            'oxygen': (0, 500),       # micromole/kg
            'nitrate': (0, 100),      # micromole/kg
            'ph': (6, 9),             # pH units
            'chlorophyll': (0, 100)   # mg/m3
        }
    
    def clean_measurements(self, measurements_df: pd.DataFrame) -> pd.DataFrame:
        """Clean measurement data by removing outliers and invalid values"""
        try:
            cleaned_df = measurements_df.copy()
            
            # Remove rows where all measurement values are null
            measurement_cols = ['temperature', 'salinity', 'pressure', 'oxygen', 'nitrate', 'ph', 'chlorophyll']
            available_cols = [col for col in measurement_cols if col in cleaned_df.columns]
            
            if available_cols:
                cleaned_df = cleaned_df.dropna(subset=available_cols, how='all')
            
            # Apply range filters
            for param, (min_val, max_val) in self.parameter_ranges.items():
                if param in cleaned_df.columns:
                    # Mark values outside range as invalid
                    invalid_mask = (cleaned_df[param] < min_val) | (cleaned_df[param] > max_val)
                    cleaned_df.loc[invalid_mask, param] = np.nan
                    
                    # Update quality flag for invalid values
                    if 'quality_flag' in cleaned_df.columns:
                        cleaned_df.loc[invalid_mask, 'quality_flag'] = 4  # Bad data
            
            # Remove duplicate depth levels
            if 'depth' in cleaned_df.columns:
                cleaned_df = cleaned_df.drop_duplicates(subset=['depth'], keep='first')
            
            # Sort by depth
            if 'depth' in cleaned_df.columns:
                cleaned_df = cleaned_df.sort_values('depth')
            elif 'pressure' in cleaned_df.columns:
                cleaned_df = cleaned_df.sort_values('pressure')
            
            logger.info(f"Cleaned measurements: {len(measurements_df)} -> {len(cleaned_df)} records")
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Failed to clean measurements: {str(e)}")
            return measurements_df
    
    def interpolate_missing_depth(self, measurements_df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing depth values from pressure"""
        try:
            df = measurements_df.copy()
            
            if 'depth' not in df.columns and 'pressure' in df.columns:
                # Simple approximation: depth ≈ pressure
                df['depth'] = df['pressure']
            elif 'depth' in df.columns and 'pressure' in df.columns:
                # Fill missing depth values
                mask = df['depth'].isna() & df['pressure'].notna()
                df.loc[mask, 'depth'] = df.loc[mask, 'pressure']
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to interpolate depth: {str(e)}")
            return measurements_df
    
    def calculate_derived_parameters(self, measurements_df: pd.DataFrame, profile_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Calculate derived oceanographic parameters"""
        try:
            df = measurements_df.copy()
            
            # Calculate potential temperature (simplified)
            if 'temperature' in df.columns and 'pressure' in df.columns:
                # Very simplified potential temperature calculation
                # Real calculation would use full thermodynamic equations
                df['potential_temperature'] = df['temperature'] - (df['pressure'] / 1000) * 0.1
            
            # Calculate density (simplified)
            if 'temperature' in df.columns and 'salinity' in df.columns:
                # Simplified density calculation (real would use EOS-80 or TEOS-10)
                df['density'] = 1000 + (df['salinity'] * 0.8) - (df['temperature'] * 0.2)
            
            # Calculate mixed layer depth (simplified)
            if 'temperature' in df.columns and 'depth' in df.columns:
                surface_temp = df[df['depth'] <= 10]['temperature'].mean()
                if not np.isnan(surface_temp):
                    temp_diff = np.abs(df['temperature'] - surface_temp)
                    mld_idx = np.where(temp_diff > 0.2)[0]  # 0.2°C threshold
                    if len(mld_idx) > 0:
                        df['mixed_layer_depth'] = df.iloc[mld_idx[0]]['depth']
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate derived parameters: {str(e)}")
            return measurements_df
    
    def create_profile_summary(self, measurements_df: pd.DataFrame, profile_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the profile for vector database storage"""
        try:
            summary = profile_metadata.copy()
            
            if measurements_df.empty:
                summary['summary_text'] = "Empty profile with no measurements"
                return summary
            
            # Basic statistics
            stats = {}
            for param in ['temperature', 'salinity', 'pressure', 'depth', 'oxygen']:
                if param in measurements_df.columns:
                    param_data = measurements_df[param].dropna()
                    if not param_data.empty:
                        stats[param] = {
                            'min': float(param_data.min()),
                            'max': float(param_data.max()),
                            'mean': float(param_data.mean()),
                            'std': float(param_data.std())
                        }
            
            summary['statistics'] = stats
            
            # Depth range
            if 'depth' in measurements_df.columns:
                depth_data = measurements_df['depth'].dropna()
                if not depth_data.empty:
                    summary['depth_range'] = {
                        'min_depth': float(depth_data.min()),
                        'max_depth': float(depth_data.max())
                    }
            
            # Data quality assessment
            total_measurements = len(measurements_df)
            good_quality = len(measurements_df[measurements_df.get('quality_flag', 1) <= 2])
            summary['data_quality'] = {
                'total_measurements': total_measurements,
                'good_quality_measurements': good_quality,
                'quality_percentage': (good_quality / total_measurements * 100) if total_measurements > 0 else 0
            }
            
            # Create descriptive text for vector search
            text_parts = []
            
            # Location description
            lat = summary.get('latitude', 0)
            lon = summary.get('longitude', 0)
            text_parts.append(f"ARGO float {summary.get('float_id', 'unknown')} profile at {lat:.2f}°N, {lon:.2f}°E")
            
            # Date description
            if 'measurement_date' in summary:
                date = summary['measurement_date']
                if isinstance(date, str):
                    date = pd.to_datetime(date)
                text_parts.append(f"measured on {date.strftime('%Y-%m-%d')}")
            
            # Depth description
            if 'depth_range' in summary:
                depth_range = summary['depth_range']
                text_parts.append(f"depth range {depth_range['min_depth']:.1f}m to {depth_range['max_depth']:.1f}m")
            
            # Parameter descriptions
            if 'temperature' in stats:
                temp_stats = stats['temperature']
                text_parts.append(f"temperature {temp_stats['min']:.2f}°C to {temp_stats['max']:.2f}°C")
            
            if 'salinity' in stats:
                sal_stats = stats['salinity']
                text_parts.append(f"salinity {sal_stats['min']:.2f} to {sal_stats['max']:.2f} PSU")
            
            if 'oxygen' in stats:
                oxy_stats = stats['oxygen']
                text_parts.append(f"oxygen {oxy_stats['min']:.1f} to {oxy_stats['max']:.1f} μmol/kg")
            
            summary['summary_text'] = ". ".join(text_parts) + "."
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to create profile summary: {str(e)}")
            return profile_metadata
    
    def aggregate_profiles_by_region(self, profiles_df: pd.DataFrame, grid_size: float = 1.0) -> pd.DataFrame:
        """Aggregate profiles by geographic grid"""
        try:
            if profiles_df.empty:
                return pd.DataFrame()
            
            # Create grid coordinates
            profiles_df = profiles_df.copy()
            profiles_df['lat_grid'] = np.floor(profiles_df['latitude'] / grid_size) * grid_size
            profiles_df['lon_grid'] = np.floor(profiles_df['longitude'] / grid_size) * grid_size
            
            # Aggregate by grid cell
            aggregated = profiles_df.groupby(['lat_grid', 'lon_grid']).agg({
                'id': 'count',
                'float_id': 'nunique',
                'measurement_date': ['min', 'max'],
                'latitude': 'mean',
                'longitude': 'mean'
            }).reset_index()
            
            # Flatten column names
            aggregated.columns = [
                'lat_grid', 'lon_grid', 'profile_count', 'unique_floats',
                'earliest_date', 'latest_date', 'mean_latitude', 'mean_longitude'
            ]
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Failed to aggregate profiles by region: {str(e)}")
            return pd.DataFrame()
    
    def create_time_series(self, measurements_df: pd.DataFrame, parameter: str, depth_levels: List[float] = None) -> pd.DataFrame:
        """Create time series data for a specific parameter at different depth levels"""
        try:
            if measurements_df.empty or parameter not in measurements_df.columns:
                return pd.DataFrame()
            
            if depth_levels is None:
                depth_levels = [10, 50, 100, 200, 500, 1000]  # Standard depth levels
            
            # Interpolate parameter values at standard depth levels
            time_series_data = []
            
            for depth in depth_levels:
                # Find closest measurements to target depth
                if 'depth' in measurements_df.columns:
                    depth_diff = np.abs(measurements_df['depth'] - depth)
                    closest_idx = depth_diff.idxmin()
                    
                    if depth_diff.iloc[closest_idx] <= 50:  # Within 50m tolerance
                        value = measurements_df.loc[closest_idx, parameter]
                        if not np.isnan(value):
                            time_series_data.append({
                                'depth_level': depth,
                                'value': value,
                                'actual_depth': measurements_df.loc[closest_idx, 'depth']
                            })
            
            return pd.DataFrame(time_series_data)
            
        except Exception as e:
            logger.error(f"Failed to create time series: {str(e)}")
            return pd.DataFrame()
    
    def detect_anomalies(self, measurements_df: pd.DataFrame, parameter: str) -> pd.DataFrame:
        """Detect anomalies in measurement data using statistical methods"""
        try:
            if measurements_df.empty or parameter not in measurements_df.columns:
                return measurements_df
            
            df = measurements_df.copy()
            param_data = df[parameter].dropna()
            
            if len(param_data) < 10:  # Need sufficient data for anomaly detection
                df['anomaly_flag'] = False
                return df
            
            # Z-score method
            mean_val = param_data.mean()
            std_val = param_data.std()
            z_scores = np.abs((df[parameter] - mean_val) / std_val)
            
            # IQR method
            q1 = param_data.quantile(0.25)
            q3 = param_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Combine both methods
            z_anomalies = z_scores > 3  # Z-score > 3
            iqr_anomalies = (df[parameter] < lower_bound) | (df[parameter] > upper_bound)
            
            df['anomaly_flag'] = z_anomalies | iqr_anomalies
            df['z_score'] = z_scores
            
            logger.info(f"Detected {df['anomaly_flag'].sum()} anomalies in {parameter}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {str(e)}")
            return measurements_df
