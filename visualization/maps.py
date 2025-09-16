import folium
from folium import plugins
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OceanographicMaps:
    """
    Create interactive maps for ARGO float data using Folium
    """
    
    def __init__(self):
        self.parameter_colors = {
            'temperature': 'RdYlBu_r',
            'salinity': 'viridis',
            'oxygen': 'Blues',
            'chlorophyll': 'Greens',
            'nitrate': 'Oranges',
            'ph': 'RdBu'
        }
        
        self.parameter_units = {
            'temperature': '°C',
            'salinity': 'PSU',
            'pressure': 'dbar',
            'depth': 'meters',
            'oxygen': 'μmol/kg',
            'nitrate': 'μmol/kg',
            'ph': 'pH units',
            'chlorophyll': 'mg/m³'
        }
    
    def create_float_trajectory_map(self, profiles_df: pd.DataFrame, 
                                  float_id: str = None) -> folium.Map:
        """Create map showing ARGO float trajectories"""
        try:
            if profiles_df.empty:
                return self._create_empty_map("No profile data available")
            
            # Filter for specific float if requested
            if float_id:
                float_data = profiles_df[profiles_df['float_id'] == float_id].copy()
                if float_data.empty:
                    return self._create_empty_map(f"No data found for float {float_id}")
            else:
                float_data = profiles_df.copy()
            
            # Remove rows with missing coordinates
            float_data = float_data.dropna(subset=['latitude', 'longitude'])
            
            if float_data.empty:
                return self._create_empty_map("No valid coordinates found")
            
            # Calculate map center
            center_lat = float_data['latitude'].mean()
            center_lon = float_data['longitude'].mean()
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=4,
                tiles='OpenStreetMap'
            )
            
            # Add different tile layers
            folium.TileLayer('CartoDB positron').add_to(m)
            folium.TileLayer('CartoDB dark_matter').add_to(m)
            
            # Group profiles by float_id
            float_groups = float_data.groupby('float_id')
            
            # Color palette for different floats
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                     'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
                     'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 
                     'gray', 'black', 'lightgray']
            
            for i, (fid, group) in enumerate(float_groups):
                color = colors[i % len(colors)]
                
                # Sort by date to create proper trajectory
                if 'measurement_date' in group.columns:
                    group = group.sort_values('measurement_date')
                
                # Create trajectory line
                coordinates = group[['latitude', 'longitude']].values.tolist()
                
                if len(coordinates) > 1:
                    folium.PolyLine(
                        coordinates,
                        color=color,
                        weight=2,
                        opacity=0.7,
                        popup=f"Float {fid} trajectory"
                    ).add_to(m)
                
                # Add markers for each profile
                for idx, row in group.iterrows():
                    popup_text = f"""
                    <b>Float {row['float_id']}</b><br>
                    Cycle: {row.get('cycle_number', 'N/A')}<br>
                    Lat: {row['latitude']:.3f}°N<br>
                    Lon: {row['longitude']:.3f}°E<br>
                    Date: {row.get('measurement_date', 'N/A')}
                    """
                    
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=5,
                        popup=popup_text,
                        color=color,
                        fillColor=color,
                        fillOpacity=0.7,
                        weight=2
                    ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Add measurement tool
            plugins.MeasureControl().add_to(m)
            
            # Add full screen option
            plugins.Fullscreen().add_to(m)
            
            return m
            
        except Exception as e:
            logger.error(f"Failed to create trajectory map: {str(e)}")
            return self._create_empty_map(f"Error creating trajectory map: {str(e)}")
    
    def create_parameter_map(self, profiles_df: pd.DataFrame, measurements_df: pd.DataFrame,
                           parameter: str, depth_range: Tuple[float, float] = None) -> folium.Map:
        """Create map showing parameter values at profile locations"""
        try:
            if profiles_df.empty or measurements_df.empty:
                return self._create_empty_map("Profile and measurement data required")
            
            if parameter not in measurements_df.columns:
                return self._create_empty_map(f"Parameter {parameter} not found in measurements")
            
            # Filter measurements by depth range if specified
            if depth_range:
                min_depth, max_depth = depth_range
                filtered_measurements = measurements_df[
                    (measurements_df['depth'] >= min_depth) & 
                    (measurements_df['depth'] <= max_depth)
                ].copy()
            else:
                filtered_measurements = measurements_df.copy()
            
            # Calculate mean parameter value for each profile
            param_means = filtered_measurements.groupby('profile_id')[parameter].mean().reset_index()
            param_means.columns = ['id', f'mean_{parameter}']
            
            # Merge with profile data
            map_data = profiles_df.merge(param_means, on='id', how='inner')
            map_data = map_data.dropna(subset=['latitude', 'longitude', f'mean_{parameter}'])
            
            if map_data.empty:
                return self._create_empty_map(f"No valid {parameter} data for mapping")
            
            # Calculate map center
            center_lat = map_data['latitude'].mean()
            center_lon = map_data['longitude'].mean()
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=4,
                tiles='OpenStreetMap'
            )
            
            # Add different tile layers
            folium.TileLayer('CartoDB positron').add_to(m)
            folium.TileLayer('CartoDB dark_matter').add_to(m)
            
            # Create color scale
            param_values = map_data[f'mean_{parameter}']
            vmin, vmax = param_values.min(), param_values.max()
            
            # Normalize values for color mapping
            if vmax > vmin:
                normalized_values = (param_values - vmin) / (vmax - vmin)
            else:
                normalized_values = np.ones(len(param_values)) * 0.5
            
            # Color map function
            def get_color(value):
                """Get color based on normalized value"""
                if np.isnan(value):
                    return 'gray'
                
                # Simple color gradient from blue to red
                red = int(255 * value)
                blue = int(255 * (1 - value))
                return f'#{red:02x}00{blue:02x}'
            
            # Add markers
            for idx, row in map_data.iterrows():
                param_value = row[f'mean_{parameter}']
                normalized_val = normalized_values.iloc[idx]
                
                popup_text = f"""
                <b>Float {row['float_id']}</b><br>
                Cycle: {row.get('cycle_number', 'N/A')}<br>
                Lat: {row['latitude']:.3f}°N<br>
                Lon: {row['longitude']:.3f}°E<br>
                {parameter.title()}: {param_value:.2f} {self.parameter_units.get(parameter, '')}<br>
                Date: {row.get('measurement_date', 'N/A')}
                """
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=8,
                    popup=popup_text,
                    color='black',
                    fillColor=get_color(normalized_val),
                    fillOpacity=0.8,
                    weight=1
                ).add_to(m)
            
            # Add color legend
            legend_html = self._create_color_legend(parameter, vmin, vmax)
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Add measurement tool
            plugins.MeasureControl().add_to(m)
            
            # Add full screen option
            plugins.Fullscreen().add_to(m)
            
            return m
            
        except Exception as e:
            logger.error(f"Failed to create parameter map: {str(e)}")
            return self._create_empty_map(f"Error creating parameter map: {str(e)}")
    
    def create_density_map(self, profiles_df: pd.DataFrame) -> folium.Map:
        """Create heatmap showing density of ARGO profiles"""
        try:
            if profiles_df.empty:
                return self._create_empty_map("No profile data available")
            
            # Remove rows with missing coordinates
            valid_coords = profiles_df.dropna(subset=['latitude', 'longitude'])
            
            if valid_coords.empty:
                return self._create_empty_map("No valid coordinates found")
            
            # Calculate map center
            center_lat = valid_coords['latitude'].mean()
            center_lon = valid_coords['longitude'].mean()
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=3,
                tiles='OpenStreetMap'
            )
            
            # Add different tile layers
            folium.TileLayer('CartoDB positron').add_to(m)
            folium.TileLayer('CartoDB dark_matter').add_to(m)
            
            # Prepare heat map data
            heat_data = valid_coords[['latitude', 'longitude']].values.tolist()
            
            # Add heat map layer
            plugins.HeatMap(
                heat_data,
                radius=15,
                blur=10,
                max_zoom=1,
                gradient={
                    0.0: 'blue',
                    0.3: 'cyan',
                    0.5: 'lime',
                    0.7: 'yellow',
                    1.0: 'red'
                }
            ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Add measurement tool
            plugins.MeasureControl().add_to(m)
            
            # Add full screen option
            plugins.Fullscreen().add_to(m)
            
            return m
            
        except Exception as e:
            logger.error(f"Failed to create density map: {str(e)}")
            return self._create_empty_map(f"Error creating density map: {str(e)}")
    
    def create_regional_map(self, profiles_df: pd.DataFrame, region_bounds: Dict[str, float]) -> folium.Map:
        """Create map focused on a specific region"""
        try:
            if profiles_df.empty:
                return self._create_empty_map("No profile data available")
            
            # Extract region bounds
            min_lat = region_bounds.get('min_lat', -90)
            max_lat = region_bounds.get('max_lat', 90)
            min_lon = region_bounds.get('min_lon', -180)
            max_lon = region_bounds.get('max_lon', 180)
            
            # Filter profiles within region
            regional_data = profiles_df[
                (profiles_df['latitude'] >= min_lat) &
                (profiles_df['latitude'] <= max_lat) &
                (profiles_df['longitude'] >= min_lon) &
                (profiles_df['longitude'] <= max_lon)
            ].copy()
            
            if regional_data.empty:
                return self._create_empty_map("No profiles found in specified region")
            
            # Calculate map center
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=5,
                tiles='OpenStreetMap'
            )
            
            # Add different tile layers
            folium.TileLayer('CartoDB positron').add_to(m)
            folium.TileLayer('CartoDB dark_matter').add_to(m)
            
            # Add region boundary rectangle
            folium.Rectangle(
                bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                color='red',
                fill=False,
                weight=2,
                popup="Region Boundary"
            ).add_to(m)
            
            # Add profile markers
            for idx, row in regional_data.iterrows():
                popup_text = f"""
                <b>Float {row['float_id']}</b><br>
                Cycle: {row.get('cycle_number', 'N/A')}<br>
                Lat: {row['latitude']:.3f}°N<br>
                Lon: {row['longitude']:.3f}°E<br>
                Date: {row.get('measurement_date', 'N/A')}
                """
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=6,
                    popup=popup_text,
                    color='blue',
                    fillColor='lightblue',
                    fillOpacity=0.7,
                    weight=2
                ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Add measurement tool
            plugins.MeasureControl().add_to(m)
            
            # Add full screen option
            plugins.Fullscreen().add_to(m)
            
            return m
            
        except Exception as e:
            logger.error(f"Failed to create regional map: {str(e)}")
            return self._create_empty_map(f"Error creating regional map: {str(e)}")
    
    def _create_color_legend(self, parameter: str, vmin: float, vmax: float) -> str:
        """Create HTML color legend for parameter maps"""
        units = self.parameter_units.get(parameter, '')
        
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>{parameter.title()}</b></p>
        <div style="display: flex; align-items: center;">
            <div style="width: 20px; height: 100%; background: linear-gradient(to top, blue, red); margin-right: 10px;"></div>
            <div>
                <div>{vmax:.2f} {units}</div>
                <div style="margin-top: 40px;">{vmin:.2f} {units}</div>
            </div>
        </div>
        </div>
        '''
        return legend_html
    
    def _create_empty_map(self, message: str) -> folium.Map:
        """Create an empty map with an informative message"""
        # Create a basic world map
        m = folium.Map(
            location=[0, 0],
            zoom_start=2,
            tiles='OpenStreetMap'
        )
        
        # Add message popup
        folium.Marker(
            location=[0, 0],
            popup=f"<b>No Data Available</b><br>{message}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        return m
    
    def create_comparison_map(self, profiles_df: pd.DataFrame, measurements_df: pd.DataFrame,
                            param1: str, param2: str) -> folium.Map:
        """Create map comparing two parameters"""
        try:
            if profiles_df.empty or measurements_df.empty:
                return self._create_empty_map("Profile and measurement data required")
            
            if param1 not in measurements_df.columns or param2 not in measurements_df.columns:
                return self._create_empty_map(f"Parameters {param1} and {param2} not found")
            
            # Calculate mean values for each profile
            param_means = measurements_df.groupby('profile_id').agg({
                param1: 'mean',
                param2: 'mean'
            }).reset_index()
            param_means.columns = ['id', f'mean_{param1}', f'mean_{param2}']
            
            # Merge with profile data
            map_data = profiles_df.merge(param_means, on='id', how='inner')
            map_data = map_data.dropna(subset=['latitude', 'longitude', f'mean_{param1}', f'mean_{param2}'])
            
            if map_data.empty:
                return self._create_empty_map(f"No valid data for {param1} vs {param2} comparison")
            
            # Calculate map center
            center_lat = map_data['latitude'].mean()
            center_lon = map_data['longitude'].mean()
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=4,
                tiles='OpenStreetMap'
            )
            
            # Calculate ratio for color coding
            map_data['ratio'] = map_data[f'mean_{param1}'] / map_data[f'mean_{param2}']
            ratio_values = map_data['ratio'].replace([np.inf, -np.inf], np.nan).dropna()
            
            if not ratio_values.empty:
                vmin, vmax = ratio_values.min(), ratio_values.max()
                
                # Normalize for color mapping
                if vmax > vmin:
                    normalized_ratios = (ratio_values - vmin) / (vmax - vmin)
                else:
                    normalized_ratios = np.ones(len(ratio_values)) * 0.5
                
                # Add markers
                for idx, row in map_data.iterrows():
                    if not np.isnan(row['ratio']):
                        ratio_val = row['ratio']
                        norm_val = (ratio_val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                        
                        # Color based on ratio
                        red = int(255 * norm_val)
                        blue = int(255 * (1 - norm_val))
                        color = f'#{red:02x}00{blue:02x}'
                        
                        popup_text = f"""
                        <b>Float {row['float_id']}</b><br>
                        Lat: {row['latitude']:.3f}°N<br>
                        Lon: {row['longitude']:.3f}°E<br>
                        {param1.title()}: {row[f'mean_{param1}']:.2f}<br>
                        {param2.title()}: {row[f'mean_{param2}']:.2f}<br>
                        Ratio: {ratio_val:.2f}
                        """
                        
                        folium.CircleMarker(
                            location=[row['latitude'], row['longitude']],
                            radius=8,
                            popup=popup_text,
                            color='black',
                            fillColor=color,
                            fillOpacity=0.8,
                            weight=1
                        ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            return m
            
        except Exception as e:
            logger.error(f"Failed to create comparison map: {str(e)}")
            return self._create_empty_map(f"Error creating comparison map: {str(e)}")
