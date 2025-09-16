import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OceanographicPlots:
    """
    Create oceanographic visualizations using Plotly
    """
    
    def __init__(self):
        self.color_schemes = {
            'temperature': 'RdYlBu_r',
            'salinity': 'Viridis',
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
    
    def create_depth_profile(self, measurements_df: pd.DataFrame, parameters: List[str], 
                           profile_title: str = "ARGO Profile") -> go.Figure:
        """Create depth profile plots for specified parameters"""
        try:
            if measurements_df.empty:
                return self._create_empty_plot("No measurement data available")
            
            # Filter for good quality data
            if 'quality_flag' in measurements_df.columns:
                good_data = measurements_df[measurements_df['quality_flag'] <= 2].copy()
            else:
                good_data = measurements_df.copy()
            
            if good_data.empty:
                return self._create_empty_plot("No good quality data available")
            
            # Create subplots
            n_params = len(parameters)
            fig = make_subplots(
                rows=1, 
                cols=n_params,
                subplot_titles=[f"{param.title()} ({self.parameter_units.get(param, '')})" 
                              for param in parameters],
                shared_yaxes=True
            )
            
            # Add traces for each parameter
            for i, param in enumerate(parameters, 1):
                if param in good_data.columns:
                    param_data = good_data.dropna(subset=[param, 'depth'])
                    
                    if not param_data.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=param_data[param],
                                y=param_data['depth'],
                                mode='lines+markers',
                                name=param.title(),
                                line=dict(width=2),
                                marker=dict(size=4),
                                hovertemplate=f"<b>{param.title()}</b><br>" +
                                            f"Value: %{{x:.2f}} {self.parameter_units.get(param, '')}<br>" +
                                            "Depth: %{y:.1f} m<extra></extra>"
                            ),
                            row=1, col=i
                        )
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=profile_title,
                    x=0.5,
                    font=dict(size=16)
                ),
                height=600,
                showlegend=False,
                margin=dict(t=80, b=50, l=50, r=50)
            )
            
            # Update y-axes (depth)
            fig.update_yaxes(
                title_text="Depth (m)",
                autorange="reversed",  # Depth increases downward
                row=1, col=1
            )
            
            # Update x-axes
            for i, param in enumerate(parameters, 1):
                fig.update_xaxes(
                    title_text=f"{param.title()} ({self.parameter_units.get(param, '')})",
                    row=1, col=i
                )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create depth profile: {str(e)}")
            return self._create_empty_plot(f"Error creating depth profile: {str(e)}")
    
    def create_ts_diagram(self, measurements_df: pd.DataFrame) -> go.Figure:
        """Create Temperature-Salinity diagram"""
        try:
            if measurements_df.empty or 'temperature' not in measurements_df.columns or 'salinity' not in measurements_df.columns:
                return self._create_empty_plot("Temperature and salinity data required for T-S diagram")
            
            # Filter for good quality data
            if 'quality_flag' in measurements_df.columns:
                good_data = measurements_df[measurements_df['quality_flag'] <= 2].copy()
            else:
                good_data = measurements_df.copy()
            
            ts_data = good_data.dropna(subset=['temperature', 'salinity'])
            
            if ts_data.empty:
                return self._create_empty_plot("No valid temperature-salinity data available")
            
            # Color by depth if available
            if 'depth' in ts_data.columns:
                color_col = 'depth'
                color_title = 'Depth (m)'
                colorscale = 'Viridis'
            else:
                color_col = None
                color_title = None
                colorscale = None
            
            fig = go.Figure()
            
            if color_col:
                fig.add_trace(
                    go.Scatter(
                        x=ts_data['salinity'],
                        y=ts_data['temperature'],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=ts_data[color_col],
                            colorscale=colorscale,
                            colorbar=dict(title=color_title),
                            line=dict(width=0.5, color='DarkSlateGrey')
                        ),
                        hovertemplate="<b>T-S Point</b><br>" +
                                    "Salinity: %{x:.2f} PSU<br>" +
                                    "Temperature: %{y:.2f} °C<br>" +
                                    f"{color_title}: %{{marker.color:.1f}}<extra></extra>",
                        name="T-S Data"
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=ts_data['salinity'],
                        y=ts_data['temperature'],
                        mode='markers',
                        marker=dict(size=6, color='blue'),
                        hovertemplate="<b>T-S Point</b><br>" +
                                    "Salinity: %{x:.2f} PSU<br>" +
                                    "Temperature: %{y:.2f} °C<extra></extra>",
                        name="T-S Data"
                    )
                )
            
            fig.update_layout(
                title="Temperature-Salinity Diagram",
                xaxis_title="Salinity (PSU)",
                yaxis_title="Temperature (°C)",
                height=500,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create T-S diagram: {str(e)}")
            return self._create_empty_plot(f"Error creating T-S diagram: {str(e)}")
    
    def create_time_series(self, profiles_df: pd.DataFrame, parameter: str) -> go.Figure:
        """Create time series plot for a parameter across multiple profiles"""
        try:
            if profiles_df.empty or parameter not in profiles_df.columns:
                return self._create_empty_plot(f"No {parameter} data available for time series")
            
            # Ensure measurement_date is datetime
            if 'measurement_date' in profiles_df.columns:
                profiles_df = profiles_df.copy()
                profiles_df['measurement_date'] = pd.to_datetime(profiles_df['measurement_date'])
                
                # Sort by date
                profiles_df = profiles_df.sort_values('measurement_date')
                
                # Remove null values
                data = profiles_df.dropna(subset=[parameter, 'measurement_date'])
                
                if data.empty:
                    return self._create_empty_plot(f"No valid {parameter} data with dates")
                
                fig = go.Figure()
                
                fig.add_trace(
                    go.Scatter(
                        x=data['measurement_date'],
                        y=data[parameter],
                        mode='lines+markers',
                        name=parameter.title(),
                        line=dict(width=2),
                        marker=dict(size=4),
                        hovertemplate=f"<b>{parameter.title()}</b><br>" +
                                    "Date: %{x}<br>" +
                                    f"Value: %{{y:.2f}} {self.parameter_units.get(parameter, '')}<extra></extra>"
                    )
                )
                
                fig.update_layout(
                    title=f"{parameter.title()} Time Series",
                    xaxis_title="Date",
                    yaxis_title=f"{parameter.title()} ({self.parameter_units.get(parameter, '')})",
                    height=400,
                    showlegend=False
                )
                
                return fig
            else:
                return self._create_empty_plot("No date information available for time series")
                
        except Exception as e:
            logger.error(f"Failed to create time series: {str(e)}")
            return self._create_empty_plot(f"Error creating time series: {str(e)}")
    
    def create_parameter_comparison(self, measurements_df: pd.DataFrame, param1: str, param2: str) -> go.Figure:
        """Create scatter plot comparing two parameters"""
        try:
            if measurements_df.empty or param1 not in measurements_df.columns or param2 not in measurements_df.columns:
                return self._create_empty_plot(f"Both {param1} and {param2} data required for comparison")
            
            # Filter for good quality data
            if 'quality_flag' in measurements_df.columns:
                good_data = measurements_df[measurements_df['quality_flag'] <= 2].copy()
            else:
                good_data = measurements_df.copy()
            
            comparison_data = good_data.dropna(subset=[param1, param2])
            
            if comparison_data.empty:
                return self._create_empty_plot(f"No valid data for {param1} vs {param2} comparison")
            
            # Color by depth if available
            if 'depth' in comparison_data.columns:
                color_col = 'depth'
                color_title = 'Depth (m)'
                colorscale = 'Viridis'
            else:
                color_col = None
                color_title = None
                colorscale = None
            
            fig = go.Figure()
            
            if color_col:
                fig.add_trace(
                    go.Scatter(
                        x=comparison_data[param1],
                        y=comparison_data[param2],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=comparison_data[color_col],
                            colorscale=colorscale,
                            colorbar=dict(title=color_title),
                            line=dict(width=0.5, color='DarkSlateGrey')
                        ),
                        hovertemplate=f"<b>Parameter Comparison</b><br>" +
                                    f"{param1.title()}: %{{x:.2f}} {self.parameter_units.get(param1, '')}<br>" +
                                    f"{param2.title()}: %{{y:.2f}} {self.parameter_units.get(param2, '')}<br>" +
                                    f"{color_title}: %{{marker.color:.1f}}<extra></extra>",
                        name="Comparison"
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=comparison_data[param1],
                        y=comparison_data[param2],
                        mode='markers',
                        marker=dict(size=6, color='blue'),
                        hovertemplate=f"<b>Parameter Comparison</b><br>" +
                                    f"{param1.title()}: %{{x:.2f}} {self.parameter_units.get(param1, '')}<br>" +
                                    f"{param2.title()}: %{{y:.2f}} {self.parameter_units.get(param2, '')}<extra></extra>",
                        name="Comparison"
                    )
                )
            
            fig.update_layout(
                title=f"{param1.title()} vs {param2.title()}",
                xaxis_title=f"{param1.title()} ({self.parameter_units.get(param1, '')})",
                yaxis_title=f"{param2.title()} ({self.parameter_units.get(param2, '')})",
                height=500,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create parameter comparison: {str(e)}")
            return self._create_empty_plot(f"Error creating comparison plot: {str(e)}")
    
    def create_depth_time_plot(self, profiles_df: pd.DataFrame, measurements_df: pd.DataFrame, 
                              parameter: str) -> go.Figure:
        """Create depth-time contour plot for a parameter"""
        try:
            if profiles_df.empty or measurements_df.empty:
                return self._create_empty_plot("Profile and measurement data required for depth-time plot")
            
            # Merge profile and measurement data
            merged_data = measurements_df.merge(
                profiles_df[['id', 'measurement_date', 'float_id']], 
                left_on='profile_id', 
                right_on='id',
                how='left'
            )
            
            if parameter not in merged_data.columns:
                return self._create_empty_plot(f"Parameter {parameter} not found in measurement data")
            
            # Filter for good quality data
            if 'quality_flag' in merged_data.columns:
                good_data = merged_data[merged_data['quality_flag'] <= 2].copy()
            else:
                good_data = merged_data.copy()
            
            plot_data = good_data.dropna(subset=[parameter, 'depth', 'measurement_date'])
            
            if plot_data.empty:
                return self._create_empty_plot(f"No valid data for {parameter} depth-time plot")
            
            # Convert measurement_date to datetime
            plot_data['measurement_date'] = pd.to_datetime(plot_data['measurement_date'])
            
            # Create pivot table for contour plot
            try:
                # Group by date and depth, taking mean of parameter values
                pivot_data = plot_data.groupby(['measurement_date', 'depth'])[parameter].mean().reset_index()
                
                # Create regular grid for interpolation
                dates = pd.date_range(plot_data['measurement_date'].min(), 
                                    plot_data['measurement_date'].max(), 
                                    periods=50)
                depths = np.linspace(plot_data['depth'].min(), plot_data['depth'].max(), 50)
                
                # Simple gridding - in production, use proper interpolation
                grid_z = np.full((len(depths), len(dates)), np.nan)
                
                fig = go.Figure()
                
                # Add scatter plot of actual data points
                fig.add_trace(
                    go.Scatter(
                        x=plot_data['measurement_date'],
                        y=plot_data['depth'],
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=plot_data[parameter],
                            colorscale=self.color_schemes.get(parameter, 'Viridis'),
                            colorbar=dict(title=f"{parameter.title()} ({self.parameter_units.get(parameter, '')})"),
                            line=dict(width=0.5, color='black')
                        ),
                        hovertemplate=f"<b>{parameter.title()}</b><br>" +
                                    "Date: %{x}<br>" +
                                    "Depth: %{y:.1f} m<br>" +
                                    f"Value: %{{marker.color:.2f}} {self.parameter_units.get(parameter, '')}<extra></extra>",
                        name=parameter.title()
                    )
                )
                
                fig.update_layout(
                    title=f"{parameter.title()} Depth-Time Distribution",
                    xaxis_title="Date",
                    yaxis_title="Depth (m)",
                    yaxis=dict(autorange="reversed"),  # Depth increases downward
                    height=500,
                    showlegend=False
                )
                
                return fig
                
            except Exception as e:
                # If contouring fails, fall back to scatter plot
                fig = go.Figure()
                
                fig.add_trace(
                    go.Scatter(
                        x=plot_data['measurement_date'],
                        y=plot_data['depth'],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=plot_data[parameter],
                            colorscale=self.color_schemes.get(parameter, 'Viridis'),
                            colorbar=dict(title=f"{parameter.title()} ({self.parameter_units.get(parameter, '')})"),
                            line=dict(width=0.5, color='black')
                        ),
                        hovertemplate=f"<b>{parameter.title()}</b><br>" +
                                    "Date: %{x}<br>" +
                                    "Depth: %{y:.1f} m<br>" +
                                    f"Value: %{{marker.color:.2f}} {self.parameter_units.get(parameter, '')}<extra></extra>",
                        name=parameter.title()
                    )
                )
                
                fig.update_layout(
                    title=f"{parameter.title()} Depth-Time Distribution",
                    xaxis_title="Date",
                    yaxis_title="Depth (m)",
                    yaxis=dict(autorange="reversed"),
                    height=500,
                    showlegend=False
                )
                
                return fig
                
        except Exception as e:
            logger.error(f"Failed to create depth-time plot: {str(e)}")
            return self._create_empty_plot(f"Error creating depth-time plot: {str(e)}")
    
    def create_histogram(self, measurements_df: pd.DataFrame, parameter: str) -> go.Figure:
        """Create histogram for parameter distribution"""
        try:
            if measurements_df.empty or parameter not in measurements_df.columns:
                return self._create_empty_plot(f"No {parameter} data available for histogram")
            
            # Filter for good quality data
            if 'quality_flag' in measurements_df.columns:
                good_data = measurements_df[measurements_df['quality_flag'] <= 2].copy()
            else:
                good_data = measurements_df.copy()
            
            param_data = good_data[parameter].dropna()
            
            if param_data.empty:
                return self._create_empty_plot(f"No valid {parameter} data for histogram")
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Histogram(
                    x=param_data,
                    nbinsx=30,
                    marker_color='lightblue',
                    marker_line_color='black',
                    marker_line_width=1,
                    hovertemplate="<b>Frequency Distribution</b><br>" +
                                f"Range: %{{x}} {self.parameter_units.get(parameter, '')}<br>" +
                                "Count: %{y}<extra></extra>",
                    name=parameter.title()
                )
            )
            
            # Add statistics text
            stats_text = f"Mean: {param_data.mean():.2f}<br>"
            stats_text += f"Std: {param_data.std():.2f}<br>"
            stats_text += f"Min: {param_data.min():.2f}<br>"
            stats_text += f"Max: {param_data.max():.2f}"
            
            fig.add_annotation(
                text=stats_text,
                xref="paper", yref="paper",
                x=0.7, y=0.9,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
            
            fig.update_layout(
                title=f"{parameter.title()} Distribution",
                xaxis_title=f"{parameter.title()} ({self.parameter_units.get(parameter, '')})",
                yaxis_title="Frequency",
                height=400,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create histogram: {str(e)}")
            return self._create_empty_plot(f"Error creating histogram: {str(e)}")
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create an empty plot with an informative message"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
            align="center"
        )
        
        fig.update_layout(
            title="No Data Available",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400,
            showlegend=False
        )
        
        return fig
