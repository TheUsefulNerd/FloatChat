import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from database.connection import DatabaseManager
from visualization.plots import OceanographicPlots
from visualization.maps import OceanographicMaps
from config.settings import load_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Visualizations - ARGO Platform",
    page_icon="📊",
    layout="wide"
)

def initialize_components():
    """Initialize application components"""
    try:
        if 'config' not in st.session_state:
            st.session_state.config = load_config()
        
        if 'db_manager' not in st.session_state:
            st.session_state.db_manager = DatabaseManager(st.session_state.config)
        
        if 'plotter' not in st.session_state:
            st.session_state.plotter = OceanographicPlots()
            
        if 'mapper' not in st.session_state:
            st.session_state.mapper = OceanographicMaps()
            
        return True
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        return False

def load_data_for_visualization():
    """Load and prepare data for visualization"""
    try:
        # Get recent profiles for visualization
        profiles_df = st.session_state.db_manager.get_profiles(limit=1000)
        
        if profiles_df.empty:
            return None, None, "No profile data available"
        
        # Get sample measurements for plotting
        sample_profiles = profiles_df.head(50)  # Limit for performance
        measurements_list = []
        
        for profile_id in sample_profiles['id']:
            measurements = st.session_state.db_manager.get_measurements_by_profile(profile_id)
            if not measurements.empty:
                measurements['profile_id'] = profile_id
                measurements_list.append(measurements)
        
        if measurements_list:
            all_measurements = pd.concat(measurements_list, ignore_index=True)
            return profiles_df, all_measurements, None
        else:
            return profiles_df, pd.DataFrame(), "No measurement data available"
    
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        return None, None, f"Error loading data: {str(e)}"

def create_filter_sidebar():
    """Create sidebar filters for visualization"""
    st.sidebar.subheader("🎛️ Visualization Filters")
    
    filters = {}
    
    # Float selection
    float_filter = st.sidebar.selectbox(
        "Select Float",
        ["All Floats", "Specific Float"],
        help="Filter by specific float or show all"
    )
    
    if float_filter == "Specific Float":
        float_id = st.sidebar.text_input("Float ID")
        if float_id:
            filters['float_id'] = float_id
    
    # Date range
    date_range = st.sidebar.selectbox(
        "Date Range",
        ["All Time", "Last Month", "Last 6 Months", "Last Year", "Custom"]
    )
    
    if date_range == "Last Month":
        filters['start_date'] = datetime.now() - timedelta(days=30)
        filters['end_date'] = datetime.now()
    elif date_range == "Last 6 Months":
        filters['start_date'] = datetime.now() - timedelta(days=180)
        filters['end_date'] = datetime.now()
    elif date_range == "Last Year":
        filters['start_date'] = datetime.now() - timedelta(days=365)
        filters['end_date'] = datetime.now()
    elif date_range == "Custom":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")
        
        if start_date and end_date:
            filters['start_date'] = datetime.combine(start_date, datetime.min.time())
            filters['end_date'] = datetime.combine(end_date, datetime.max.time())
    
    # Geographic region
    region = st.sidebar.selectbox(
        "Geographic Region",
        ["Global", "Indian Ocean", "Pacific Ocean", "Atlantic Ocean", "Custom Bounds"]
    )
    
    region_bounds = {
        "Indian Ocean": {"min_lat": -40, "max_lat": 25, "min_lon": 20, "max_lon": 120},
        "Pacific Ocean": {"min_lat": -60, "max_lat": 60, "min_lon": 120, "max_lon": -60},
        "Atlantic Ocean": {"min_lat": -60, "max_lat": 70, "min_lon": -80, "max_lon": 20}
    }
    
    if region in region_bounds:
        filters.update(region_bounds[region])
    elif region == "Custom Bounds":
        st.sidebar.write("**Custom Coordinates**")
        min_lat = st.sidebar.number_input("Min Latitude", value=-90.0, min_value=-90.0, max_value=90.0)
        max_lat = st.sidebar.number_input("Max Latitude", value=90.0, min_value=-90.0, max_value=90.0)
        min_lon = st.sidebar.number_input("Min Longitude", value=-180.0, min_value=-180.0, max_value=180.0)
        max_lon = st.sidebar.number_input("Max Longitude", value=180.0, min_value=-180.0, max_value=180.0)
        
        if min_lat < max_lat and min_lon < max_lon:
            filters.update({
                "min_lat": min_lat, "max_lat": max_lat,
                "min_lon": min_lon, "max_lon": max_lon
            })
    
    # Data quality
    quality_filter = st.sidebar.selectbox(
        "Data Quality",
        ["All Data", "Good Quality Only"],
        help="Filter measurements by quality flags"
    )
    
    return filters, quality_filter

def main():
    """Main visualizations interface"""
    
    st.title("📊 ARGO Data Visualizations")
    st.markdown("Create and explore interactive visualizations of ARGO oceanographic data.")
    
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # Create sidebar filters
    filters, quality_filter = create_filter_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌊 Profile Analysis", 
        "🗺️ Geographic Maps", 
        "📈 Time Series", 
        "🔄 Parameter Comparison",
        "📊 Statistical Analysis"
    ])
    
    with tab1:
        st.subheader("Profile Analysis")
        
        try:
            # Load data
            profiles_df, measurements_df, error_msg = load_data_for_visualization()
            
            if error_msg:
                st.error(error_msg)
            elif profiles_df is not None and not measurements_df.empty:
                
                # Apply filters
                filtered_profiles = st.session_state.db_manager.get_profiles(
                    limit=100, filters=filters
                )
                
                if filtered_profiles.empty:
                    st.info("No profiles found with current filters.")
                else:
                    # Profile selection
                    profile_options = {}
                    for idx, row in filtered_profiles.head(20).iterrows():
                        label = f"Float {row['float_id']} - Cycle {row['cycle_number']}"
                        profile_options[label] = row['id']
                    
                    selected_profile = st.selectbox(
                        "Select Profile for Analysis",
                        list(profile_options.keys())
                    )
                    
                    if selected_profile:
                        profile_id = profile_options[selected_profile]
                        
                        # Get measurements for selected profile
                        measurements = st.session_state.db_manager.get_measurements_by_profile(profile_id)
                        
                        if not measurements.empty:
                            # Filter by quality if requested
                            if quality_filter == "Good Quality Only" and 'quality_flag' in measurements.columns:
                                measurements = measurements[measurements['quality_flag'] <= 2]
                            
                            # Parameter selection
                            available_params = [col for col in measurements.columns 
                                              if col in ['temperature', 'salinity', 'pressure', 'oxygen', 'nitrate', 'ph', 'chlorophyll']
                                              and measurements[col].notna().any()]
                            
                            if available_params:
                                # Multi-parameter depth profile
                                st.subheader("Multi-Parameter Depth Profiles")
                                
                                selected_params = st.multiselect(
                                    "Select parameters to display",
                                    available_params,
                                    default=available_params[:3] if len(available_params) >= 3 else available_params
                                )
                                
                                if selected_params:
                                    profile_info = filtered_profiles[filtered_profiles['id'] == profile_id].iloc[0]
                                    title = f"Float {profile_info['float_id']} - Cycle {profile_info['cycle_number']}"
                                    
                                    depth_fig = st.session_state.plotter.create_depth_profile(
                                        measurements, selected_params, title
                                    )
                                    st.plotly_chart(depth_fig, width=True)
                                
                                # T-S Diagram
                                if 'temperature' in measurements.columns and 'salinity' in measurements.columns:
                                    st.subheader("Temperature-Salinity Diagram")
                                    ts_fig = st.session_state.plotter.create_ts_diagram(measurements)
                                    st.plotly_chart(ts_fig, width=True)
                                
                                # Parameter distributions
                                st.subheader("Parameter Distributions")
                                
                                param_for_hist = st.selectbox(
                                    "Select parameter for distribution analysis",
                                    available_params
                                )
                                
                                hist_fig = st.session_state.plotter.create_histogram(measurements, param_for_hist)
                                st.plotly_chart(hist_fig, width=True)
                                
                                # Parameter comparison
                                if len(available_params) >= 2:
                                    st.subheader("Parameter Relationship")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        param1 = st.selectbox("X-axis parameter", available_params, key="param1")
                                    with col2:
                                        param2 = st.selectbox("Y-axis parameter", 
                                                            [p for p in available_params if p != param1], key="param2")
                                    
                                    if param1 and param2:
                                        comparison_fig = st.session_state.plotter.create_parameter_comparison(
                                            measurements, param1, param2
                                        )
                                        st.plotly_chart(comparison_fig, width=True)
                            else:
                                st.warning("No suitable parameters found for visualization.")
                        else:
                            st.warning("No measurements found for selected profile.")
            else:
                st.info("No data available for visualization.")
                
        except Exception as e:
            st.error(f"Error creating profile visualizations: {str(e)}")
    
    with tab2:
        st.subheader("Geographic Maps")
        
        try:
            # Load profiles for mapping
            profiles_df = st.session_state.db_manager.get_profiles(
                limit=2000, filters=filters
            )
            
            if profiles_df.empty:
                st.info("No profiles found for mapping.")
            else:
                # Map type selection
                map_type = st.selectbox(
                    "Select map visualization type",
                    ["Float Trajectories", "Profile Density Heatmap", "Parameter Distribution", "Regional Analysis"]
                )
                
                if map_type == "Float Trajectories":
                    st.subheader("ARGO Float Trajectories")
                    
                    # Float selection for trajectory
                    unique_floats = profiles_df['float_id'].unique()
                    float_selection = st.selectbox(
                        "Select float for trajectory (optional)",
                        ["All Floats"] + list(unique_floats[:20])  # Limit for performance
                    )
                    
                    selected_float = None if float_selection == "All Floats" else float_selection
                    
                    trajectory_map = st.session_state.mapper.create_float_trajectory_map(
                        profiles_df, selected_float
                    )
                    st.components.v1.html(trajectory_map._repr_html_(), height=600)
                
                elif map_type == "Profile Density Heatmap":
                    st.subheader("Profile Density Distribution")
                    
                    density_map = st.session_state.mapper.create_density_map(profiles_df)
                    st.components.v1.html(density_map._repr_html_(), height=600)
                
                elif map_type == "Parameter Distribution":
                    st.subheader("Parameter Spatial Distribution")
                    
                    # Load measurements for parameter mapping
                    sample_profiles = profiles_df.head(100)  # Limit for performance
                    measurements_list = []
                    
                    for profile_id in sample_profiles['id']:
                        measurements = st.session_state.db_manager.get_measurements_by_profile(profile_id)
                        if not measurements.empty:
                            measurements['profile_id'] = profile_id
                            measurements_list.append(measurements)
                    
                    if measurements_list:
                        all_measurements = pd.concat(measurements_list, ignore_index=True)
                        
                        # Parameter selection
                        available_params = [col for col in all_measurements.columns 
                                          if col in ['temperature', 'salinity', 'oxygen', 'nitrate', 'ph', 'chlorophyll']
                                          and all_measurements[col].notna().any()]
                        
                        if available_params:
                            selected_param = st.selectbox("Select parameter to map", available_params)
                            
                            # Depth range selection
                            depth_ranges = {
                                "Surface (0-50m)": (0, 50),
                                "Intermediate (50-500m)": (50, 500),
                                "Deep (>500m)": (500, 10000),
                                "All depths": None
                            }
                            
                            depth_selection = st.selectbox("Select depth range", list(depth_ranges.keys()))
                            depth_range = depth_ranges[depth_selection]
                            
                            param_map = st.session_state.mapper.create_parameter_map(
                                sample_profiles, all_measurements, selected_param, depth_range
                            )
                            st.components.v1.html(param_map._repr_html_(), height=600)
                        else:
                            st.warning("No suitable parameters found for mapping.")
                    else:
                        st.warning("No measurement data available for parameter mapping.")
                
                elif map_type == "Regional Analysis":
                    st.subheader("Regional Data Analysis")
                    
                    # Define analysis regions
                    regions = {
                        "Arabian Sea": {"min_lat": 10, "max_lat": 25, "min_lon": 50, "max_lon": 80},
                        "Bay of Bengal": {"min_lat": 5, "max_lat": 22, "min_lon": 80, "max_lon": 100},
                        "Equatorial Indian Ocean": {"min_lat": -10, "max_lat": 10, "min_lon": 50, "max_lon": 100},
                        "Southern Ocean": {"min_lat": -60, "max_lat": -30, "min_lon": 20, "max_lon": 120}
                    }
                    
                    selected_region = st.selectbox("Select region for analysis", list(regions.keys()))
                    region_bounds = regions[selected_region]
                    
                    regional_map = st.session_state.mapper.create_regional_map(profiles_df, region_bounds)
                    st.components.v1.html(regional_map._repr_html_(), height=600)
                    
                    # Regional statistics
                    regional_profiles = profiles_df[
                        (profiles_df['latitude'] >= region_bounds['min_lat']) &
                        (profiles_df['latitude'] <= region_bounds['max_lat']) &
                        (profiles_df['longitude'] >= region_bounds['min_lon']) &
                        (profiles_df['longitude'] <= region_bounds['max_lon'])
                    ]
                    
                    if not regional_profiles.empty:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Profiles in Region", len(regional_profiles))
                        with col2:
                            st.metric("Unique Floats", regional_profiles['float_id'].nunique())
                        with col3:
                            if 'measurement_date' in regional_profiles.columns:
                                date_span = (regional_profiles['measurement_date'].max() - 
                                           regional_profiles['measurement_date'].min()).days
                                st.metric("Date Span (days)", date_span)
                
        except Exception as e:
            st.error(f"Error creating geographic visualizations: {str(e)}")
    
    with tab3:
        st.subheader("Time Series Analysis")
        
        try:
            # Load data with temporal focus
            profiles_df = st.session_state.db_manager.get_profiles(
                limit=500, filters=filters
            )
            
            if profiles_df.empty:
                st.info("No profiles found for time series analysis.")
            else:
                # Ensure we have date information
                if 'measurement_date' not in profiles_df.columns:
                    st.warning("No date information available for time series analysis.")
                else:
                    # Parameter selection for time series
                    st.subheader("Parameter Time Series")
                    
                    # Get sample measurements to determine available parameters
                    sample_measurements = []
                    for profile_id in profiles_df['id'].head(20):
                        measurements = st.session_state.db_manager.get_measurements_by_profile(profile_id)
                        if not measurements.empty:
                            measurements['profile_id'] = profile_id
                            sample_measurements.append(measurements)
                    
                    if sample_measurements:
                        combined_measurements = pd.concat(sample_measurements, ignore_index=True)
                        
                        available_params = [col for col in combined_measurements.columns 
                                          if col in ['temperature', 'salinity', 'oxygen', 'nitrate', 'ph', 'chlorophyll']
                                          and combined_measurements[col].notna().any()]
                        
                        if available_params:
                            selected_param = st.selectbox("Select parameter for time series", available_params)
                            
                            # Depth level selection
                            depth_levels = st.selectbox(
                                "Select depth level",
                                ["Surface (0-50m)", "100m", "200m", "500m", "1000m", "All depths (mean)"]
                            )
                            
                            # Create time series data
                            time_series_data = []
                            
                            for _, profile in profiles_df.iterrows():
                                measurements = st.session_state.db_manager.get_measurements_by_profile(profile['id'])
                                
                                if not measurements.empty and selected_param in measurements.columns:
                                    # Filter by quality if requested
                                    if quality_filter == "Good Quality Only" and 'quality_flag' in measurements.columns:
                                        measurements = measurements[measurements['quality_flag'] <= 2]
                                    
                                    # Select depth range
                                    if depth_levels == "Surface (0-50m)":
                                        depth_data = measurements[measurements['depth'] <= 50]
                                    elif depth_levels == "100m":
                                        depth_data = measurements[
                                            (measurements['depth'] >= 80) & (measurements['depth'] <= 120)
                                        ]
                                    elif depth_levels == "200m":
                                        depth_data = measurements[
                                            (measurements['depth'] >= 180) & (measurements['depth'] <= 220)
                                        ]
                                    elif depth_levels == "500m":
                                        depth_data = measurements[
                                            (measurements['depth'] >= 480) & (measurements['depth'] <= 520)
                                        ]
                                    elif depth_levels == "1000m":
                                        depth_data = measurements[
                                            (measurements['depth'] >= 980) & (measurements['depth'] <= 1020)
                                        ]
                                    else:
                                        depth_data = measurements
                                    
                                    if not depth_data.empty:
                                        mean_value = depth_data[selected_param].mean()
                                        if not pd.isna(mean_value):
                                            time_series_data.append({
                                                'measurement_date': profile['measurement_date'],
                                                selected_param: mean_value,
                                                'float_id': profile['float_id']
                                            })
                            
                            if time_series_data:
                                time_series_df = pd.DataFrame(time_series_data)
                                
                                # Overall time series
                                ts_fig = st.session_state.plotter.create_time_series(time_series_df, selected_param)
                                st.plotly_chart(ts_fig, width=True)
                                
                                # Time series by float
                                if len(time_series_df['float_id'].unique()) > 1:
                                    st.subheader("Time Series by Float")
                                    
                                    selected_floats = st.multiselect(
                                        "Select floats to compare",
                                        time_series_df['float_id'].unique()[:10],
                                        default=list(time_series_df['float_id'].unique()[:3])
                                    )
                                    
                                    if selected_floats:
                                        float_data = time_series_df[time_series_df['float_id'].isin(selected_floats)]
                                        
                                        import plotly.express as px
                                        fig = px.line(
                                            float_data, 
                                            x='measurement_date', 
                                            y=selected_param,
                                            color='float_id',
                                            title=f"{selected_param.title()} Time Series by Float"
                                        )
                                        st.plotly_chart(fig, width=True)
                            else:
                                st.warning(f"No valid time series data found for {selected_param}.")
                        else:
                            st.warning("No suitable parameters found for time series analysis.")
                    else:
                        st.warning("No measurement data available for time series analysis.")
                        
        except Exception as e:
            st.error(f"Error creating time series visualizations: {str(e)}")
    
    with tab4:
        st.subheader("Parameter Comparison Analysis")
        
        try:
            # Load measurement data for comparison
            profiles_df = st.session_state.db_manager.get_profiles(
                limit=100, filters=filters
            )
            
            if profiles_df.empty:
                st.info("No profiles found for parameter comparison.")
            else:
                # Load measurements
                measurements_list = []
                for profile_id in profiles_df['id']:
                    measurements = st.session_state.db_manager.get_measurements_by_profile(profile_id)
                    if not measurements.empty:
                        measurements['profile_id'] = profile_id
                        measurements_list.append(measurements)
                
                if measurements_list:
                    all_measurements = pd.concat(measurements_list, ignore_index=True)
                    
                    # Filter by quality if requested
                    if quality_filter == "Good Quality Only" and 'quality_flag' in all_measurements.columns:
                        all_measurements = all_measurements[all_measurements['quality_flag'] <= 2]
                    
                    # Available parameters
                    available_params = [col for col in all_measurements.columns 
                                      if col in ['temperature', 'salinity', 'oxygen', 'nitrate', 'ph', 'chlorophyll']
                                      and all_measurements[col].notna().any()]
                    
                    if len(available_params) >= 2:
                        # Parameter selection
                        col1, col2 = st.columns(2)
                        with col1:
                            param1 = st.selectbox("Select first parameter", available_params, key="comp_param1")
                        with col2:
                            param2 = st.selectbox("Select second parameter", 
                                                [p for p in available_params if p != param1], key="comp_param2")
                        
                        if param1 and param2:
                            # Scatter plot comparison
                            st.subheader(f"{param1.title()} vs {param2.title()}")
                            
                            comparison_fig = st.session_state.plotter.create_parameter_comparison(
                                all_measurements, param1, param2
                            )
                            st.plotly_chart(comparison_fig, width=True)
                            
                            # Correlation analysis
                            correlation = all_measurements[[param1, param2]].corr().iloc[0, 1]
                            st.metric("Correlation Coefficient", f"{correlation:.3f}")
                            
                            # Statistical summary
                            st.subheader("Statistical Summary")
                            
                            summary_stats = all_measurements[[param1, param2]].describe()
                            st.dataframe(summary_stats)
                            
                            # Depth-stratified comparison
                            if 'depth' in all_measurements.columns:
                                st.subheader("Depth-Stratified Analysis")
                                
                                # Create depth bins
                                all_measurements['depth_bin'] = pd.cut(
                                    all_measurements['depth'], 
                                    bins=[0, 50, 200, 500, 1000, 2000, 10000],
                                    labels=['0-50m', '50-200m', '200-500m', '500-1000m', '1000-2000m', '>2000m']
                                )
                                
                                # Calculate correlations by depth
                                depth_correlations = []
                                for depth_bin in all_measurements['depth_bin'].cat.categories:
                                    depth_data = all_measurements[all_measurements['depth_bin'] == depth_bin]
                                    if len(depth_data) > 10:  # Minimum sample size
                                        corr = depth_data[[param1, param2]].corr().iloc[0, 1]
                                        if not pd.isna(corr):
                                            depth_correlations.append({
                                                'depth_range': depth_bin,
                                                'correlation': corr,
                                                'sample_size': len(depth_data)
                                            })
                                
                                if depth_correlations:
                                    depth_corr_df = pd.DataFrame(depth_correlations)
                                    st.dataframe(depth_corr_df)
                    else:
                        st.warning("Need at least 2 parameters for comparison analysis.")
                else:
                    st.warning("No measurement data available for comparison.")
                    
        except Exception as e:
            st.error(f"Error creating parameter comparison: {str(e)}")
    
    with tab5:
        st.subheader("Statistical Analysis")
        
        try:
            # Load data for statistical analysis
            profiles_df = st.session_state.db_manager.get_profiles(
                limit=200, filters=filters
            )
            
            if profiles_df.empty:
                st.info("No profiles found for statistical analysis.")
            else:
                # Load measurements
                measurements_list = []
                for profile_id in profiles_df['id']:
                    measurements = st.session_state.db_manager.get_measurements_by_profile(profile_id)
                    if not measurements.empty:
                        measurements['profile_id'] = profile_id
                        measurements_list.append(measurements)
                
                if measurements_list:
                    all_measurements = pd.concat(measurements_list, ignore_index=True)
                    
                    # Filter by quality if requested
                    if quality_filter == "Good Quality Only" and 'quality_flag' in all_measurements.columns:
                        all_measurements = all_measurements[all_measurements['quality_flag'] <= 2]
                    
                    # Available parameters
                    available_params = [col for col in all_measurements.columns 
                                      if col in ['temperature', 'salinity', 'oxygen', 'nitrate', 'ph', 'chlorophyll']
                                      and all_measurements[col].notna().any()]
                    
                    if available_params:
                        # Parameter selection for statistical analysis
                        selected_param = st.selectbox(
                            "Select parameter for statistical analysis", 
                            available_params
                        )
                        
                        param_data = all_measurements[selected_param].dropna()
                        
                        if not param_data.empty:
                            # Basic statistics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Mean", f"{param_data.mean():.3f}")
                            with col2:
                                st.metric("Std Dev", f"{param_data.std():.3f}")
                            with col3:
                                st.metric("Median", f"{param_data.median():.3f}")
                            with col4:
                                st.metric("Count", f"{len(param_data):,}")
                            
                            # Distribution histogram
                            st.subheader("Distribution Analysis")
                            hist_fig = st.session_state.plotter.create_histogram(all_measurements, selected_param)
                            st.plotly_chart(hist_fig, width=True)
                            
                            # Quantile analysis
                            st.subheader("Quantile Analysis")
                            quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
                            quantile_values = param_data.quantile(quantiles)
                            
                            quantile_df = pd.DataFrame({
                                'Quantile': [f"{q*100:.0f}%" for q in quantiles],
                                'Value': quantile_values.values
                            })
                            st.dataframe(quantile_df)
                            
                            # Depth profile statistics
                            if 'depth' in all_measurements.columns:
                                st.subheader("Depth-Stratified Statistics")
                                
                                # Create depth bins
                                depth_bins = [0, 50, 100, 200, 500, 1000, 2000, 5000]
                                all_measurements['depth_bin'] = pd.cut(
                                    all_measurements['depth'], 
                                    bins=depth_bins,
                                    labels=[f"{depth_bins[i]}-{depth_bins[i+1]}m" for i in range(len(depth_bins)-1)]
                                )
                                
                                # Calculate statistics by depth
                                depth_stats = all_measurements.groupby('depth_bin')[selected_param].agg([
                                    'count', 'mean', 'std', 'min', 'max'
                                ]).round(3)
                                
                                st.dataframe(depth_stats)
                            
                            # Anomaly detection
                            st.subheader("Anomaly Detection")
                            
                            # Z-score method
                            z_scores = np.abs((param_data - param_data.mean()) / param_data.std())
                            anomalies_zscore = len(z_scores[z_scores > 3])
                            
                            # IQR method
                            q1 = param_data.quantile(0.25)
                            q3 = param_data.quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            anomalies_iqr = len(param_data[(param_data < lower_bound) | (param_data > upper_bound)])
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Measurements", len(param_data))
                            with col2:
                                st.metric("Z-score Anomalies (>3σ)", anomalies_zscore)
                            with col3:
                                st.metric("IQR Anomalies", anomalies_iqr)
                        else:
                            st.warning(f"No valid data found for {selected_param}.")
                    else:
                        st.warning("No suitable parameters found for statistical analysis.")
                else:
                    st.warning("No measurement data available for statistical analysis.")
                    
        except Exception as e:
            st.error(f"Error in statistical analysis: {str(e)}")
    
    # Export functionality
    st.subheader("📁 Export Visualizations")
    
    export_format = st.selectbox(
        "Select export format",
        ["PNG", "HTML", "PDF", "SVG"]
    )
    
    if st.button("Generate Export Package"):
        st.info("Export functionality can be extended to save current visualizations.")
        st.markdown("*Note: Individual plots can be exported using the plotly controls in each visualization.*")

if __name__ == "__main__":
    main()