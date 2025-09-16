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
from utils.helpers import format_data_for_display, create_download_link
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Data Explorer - ARGO Platform",
    page_icon="🔍",
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

def build_filters_from_sidebar():
    """Build filter dictionary from sidebar inputs"""
    filters = {}
    
    st.sidebar.subheader("🔍 Data Filters")
    
    # Float ID filter
    float_id = st.sidebar.text_input("Float ID", help="Enter specific float ID to filter")
    if float_id:
        filters['float_id'] = float_id
    
    # Date range filter
    st.sidebar.write("**Date Range**")
    date_option = st.sidebar.selectbox(
        "Select date range",
        ["All dates", "Last 30 days", "Last 6 months", "Last year", "Custom range"]
    )
    
    if date_option == "Last 30 days":
        filters['start_date'] = datetime.now() - timedelta(days=30)
        filters['end_date'] = datetime.now()
    elif date_option == "Last 6 months":
        filters['start_date'] = datetime.now() - timedelta(days=180)
        filters['end_date'] = datetime.now()
    elif date_option == "Last year":
        filters['start_date'] = datetime.now() - timedelta(days=365)
        filters['end_date'] = datetime.now()
    elif date_option == "Custom range":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start date")
            if start_date:
                filters['start_date'] = datetime.combine(start_date, datetime.min.time())
        with col2:
            end_date = st.date_input("End date")
            if end_date:
                filters['end_date'] = datetime.combine(end_date, datetime.max.time())
    
    # Geographic filters
    st.sidebar.write("**Geographic Region**")
    
    # Predefined regions
    region_option = st.sidebar.selectbox(
        "Select region",
        ["Global", "Indian Ocean", "Pacific Ocean", "Atlantic Ocean", "Arabian Sea", "Mediterranean Sea", "Custom coordinates"]
    )
    
    region_bounds = {
        "Indian Ocean": {"min_lat": -40, "max_lat": 25, "min_lon": 20, "max_lon": 120},
        "Pacific Ocean": {"min_lat": -60, "max_lat": 60, "min_lon": 120, "max_lon": -60},
        "Atlantic Ocean": {"min_lat": -60, "max_lat": 70, "min_lon": -80, "max_lon": 20},
        "Arabian Sea": {"min_lat": 10, "max_lat": 25, "min_lon": 50, "max_lon": 80},
        "Mediterranean Sea": {"min_lat": 30, "max_lat": 46, "min_lon": -6, "max_lon": 36}
    }
    
    if region_option in region_bounds:
        bounds = region_bounds[region_option]
        filters.update(bounds)
    elif region_option == "Custom coordinates":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            min_lat = st.number_input("Min Latitude", value=-90.0, min_value=-90.0, max_value=90.0)
            min_lon = st.number_input("Min Longitude", value=-180.0, min_value=-180.0, max_value=180.0)
        with col2:
            max_lat = st.number_input("Max Latitude", value=90.0, min_value=-90.0, max_value=90.0)
            max_lon = st.number_input("Max Longitude", value=180.0, min_value=-180.0, max_value=180.0)
        
        if min_lat < max_lat and min_lon < max_lon:
            filters.update({
                "min_lat": min_lat, "max_lat": max_lat,
                "min_lon": min_lon, "max_lon": max_lon
            })
    
    # Data quality filter
    quality_filter = st.sidebar.selectbox(
        "Data Quality",
        ["All data", "Good quality only", "Questionable data only"],
        help="Filter by data quality flags"
    )
    
    return filters, quality_filter

def main():
    """Main data explorer interface"""
    
    st.title("🔍 ARGO Data Explorer")
    st.markdown("Browse, filter, and explore ARGO oceanographic data.")
    
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # Sidebar filters
    filters, quality_filter = build_filters_from_sidebar()
    
    # Display options sidebar
    st.sidebar.subheader("📊 Display Options")
    
    records_per_page = st.sidebar.selectbox(
        "Records per page",
        [10, 25, 50, 100, 200],
        index=2
    )
    
    show_coordinates = st.sidebar.checkbox("Show coordinates", value=True)
    show_metadata = st.sidebar.checkbox("Show metadata", value=False)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Profile Browser", "🗺️ Geographic View", "📊 Quick Statistics", "💾 Data Export"])
    
    with tab1:
        st.subheader("Profile Browser")
        
        try:
            # Get profiles based on filters
            profiles_df = st.session_state.db_manager.get_profiles(
                limit=records_per_page * 5,  # Get more records for pagination
                filters=filters
            )
            
            if profiles_df.empty:
                st.info("No profiles found matching your criteria. Try adjusting the filters.")
            else:
                # Apply quality filter on display
                display_df = profiles_df.copy()
                
                # Pagination
                total_records = len(display_df)
                total_pages = max(1, (total_records - 1) // records_per_page + 1)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    page = st.selectbox(
                        f"Page (showing {records_per_page} of {total_records} records)",
                        range(1, total_pages + 1)
                    )
                
                # Calculate page slice
                start_idx = (page - 1) * records_per_page
                end_idx = min(start_idx + records_per_page, total_records)
                page_df = display_df.iloc[start_idx:end_idx]
                
                # Format data for display
                formatted_df = format_data_for_display(page_df, show_coordinates, show_metadata)
                
                # Display profiles table
                st.dataframe(formatted_df, use_container_width=True)
                
                # Profile details section
                if not page_df.empty:
                    st.subheader("Profile Details")
                    
                    # Profile selector
                    profile_options = {}
                    for idx, row in page_df.iterrows():
                        label = f"Float {row['float_id']} - Cycle {row['cycle_number']} ({row['measurement_date']})"
                        profile_options[label] = row['id']
                    
                    selected_profile_label = st.selectbox(
                        "Select profile to view details",
                        list(profile_options.keys())
                    )
                    
                    if selected_profile_label:
                        profile_id = profile_options[selected_profile_label]
                        
                        # Get measurements for selected profile
                        measurements_df = st.session_state.db_manager.get_measurements_by_profile(profile_id)
                        
                        if not measurements_df.empty:
                            # Profile overview
                            profile_info = page_df[page_df['id'] == profile_id].iloc[0]
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Float ID", profile_info['float_id'])
                            with col2:
                                st.metric("Cycle Number", profile_info['cycle_number'])
                            with col3:
                                st.metric("Measurements", len(measurements_df))
                            with col4:
                                st.metric("Max Depth", f"{measurements_df['depth'].max():.1f} m")
                            
                            # Measurement plots
                            st.subheader("Measurement Profiles")
                            
                            # Parameter selection for plotting
                            available_params = [col for col in measurements_df.columns 
                                              if col in ['temperature', 'salinity', 'pressure', 'oxygen', 'nitrate', 'ph', 'chlorophyll']
                                              and measurements_df[col].notna().any()]
                            
                            if available_params:
                                selected_params = st.multiselect(
                                    "Select parameters to plot",
                                    available_params,
                                    default=available_params[:2]
                                )
                                
                                if selected_params:
                                    # Create depth profile plot
                                    profile_title = f"Float {profile_info['float_id']} - Cycle {profile_info['cycle_number']}"
                                    fig = st.session_state.plotter.create_depth_profile(
                                        measurements_df, selected_params, profile_title
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # T-S diagram if both temperature and salinity are available
                                    if 'temperature' in measurements_df.columns and 'salinity' in measurements_df.columns:
                                        ts_fig = st.session_state.plotter.create_ts_diagram(measurements_df)
                                        st.plotly_chart(ts_fig, use_container_width=True)
                            
                            # Raw data table
                            with st.expander("📋 View Raw Measurements"):
                                st.dataframe(measurements_df, use_container_width=True)
                        
                        else:
                            st.warning("No measurements found for this profile.")
        
        except Exception as e:
            st.error(f"Error loading profiles: {str(e)}")
    
    with tab2:
        st.subheader("Geographic View")
        
        try:
            # Get profiles for mapping
            profiles_df = st.session_state.db_manager.get_profiles(
                limit=1000,  # Limit for map performance
                filters=filters
            )
            
            if profiles_df.empty:
                st.info("No profiles found for mapping. Try adjusting the filters.")
            else:
                # Map type selection
                map_type = st.selectbox(
                    "Select map type",
                    ["Float Trajectories", "Profile Density", "Parameter Values"]
                )
                
                if map_type == "Float Trajectories":
                    # Float selection for trajectory
                    unique_floats = profiles_df['float_id'].unique()
                    if len(unique_floats) > 20:
                        st.info(f"Showing trajectories for all {len(unique_floats)} floats. This may take a moment to load.")
                    
                    selected_float = st.selectbox(
                        "Select specific float (optional)",
                        ["All floats"] + list(unique_floats)
                    )
                    
                    float_id = None if selected_float == "All floats" else selected_float
                    
                    # Create trajectory map
                    trajectory_map = st.session_state.mapper.create_float_trajectory_map(
                        profiles_df, float_id
                    )
                    st_folium_map = st.components.v1.html(
                        trajectory_map._repr_html_(), height=600
                    )
                
                elif map_type == "Profile Density":
                    # Create density heatmap
                    density_map = st.session_state.mapper.create_density_map(profiles_df)
                    st.components.v1.html(density_map._repr_html_(), height=600)
                
                elif map_type == "Parameter Values":
                    # Parameter mapping requires measurements
                    st.info("Loading measurement data for parameter mapping...")
                    
                    # Get a sample of measurements for mapping
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
                            depth_option = st.selectbox(
                                "Depth range",
                                ["Surface (0-50m)", "Intermediate (50-500m)", "Deep (>500m)", "All depths"]
                            )
                            
                            depth_ranges = {
                                "Surface (0-50m)": (0, 50),
                                "Intermediate (50-500m)": (50, 500),
                                "Deep (>500m)": (500, 10000),
                                "All depths": None
                            }
                            
                            depth_range = depth_ranges[depth_option]
                            
                            # Create parameter map
                            param_map = st.session_state.mapper.create_parameter_map(
                                sample_profiles, all_measurements, selected_param, depth_range
                            )
                            st.components.v1.html(param_map._repr_html_(), height=600)
                        else:
                            st.warning("No suitable parameters found for mapping.")
                    else:
                        st.warning("No measurement data available for parameter mapping.")
        
        except Exception as e:
            st.error(f"Error creating maps: {str(e)}")
    
    with tab3:
        st.subheader("Quick Statistics")
        
        try:
            # Get database statistics
            stats = st.session_state.db_manager.get_summary_statistics()
            
            if stats:
                # Overview metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Profiles", f"{stats.get('total_profiles', 0):,}")
                
                with col2:
                    st.metric("Total Measurements", f"{stats.get('total_measurements', 0):,}")
                
                with col3:
                    st.metric("Unique Floats", f"{stats.get('unique_floats', 0):,}")
                
                with col4:
                    if stats.get('total_measurements', 0) > 0 and stats.get('total_profiles', 0) > 0:
                        avg_measurements = stats['total_measurements'] / stats['total_profiles']
                        st.metric("Avg Measurements/Profile", f"{avg_measurements:.0f}")
                
                # Temporal coverage
                if stats.get('date_range'):
                    date_range = stats['date_range']
                    if date_range['earliest'] and date_range['latest']:
                        st.subheader("Temporal Coverage")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Earliest measurement:** {date_range['earliest'].strftime('%Y-%m-%d')}")
                        with col2:
                            st.write(f"**Latest measurement:** {date_range['latest'].strftime('%Y-%m-%d')}")
                        
                        # Duration
                        duration = date_range['latest'] - date_range['earliest']
                        st.write(f"**Data span:** {duration.days} days ({duration.days / 365.25:.1f} years)")
                
                # Geographic coverage
                if stats.get('geographic_coverage'):
                    geo = stats['geographic_coverage']
                    if all(v is not None for v in geo.values()):
                        st.subheader("Geographic Coverage")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Latitude range:** {geo['min_latitude']:.2f}°N to {geo['max_latitude']:.2f}°N")
                        with col2:
                            st.write(f"**Longitude range:** {geo['min_longitude']:.2f}°E to {geo['max_longitude']:.2f}°E")
                
                # Filtered statistics
                if filters:
                    st.subheader("Filtered Data Statistics")
                    filtered_profiles = st.session_state.db_manager.get_profiles(
                        limit=10000, filters=filters
                    )
                    
                    if not filtered_profiles.empty:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Filtered Profiles", len(filtered_profiles))
                        with col2:
                            st.metric("Unique Floats", filtered_profiles['float_id'].nunique())
                        with col3:
                            if 'measurement_date' in filtered_profiles.columns:
                                date_span = (filtered_profiles['measurement_date'].max() - 
                                           filtered_profiles['measurement_date'].min()).days
                                st.metric("Date Span (days)", date_span)
            else:
                st.info("No statistics available. Please check database connection.")
        
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}")
    
    with tab4:
        st.subheader("Data Export")
        
        try:
            # Export options
            export_format = st.selectbox(
                "Select export format",
                ["CSV", "Parquet", "NetCDF", "JSON"]
            )
            
            export_scope = st.selectbox(
                "Select data scope",
                ["Current filter results", "All profiles", "Specific float", "Date range"]
            )
            
            # Additional export parameters based on scope
            export_filters = filters.copy() if export_scope == "Current filter results" else {}
            
            if export_scope == "Specific float":
                float_id = st.text_input("Enter Float ID")
                if float_id:
                    export_filters['float_id'] = float_id
            
            elif export_scope == "Date range":
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start date", key="export_start")
                with col2:
                    end_date = st.date_input("End date", key="export_end")
                
                if start_date and end_date:
                    export_filters['start_date'] = datetime.combine(start_date, datetime.min.time())
                    export_filters['end_date'] = datetime.combine(end_date, datetime.max.time())
            
            # Include measurements option
            include_measurements = st.checkbox(
                "Include detailed measurements",
                help="Include individual depth measurements (increases file size significantly)"
            )
            
            # Export button
            if st.button("Generate Export", type="primary"):
                with st.spinner("Preparing export..."):
                    # Get profiles
                    export_profiles = st.session_state.db_manager.get_profiles(
                        limit=50000,  # Reasonable limit for export
                        filters=export_filters
                    )
                    
                    if export_profiles.empty:
                        st.warning("No data found for export with current criteria.")
                    else:
                        # Create download link
                        if include_measurements:
                            # Get measurements for all profiles
                            st.info("Loading detailed measurements... This may take a moment.")
                            
                            measurements_list = []
                            for profile_id in export_profiles['id']:
                                measurements = st.session_state.db_manager.get_measurements_by_profile(profile_id)
                                if not measurements.empty:
                                    measurements['profile_id'] = profile_id
                                    measurements_list.append(measurements)
                            
                            if measurements_list:
                                all_measurements = pd.concat(measurements_list, ignore_index=True)
                                # Merge with profile info
                                export_data = export_profiles.merge(
                                    all_measurements, left_on='id', right_on='profile_id'
                                )
                            else:
                                export_data = export_profiles
                        else:
                            export_data = export_profiles
                        
                        # Create download link
                        download_link = create_download_link(export_data, export_format)
                        
                        if download_link:
                            st.success(f"Export ready! {len(export_data)} records prepared.")
                            st.markdown(download_link, unsafe_allow_html=True)
                        else:
                            st.error("Failed to create export file.")
        
        except Exception as e:
            st.error(f"Error preparing export: {str(e)}")
    
    # Footer with current filter summary
    if filters:
        st.sidebar.subheader("Active Filters")
        for key, value in filters.items():
            if isinstance(value, datetime):
                st.sidebar.write(f"**{key}:** {value.strftime('%Y-%m-%d')}")
            else:
                st.sidebar.write(f"**{key}:** {value}")

if __name__ == "__main__":
    main()
