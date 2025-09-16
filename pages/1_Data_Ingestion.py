import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import tempfile
from datetime import datetime
from data_processing.netcdf_processor import NetCDFProcessor
from database.connection import DatabaseManager
from vector_store.faiss_manager import FAISSManager
from data_processing.data_transformer import DataTransformer
from config.settings import load_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Data Ingestion - ARGO Platform",
    page_icon="📁",
    layout="wide"
)

def initialize_components():
    """Initialize application components"""
    try:
        if 'config' not in st.session_state:
            st.session_state.config = load_config()
        
        if 'db_manager' not in st.session_state:
            st.session_state.db_manager = DatabaseManager(st.session_state.config)
        
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = FAISSManager()
            
        if 'netcdf_processor' not in st.session_state:
            st.session_state.netcdf_processor = NetCDFProcessor()
            
        if 'data_transformer' not in st.session_state:
            st.session_state.data_transformer = DataTransformer()
            
        return True
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        return False

def process_uploaded_file(uploaded_file):
    """Process uploaded NetCDF file"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Process the file
        processor = st.session_state.netcdf_processor
        transformer = st.session_state.data_transformer
        
        # Extract data from NetCDF
        profile_metadata, measurements = processor.process_file(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return profile_metadata, measurements
        
    except Exception as e:
        logger.error(f"Failed to process uploaded file: {str(e)}")
        raise

def store_data_in_database(profile_metadata, measurements):
    """Store processed data in database and vector store"""
    try:
        db_manager = st.session_state.db_manager
        vector_store = st.session_state.vector_store
        transformer = st.session_state.data_transformer
        
        # Insert profile into database
        profile_id = db_manager.insert_profile(profile_metadata)
        
        if measurements:
            # Convert measurements to DataFrame for processing
            measurements_df = pd.DataFrame(measurements)
            
            # Clean and transform measurements
            cleaned_measurements = transformer.clean_measurements(measurements_df)
            cleaned_measurements = transformer.interpolate_missing_depth(cleaned_measurements)
            
            # Convert back to list of dictionaries for database insertion
            clean_measurements_list = cleaned_measurements.to_dict('records')
            
            # Insert measurements
            db_manager.insert_measurements(profile_id, clean_measurements_list)
            
            # Create profile summary for vector store
            profile_summary = transformer.create_profile_summary(cleaned_measurements, profile_metadata)
            
            # Add to vector store
            vector_store.add_profile(profile_summary, profile_id)
            vector_store.save_index()
        
        return profile_id
        
    except Exception as e:
        logger.error(f"Failed to store data: {str(e)}")
        raise

def main():
    """Main data ingestion interface"""
    
    st.title("📁 ARGO Data Ingestion")
    st.markdown("Upload and process ARGO NetCDF files to add them to the database.")
    
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # File upload section
    st.subheader("Upload NetCDF Files")
    
    uploaded_files = st.file_uploader(
        "Choose NetCDF files",
        type=['nc', 'netcdf'],
        accept_multiple_files=True,
        help="Upload ARGO NetCDF files for processing and storage"
    )
    
    if uploaded_files:
        st.info(f"Selected {len(uploaded_files)} file(s) for processing")
        
        # Processing options
        col1, col2 = st.columns(2)
        
        with col1:
            skip_duplicates = st.checkbox("Skip duplicate files", value=True, 
                                        help="Skip files that have already been processed")
        
        with col2:
            validate_data = st.checkbox("Validate data quality", value=True,
                                      help="Perform data quality checks during processing")
        
        # Process button
        if st.button("Process Files", type="primary"):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            total_files = len(uploaded_files)
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Process the file
                    profile_metadata, measurements = process_uploaded_file(uploaded_file)
                    
                    # Check for duplicates if option is selected
                    if skip_duplicates:
                        existing_profile = st.session_state.db_manager.get_profile_id_by_hash(
                            profile_metadata['file_hash']
                        )
                        if existing_profile:
                            results.append({
                                'file': uploaded_file.name,
                                'status': 'skipped',
                                'message': 'Duplicate file (already processed)',
                                'profile_id': existing_profile
                            })
                            progress_bar.progress((i + 1) / total_files)
                            continue
                    
                    # Store in database and vector store
                    profile_id = store_data_in_database(profile_metadata, measurements)
                    
                    results.append({
                        'file': uploaded_file.name,
                        'status': 'success',
                        'message': f'Successfully processed {len(measurements)} measurements',
                        'profile_id': profile_id,
                        'float_id': profile_metadata.get('float_id', 'N/A'),
                        'cycle_number': profile_metadata.get('cycle_number', 'N/A'),
                        'measurement_count': len(measurements)
                    })
                    
                except Exception as e:
                    results.append({
                        'file': uploaded_file.name,
                        'status': 'error',
                        'message': str(e),
                        'profile_id': None
                    })
                
                # Update progress
                progress_bar.progress((i + 1) / total_files)
            
            status_text.text("Processing complete!")
            
            # Display results
            st.subheader("Processing Results")
            
            # Summary statistics
            success_count = sum(1 for r in results if r['status'] == 'success')
            error_count = sum(1 for r in results if r['status'] == 'error')
            skipped_count = sum(1 for r in results if r['status'] == 'skipped')
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Files", total_files)
            with col2:
                st.metric("Successful", success_count, delta=success_count)
            with col3:
                st.metric("Errors", error_count, delta=error_count if error_count == 0 else -error_count)
            with col4:
                st.metric("Skipped", skipped_count)
            
            # Detailed results table
            if results:
                results_df = pd.DataFrame(results)
                
                # Color code the status
                def color_status(val):
                    if val == 'success':
                        return 'background-color: #d4edda'
                    elif val == 'error':
                        return 'background-color: #f8d7da'
                    elif val == 'skipped':
                        return 'background-color: #fff3cd'
                    return ''
                
                styled_df = results_df.style.applymap(color_status, subset=['status'])
                st.dataframe(styled_df, use_container_width=True)
    
    # Database statistics
    st.subheader("Database Statistics")
    
    try:
        stats = st.session_state.db_manager.get_summary_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Profiles", stats.get('total_profiles', 0))
        
        with col2:
            st.metric("Total Measurements", stats.get('total_measurements', 0))
        
        with col3:
            st.metric("Unique Floats", stats.get('unique_floats', 0))
        
        with col4:
            # Get vector store statistics
            vector_stats = st.session_state.vector_store.get_statistics()
            st.metric("Vector Embeddings", vector_stats.get('total_profiles', 0))
        
        # Date range and geographic coverage
        if stats.get('date_range'):
            date_range = stats['date_range']
            if date_range['earliest'] and date_range['latest']:
                st.write(f"**Date Range:** {date_range['earliest'].strftime('%Y-%m-%d')} to {date_range['latest'].strftime('%Y-%m-%d')}")
        
        if stats.get('geographic_coverage'):
            geo = stats['geographic_coverage']
            if all(v is not None for v in geo.values()):
                st.write(f"**Geographic Coverage:** {geo['min_latitude']:.2f}°N to {geo['max_latitude']:.2f}°N, "
                        f"{geo['min_longitude']:.2f}°E to {geo['max_longitude']:.2f}°E")
        
    except Exception as e:
        st.error(f"Failed to load database statistics: {str(e)}")
    
    # File validation section
    st.subheader("File Validation")
    st.markdown("Upload a file to check if it's a valid ARGO NetCDF file without processing it.")
    
    validation_file = st.file_uploader(
        "Choose file for validation",
        type=['nc', 'netcdf'],
        key="validation_uploader"
    )
    
    if validation_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
                tmp_file.write(validation_file.read())
                tmp_file_path = tmp_file.name
            
            # Get file summary
            processor = st.session_state.netcdf_processor
            summary = processor.get_file_summary(tmp_file_path)
            
            # Clean up
            os.unlink(tmp_file_path)
            
            if 'error' in summary:
                st.error(f"Validation Error: {summary['error']}")
            else:
                st.success("✅ Valid ARGO NetCDF file")
                
                # Display file information
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**File Information:**")
                    st.write(f"- File size: {summary.get('file_size', 0):,} bytes")
                    st.write(f"- Dimensions: {summary.get('dimensions', {})}")
                    
                    if 'float_id' in summary:
                        st.write(f"- Float ID: {summary['float_id']}")
                    if 'cycle_number' in summary:
                        st.write(f"- Cycle: {summary['cycle_number']}")
                
                with col2:
                    st.write("**Variables:**")
                    variables = summary.get('variables', [])
                    if variables:
                        for var in variables[:10]:  # Show first 10 variables
                            st.write(f"- {var}")
                        if len(variables) > 10:
                            st.write(f"... and {len(variables) - 10} more")
                    
                # Global attributes
                if 'global_attributes' in summary and summary['global_attributes']:
                    st.write("**Global Attributes:**")
                    attrs = summary['global_attributes']
                    for key, value in list(attrs.items())[:5]:  # Show first 5 attributes
                        st.write(f"- {key}: {value}")
                    if len(attrs) > 5:
                        st.write(f"... and {len(attrs) - 5} more attributes")
        
        except Exception as e:
            st.error(f"Validation failed: {str(e)}")
    
    # Help section
    with st.expander("ℹ️ Help and Information"):
        st.markdown("""
        ### About ARGO NetCDF Files
        
        ARGO NetCDF files contain oceanographic profile data from autonomous floats. These files should include:
        
        **Required Variables:**
        - `PRES` - Pressure measurements (decibars)
        - `TEMP` - Temperature measurements (Celsius)
        - `PSAL` - Practical salinity (PSU)
        
        **Optional BGC Variables:**
        - `DOXY` - Dissolved oxygen
        - `NITRATE` - Nitrate concentration
        - `PH_IN_SITU_TOTAL` - pH measurements
        - `CHLA` - Chlorophyll-a concentration
        
        **Metadata Variables:**
        - `LATITUDE`, `LONGITUDE` - Profile location
        - `JULD` - Julian day (date/time)
        - `PLATFORM_NUMBER` - Float identifier
        - `CYCLE_NUMBER` - Profile cycle number
        
        ### Data Quality
        
        The system automatically:
        - Validates file format and required variables
        - Checks for reasonable parameter ranges
        - Filters data based on quality flags
        - Detects and skips duplicate files
        - Creates searchable metadata summaries
        
        ### Troubleshooting
        
        **Common Issues:**
        - Ensure files have `.nc` or `.netcdf` extension
        - Check that required variables are present
        - Verify file is not corrupted
        - Make sure database connection is working
        """)

if __name__ == "__main__":
    main()
