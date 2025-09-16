import streamlit as st
import sys
import os
from dotenv import load_dotenv
from config.settings import load_config
load_dotenv()
# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import load_config
from database.connection import DatabaseManager
from vector_store.faiss_manager import FAISSManager

# Page configuration
st.set_page_config(
    page_title="FloatChat",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_app():
    """Initialize the application components"""
    try:
        # Load configuration
        config = load_config()
        
        # Initialize database connection
        if 'db_manager' not in st.session_state:
            st.session_state.db_manager = DatabaseManager(config)
            
        # Initialize vector store
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = FAISSManager()
            
        # Initialize session state variables
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
            
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
        return True
        
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        st.error("Please check your configuration and database connection.")
        return False

def main():
    """Main application entry point"""
    
    # Initialize the application
    if not initialize_app():
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("🌊 ARGO Platform")
    st.sidebar.markdown("---")
    
    # Main content area
    st.title("ARGO Oceanographic Data Analysis Platform")
    
    st.markdown("""
    Welcome to the AI-powered oceanographic data analysis platform for ARGO float data. 
    This platform enables you to:
    
    - 📊 **Ingest and Process** ARGO NetCDF data files
    - 🔍 **Explore** oceanographic datasets with interactive tools
    - 🤖 **Query** data using natural language with AI assistance
    - 📈 **Visualize** ocean measurements and float trajectories
    """)
    
    # Quick stats in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Database Status", "Connected" if st.session_state.get('db_manager') else "Disconnected")
    
    with col2:
        st.metric("Vector Store", "Ready" if st.session_state.get('vector_store') else "Not Ready")
    
    with col3:
        # Get data count from database
        try:
            db_manager = st.session_state.get('db_manager')
            if db_manager:
                count = db_manager.get_total_records()
                st.metric("Total Records", f"{count:,}")
            else:
                st.metric("Total Records", "0")
        except:
            st.metric("Total Records", "N/A")
    
    with col4:
        st.metric("AI Status", "Groq Ready" if os.getenv('GROQ_API_KEY') else "API Key Missing")
    
    st.markdown("---")
    
    # Recent activity or quick start guide
    st.subheader("🚀 Quick Start Guide")
    
    steps = [
        ("1. Data Ingestion", "Upload ARGO NetCDF files to begin analysis", "📁"),
        ("2. Data Explorer", "Browse and filter your oceanographic data", "🔍"),
        ("3. AI Chat", "Ask questions about your data in natural language", "🤖"),
        ("4. Visualizations", "Create interactive maps and scientific plots", "📊")
    ]
    
    for step, description, icon in steps:
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"### {icon}")
            with col2:
                st.markdown(f"**{step}**")
                st.markdown(description)
    
    st.markdown("---")
    
    # System information
    with st.expander("ℹ️ System Information"):
        st.markdown("**Configuration Status:**")
        
        config_status = {
            "Groq API Key": "✅ Configured" if os.getenv('GROQ_API_KEY') else "❌ Missing",
            "Database URL": "✅ Configured" if os.getenv('DATABASE_URL') or os.getenv('PGHOST') else "❌ Missing",
            "PostgreSQL Host": os.getenv('PGHOST', 'Not configured'),
            "PostgreSQL Database": os.getenv('PGDATABASE', 'Not configured'),
        }
        
        for key, value in config_status.items():
            st.markdown(f"- **{key}**: {value}")
    
    # Navigation instructions
    st.info("👈 Use the sidebar to navigate between different sections of the platform.")

if __name__ == "__main__":
    main()
