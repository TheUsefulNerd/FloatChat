import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from database.connection import DatabaseManager
from vector_store.faiss_manager import FAISSManager
from rag.groq_rag import GroqRAGSystem
from rag.query_processor import QueryProcessor
from mcp.integration import MCPEnhancedRAG, MCPToolHelper
from visualization.plots import OceanographicPlots
from visualization.maps import OceanographicMaps
from config.settings import load_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Chat - ARGO Platform",
    page_icon="🤖",
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
        
        if 'rag_system' not in st.session_state:
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                st.error("Groq API key not found. Please set GROQ_API_KEY environment variable.")
                return False
            # Initialize basic Groq RAG system
            groq_rag = GroqRAGSystem(api_key)
            # Initialize MCP Enhanced RAG system
            st.session_state.rag_system = MCPEnhancedRAG(groq_rag)
        
        if 'query_processor' not in st.session_state:
            st.session_state.query_processor = QueryProcessor()
        
        if 'plotter' not in st.session_state:
            st.session_state.plotter = OceanographicPlots()
            
        if 'mapper' not in st.session_state:
            st.session_state.mapper = OceanographicMaps()
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        return False

async def process_query(user_question):
    """Process user query using MCP Enhanced RAG system"""
    try:
        # Use MCP Enhanced RAG for processing
        answer = await st.session_state.rag_system.process_query(user_question)
        
        # Analyze the query for visualization purposes
        query_analysis = st.session_state.query_processor.analyze_query(user_question)
        
        # Search vector database for additional context if needed
        try:
            search_results = st.session_state.vector_store.search(user_question, k=5)
        except:
            search_results = []
        
        return {
            'answer': answer,
            'query_analysis': query_analysis,
            'search_results': search_results,
            'relevant_data': search_results,
            'mcp_enhanced': True
        }
        
    except Exception as e:
        logger.error(f"Failed to process query: {str(e)}")
        return {
            'answer': f"I encountered an error while processing your question: {str(e)}",
            'query_analysis': {},
            'search_results': [],
            'relevant_data': [],
            'mcp_enhanced': False
        }

def generate_visualizations(query_analysis, search_results):
    """Generate appropriate visualizations based on query"""
    visualizations = []
    
    try:
        if not search_results:
            return visualizations
        
        # Extract profile IDs from search results
        profile_ids = [result.get('profile_id') for result in search_results if result.get('profile_id')]
        
        if not profile_ids:
            return visualizations
        
        # Get profile and measurement data
        profiles_list = []
        measurements_list = []
        
        for profile_id in profile_ids[:10]:  # Limit to first 10 for performance
            try:
                # Get profile info
                profile_data = st.session_state.db_manager.get_profiles(
                    filters={'profile_ids': [profile_id]}, limit=1
                )
                if not profile_data.empty:
                    profiles_list.append(profile_data)
                
                # Get measurements
                measurements = st.session_state.db_manager.get_measurements_by_profile(profile_id)
                if not measurements.empty:
                    measurements['profile_id'] = profile_id
                    measurements_list.append(measurements)
            except Exception as e:
                logger.warning(f"Failed to get data for profile {profile_id}: {str(e)}")
                continue
        
        if profiles_list:
            all_profiles = pd.concat(profiles_list, ignore_index=True)
            
            # Geographic visualization
            if query_analysis.get('query_type') in ['location_search', 'general_search']:
                map_viz = st.session_state.mapper.create_float_trajectory_map(all_profiles)
                visualizations.append({
                    'type': 'map',
                    'title': 'Float Locations',
                    'content': map_viz
                })
        
        if measurements_list:
            all_measurements = pd.concat(measurements_list, ignore_index=True)
            
            # Parameter-specific visualizations
            parameters = query_analysis.get('parameters', [])
            
            if 'temperature' in parameters or 'salinity' in parameters:
                # T-S diagram if both are available
                if 'temperature' in all_measurements.columns and 'salinity' in all_measurements.columns:
                    ts_plot = st.session_state.plotter.create_ts_diagram(all_measurements)
                    visualizations.append({
                        'type': 'plot',
                        'title': 'Temperature-Salinity Diagram',
                        'content': ts_plot
                    })
            
            # Depth profiles for requested parameters
            available_params = [p for p in parameters 
                              if p in all_measurements.columns and all_measurements[p].notna().any()]
            
            if available_params:
                depth_profile = st.session_state.plotter.create_depth_profile(
                    all_measurements, available_params[:3], "Query Results - Depth Profiles"
                )
                visualizations.append({
                    'type': 'plot',
                    'title': 'Depth Profiles',
                    'content': depth_profile
                })
            
            # Time series if temporal analysis
            if query_analysis.get('query_type') == 'temporal_analysis' and profiles_list:
                for param in available_params[:2]:  # Limit to 2 parameters
                    if param in all_measurements.columns:
                        # Create time series data
                        time_series_data = []
                        for measurements_df in measurements_list:
                            if param in measurements_df.columns:
                                profile_id = measurements_df['profile_id'].iloc[0]
                                profile_info = all_profiles[all_profiles['id'] == profile_id]
                                if not profile_info.empty:
                                    mean_value = measurements_df[param].mean()
                                    if not pd.isna(mean_value):
                                        time_series_data.append({
                                            'measurement_date': profile_info['measurement_date'].iloc[0],
                                            param: mean_value
                                        })
                        
                        if time_series_data:
                            time_series_df = pd.DataFrame(time_series_data)
                            ts_plot = st.session_state.plotter.create_time_series(time_series_df, param)
                            visualizations.append({
                                'type': 'plot',
                                'title': f'{param.title()} Time Series',
                                'content': ts_plot
                            })
    
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {str(e)}")
    
    return visualizations

def display_chat_message(role, content, timestamp=None):
    """Display a chat message with proper formatting"""
    if timestamp is None:
        timestamp = datetime.now()
    
    with st.chat_message(role):
        if role == "user":
            st.write(content)
        else:
            st.markdown(content)
        
        # Show timestamp in smaller text
        st.caption(f"{timestamp.strftime('%H:%M:%S')}")

def main():
    """Main AI chat interface"""
    
    st.title("🤖 AI Chat Assistant")
    st.markdown("Ask questions about ARGO oceanographic data in natural language.")
    
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # Sidebar with example queries and tips
    with st.sidebar:
        st.subheader("🚀 MCP Enhanced AI")
        st.success("Now powered by Model Context Protocol (MCP) for advanced oceanographic analysis!")
        
        # MCP Tools section
        with st.expander("🛠️ Available Tools"):
            tool_descriptions = MCPToolHelper.get_tool_descriptions()
            for tool_name, description in tool_descriptions.items():
                st.write(f"**{tool_name.replace('_', ' ').title()}:** {description}")
        
        st.subheader("💡 Example Questions")
        
        example_queries = [
            "Show me temperature profiles in the Arabian Sea",
            "What are the salinity measurements near 20°N, 65°E?",
            "Compare oxygen levels in the Indian Ocean over the last year",
            "Find profiles with temperature greater than 25°C",
            "Show me data from float 2902746",
            "What is the average salinity at 500m depth?",
            "Explain mixed layer depth",
            "Show me BGC parameters in the equatorial region",
            "Analyze profiles between latitude 10 and 20",
            "Calculate water density for profile 12345",
            "Get trajectory for float 2902746",
            "Search for high temperature anomalies"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{hash(query)}", use_container_width=True):
                st.session_state.user_input = query
                st.rerun()
        
        st.subheader("🔧 Query Tips")
        st.markdown("""
        **Location formats:**
        - "near 20°N, 65°E"
        - "in the Arabian Sea"
        - "within 100km of coordinates"
        
        **Parameter names:**
        - temperature, temp
        - salinity, salt
        - oxygen, dissolved oxygen
        - nitrate, nitrogen
        - pH, acidity
        - chlorophyll, chla
        
        **Time expressions:**
        - "in March 2023"
        - "last 6 months"
        - "between 2020 and 2022"
        
        **Comparisons:**
        - "compare X and Y"
        - "temperature vs depth"
        - "before and after"
        """)
    
    # Chat interface
    st.subheader("💬 Chat with ARGO Data")
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(
            message['role'], 
            message['content'], 
            message.get('timestamp')
        )
    
    # Chat input
    user_input = st.chat_input("Ask a question about ARGO data...")
    
    # Handle example query selection
    if hasattr(st.session_state, 'user_input'):
        user_input = st.session_state.user_input
        delattr(st.session_state, 'user_input')
    
    if user_input:
        # Display user message
        timestamp = datetime.now()
        display_chat_message("user", user_input, timestamp)
        
        # Add to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': timestamp
        })
        
        # Process query and generate response
        with st.spinner("🤔 Analyzing with MCP tools..."):
            try:
                # Process the query with async support
                import asyncio
                response_data = asyncio.run(process_query(user_input))
                
                # Display AI response
                ai_timestamp = datetime.now()
                display_chat_message("assistant", response_data['answer'], ai_timestamp)
                
                # Add to chat history with MCP indicator
                mcp_indicator = " 🚀" if response_data.get('mcp_enhanced') else ""
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response_data['answer'] + mcp_indicator,
                    'timestamp': ai_timestamp
                })
                
                # Generate and display visualizations
                if response_data['search_results']:
                    st.subheader("📊 Related Visualizations")
                    
                    visualizations = generate_visualizations(
                        response_data['query_analysis'],
                        response_data['search_results']
                    )
                    
                    if visualizations:
                        # Display visualizations in tabs or columns
                        if len(visualizations) == 1:
                            viz = visualizations[0]
                            st.subheader(viz['title'])
                            if viz['type'] == 'plot':
                                st.plotly_chart(viz['content'], use_container_width=True)
                            elif viz['type'] == 'map':
                                st.components.v1.html(viz['content']._repr_html_(), height=500)
                        
                        elif len(visualizations) > 1:
                            # Create tabs for multiple visualizations
                            tab_names = [viz['title'] for viz in visualizations]
                            tabs = st.tabs(tab_names)
                            
                            for tab, viz in zip(tabs, visualizations):
                                with tab:
                                    if viz['type'] == 'plot':
                                        st.plotly_chart(viz['content'], use_container_width=True)
                                    elif viz['type'] == 'map':
                                        st.components.v1.html(viz['content']._repr_html_(), height=500)
                
                # Show relevant data sources
                if response_data['search_results']:
                    with st.expander("🔍 Data Sources Used"):
                        st.write(f"Found {len(response_data['search_results'])} relevant profiles:")
                        
                        for i, result in enumerate(response_data['search_results'][:5], 1):
                            summary = result.get('summary', {})
                            st.write(f"**{i}.** Float {summary.get('float_id', 'Unknown')} - "
                                   f"Similarity: {result.get('similarity_score', 0):.3f}")
                            if 'search_text' in result:
                                st.caption(result['search_text'][:200] + "...")
                
                # Query analysis details
                if response_data['query_analysis']:
                    with st.expander("🧠 Query Analysis"):
                        analysis = response_data['query_analysis']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Query Type:** {analysis.get('query_type', 'unknown')}")
                            st.write(f"**Parameters:** {', '.join(analysis.get('parameters', []))}")
                        
                        with col2:
                            if analysis.get('location'):
                                st.write(f"**Location:** {analysis['location']}")
                            if analysis.get('time_range'):
                                st.write(f"**Time Range:** {analysis['time_range']}")
                
            except Exception as e:
                error_msg = f"I encountered an error while processing your question: {str(e)}"
                display_chat_message("assistant", error_msg, datetime.now())
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': error_msg,
                    'timestamp': datetime.now()
                })
        
        # Rerun to update the display
        st.rerun()
    
    # Chat management
    st.subheader("🔧 Chat Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("Export Chat", type="secondary"):
            if st.session_state.chat_history:
                chat_export = []
                for msg in st.session_state.chat_history:
                    chat_export.append(f"**{msg['role'].title()}** ({msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S')})")
                    chat_export.append(msg['content'])
                    chat_export.append("")
                
                chat_text = "\n".join(chat_export)
                st.download_button(
                    "Download Chat History",
                    chat_text,
                    file_name=f"argo_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    with col3:
        # Show chat statistics
        if st.session_state.chat_history:
            user_messages = len([msg for msg in st.session_state.chat_history if msg['role'] == 'user'])
            st.metric("Messages", f"{user_messages} questions")
    
    # Educational section
    with st.expander("🎓 Learn About ARGO Data"):
        st.markdown("""
        ### What is ARGO?
        
        ARGO is a global network of autonomous profiling floats that measure temperature, salinity, and pressure 
        in the world's oceans. Some floats also measure biogeochemical parameters.
        
        ### Key Parameters:
        
        - **Temperature**: Sea water temperature in degrees Celsius
        - **Salinity**: Practical salinity in PSU (Practical Salinity Units)
        - **Pressure**: Water pressure in decibars (approximates depth in meters)
        - **Oxygen**: Dissolved oxygen in micromoles per kilogram
        - **Nitrate**: Nitrate concentration in micromoles per kilogram
        - **pH**: Acidity/alkalinity of seawater
        - **Chlorophyll**: Chlorophyll-a concentration indicating phytoplankton
        
        ### Oceanographic Concepts:
        
        - **Mixed Layer Depth**: Depth of the well-mixed upper ocean layer
        - **Thermocline**: Layer where temperature changes rapidly with depth
        - **Water Masses**: Bodies of water with distinct temperature and salinity characteristics
        - **BGC Parameters**: Biogeochemical measurements related to ocean biology and chemistry
        
        ### Ask Questions Like:
        
        - "Explain mixed layer depth"
        - "What causes ocean acidification?"
        - "How do temperature and salinity relate?"
        - "What are water masses?"
        """)

if __name__ == "__main__":
    main()
