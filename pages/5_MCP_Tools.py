"""
MCP Tools Page - ARGO Oceanographic Platform
Interactive interface for Model Context Protocol tools and advanced data analysis
"""

import streamlit as st
import sys
import os
import asyncio
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.integration import MCPEnhancedRAG, MCPToolHelper
from mcp.client import ArgoMCPClient
from database.connection import DatabaseManager
from config.settings import load_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="MCP Tools - ARGO Platform",
    page_icon="🛠️",
    layout="wide"
)

def initialize_mcp_components():
    """Initialize MCP components"""
    try:
        if 'config' not in st.session_state:
            st.session_state.config = load_config()
        
        if 'db_manager' not in st.session_state:
            st.session_state.db_manager = DatabaseManager(st.session_state.config)
        
        if 'mcp_client' not in st.session_state:
            st.session_state.mcp_client = ArgoMCPClient()
        
        if 'mcp_enhanced_rag' not in st.session_state:
            st.session_state.mcp_enhanced_rag = MCPEnhancedRAG()
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize MCP components: {str(e)}")
        return False

async def run_mcp_tool(tool_name: str, arguments: dict):
    """Run an MCP tool with given arguments"""
    try:
        # Initialize MCP client if needed
        if not hasattr(st.session_state.mcp_client, 'available_tools'):
            await st.session_state.mcp_client.connect()
        
        # Call the tool
        result = await st.session_state.mcp_client.call_tool(tool_name, arguments)
        return result
    except Exception as e:
        return f"Error running tool {tool_name}: {str(e)}"

def display_tool_form(tool_name: str):
    """Display a form for tool parameters"""
    st.subheader(f"🔧 {tool_name.replace('_', ' ').title()}")
    
    tool_descriptions = MCPToolHelper.get_tool_descriptions()
    if tool_name in tool_descriptions:
        st.markdown(f"**Description:** {tool_descriptions[tool_name]}")
    
    # Get parameter schema
    param_schema = MCPToolHelper.format_tool_parameters(tool_name)
    arguments = {}
    
    if tool_name == "query_argo_profiles":
        col1, col2 = st.columns(2)
        with col1:
            lat_min = st.number_input("Minimum Latitude", value=-90.0, min_value=-90.0, max_value=90.0)
            lat_max = st.number_input("Maximum Latitude", value=90.0, min_value=-90.0, max_value=90.0)
            lon_min = st.number_input("Minimum Longitude", value=-180.0, min_value=-180.0, max_value=180.0)
            lon_max = st.number_input("Maximum Longitude", value=180.0, min_value=-180.0, max_value=180.0)
        
        with col2:
            date_start = st.date_input("Start Date", value=datetime(2020, 1, 1))
            date_end = st.date_input("End Date", value=datetime.now())
            limit = st.number_input("Result Limit", value=100, min_value=1, max_value=1000)
        
        arguments = {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "date_start": str(date_start),
            "date_end": str(date_end),
            "limit": int(limit)
        }
    
    elif tool_name == "analyze_temperature_salinity":
        profile_ids_text = st.text_input("Profile IDs (comma-separated)", placeholder="1,2,3,4,5")
        analysis_type = st.selectbox("Analysis Type", ["statistics", "depth_profile", "ts_diagram"])
        
        if profile_ids_text:
            try:
                profile_ids = [int(x.strip()) for x in profile_ids_text.split(',')]
                arguments = {
                    "profile_ids": profile_ids,
                    "analysis_type": analysis_type
                }
            except ValueError:
                st.error("Please enter valid profile IDs (integers separated by commas)")
                return None
    
    elif tool_name == "search_oceanographic_data":
        query = st.text_input("Search Query", placeholder="Enter your search terms...")
        top_k = st.number_input("Number of Results", value=5, min_value=1, max_value=20)
        
        if query:
            arguments = {
                "query": query,
                "top_k": int(top_k)
            }
    
    elif tool_name == "get_float_trajectory":
        float_id = st.text_input("Float ID", placeholder="2902746")
        use_cycle_range = st.checkbox("Specify Cycle Range")
        
        if use_cycle_range:
            col1, col2 = st.columns(2)
            with col1:
                cycle_start = st.number_input("Start Cycle", value=1, min_value=1)
            with col2:
                cycle_end = st.number_input("End Cycle", value=100, min_value=1)
            
            if float_id:
                arguments = {
                    "float_id": float_id,
                    "cycle_range": [int(cycle_start), int(cycle_end)]
                }
        else:
            if float_id:
                arguments = {
                    "float_id": float_id
                }
    
    elif tool_name == "calculate_water_mass_properties":
        profile_ids_text = st.text_input("Profile IDs (comma-separated)", placeholder="1,2,3,4,5")
        property_type = st.selectbox("Property Type", ["density", "potential_temperature", "mixed_layer_depth"])
        
        if profile_ids_text:
            try:
                profile_ids = [int(x.strip()) for x in profile_ids_text.split(',')]
                arguments = {
                    "profile_ids": profile_ids,
                    "property_type": property_type
                }
            except ValueError:
                st.error("Please enter valid profile IDs (integers separated by commas)")
                return None
    
    return arguments

def main():
    """Main MCP Tools interface"""
    
    st.title("🛠️ MCP Tools Dashboard")
    st.markdown("Model Context Protocol tools for advanced oceanographic data analysis")
    
    # Initialize components
    if not initialize_mcp_components():
        st.stop()
    
    # Sidebar with MCP status and resources
    with st.sidebar:
        st.subheader("🔗 MCP Connection Status")
        
        # Check MCP status
        try:
            # Test connection
            if asyncio.run(st.session_state.mcp_client.connect()):
                st.success("✅ MCP Client Connected")
            else:
                st.error("❌ MCP Client Disconnected")
        except Exception as e:
            st.error(f"❌ Connection Error: {str(e)}")
        
        st.subheader("📊 Available Resources")
        
        # Display available tools
        tools = MCPToolHelper.get_tool_descriptions()
        st.write(f"**Available Tools:** {len(tools)}")
        for tool_name in tools.keys():
            st.write(f"• {tool_name.replace('_', ' ').title()}")
        
        # Database summary
        with st.expander("📋 Database Summary"):
            try:
                summary_result = asyncio.run(st.session_state.mcp_client.read_resource("argo://profiles/summary"))
                summary_data = json.loads(summary_result)
                
                for key, value in summary_data.items():
                    st.metric(key.replace('_', ' ').title(), str(value))
            except Exception as e:
                st.write(f"Error loading summary: {str(e)}")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Tool Explorer", "🤖 Natural Language", "📈 Batch Operations", "📚 Resources"])
    
    with tab1:
        st.subheader("Interactive Tool Explorer")
        st.markdown("Use individual MCP tools with custom parameters")
        
        # Tool selection
        available_tools = list(MCPToolHelper.get_tool_descriptions().keys())
        selected_tool = st.selectbox("Select Tool", available_tools)
        
        if selected_tool:
            # Display tool form
            arguments = display_tool_form(selected_tool)
            
            if arguments and st.button(f"Run {selected_tool.replace('_', ' ').title()}", type="primary"):
                with st.spinner("Executing tool..."):
                    try:
                        result = asyncio.run(run_mcp_tool(selected_tool, arguments))
                        
                        st.subheader("📋 Results")
                        
                        # Try to parse as JSON for better display
                        try:
                            if result.startswith("Found") or result.startswith("Float") or result.startswith("Search") or result.startswith("Statistical") or result.startswith("Density"):
                                # Handle formatted string results
                                st.text_area("Tool Output", value=result, height=300)
                            else:
                                # Try to parse as JSON
                                json_data = json.loads(result)
                                st.json(json_data)
                        except (json.JSONDecodeError, AttributeError):
                            # Display as text if not JSON
                            st.text_area("Tool Output", value=str(result), height=300)
                        
                        # Show execution details
                        with st.expander("🔍 Execution Details"):
                            st.write(f"**Tool:** {selected_tool}")
                            st.write(f"**Arguments:** {arguments}")
                            st.write(f"**Execution Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    except Exception as e:
                        st.error(f"Error executing tool: {str(e)}")
    
    with tab2:
        st.subheader("Natural Language Query Interface")
        st.markdown("Ask questions in natural language - MCP tools will be automatically selected and executed")
        
        # Query input
        user_query = st.text_area("Enter your question", placeholder="Example: Show me temperature profiles between latitude 10 and 20 degrees")
        
        if st.button("Process Query", type="primary") and user_query:
            with st.spinner("Processing query with MCP tools..."):
                try:
                    # Use the MCP Enhanced RAG system
                    response = asyncio.run(st.session_state.mcp_enhanced_rag.process_query(user_query))
                    
                    st.subheader("🤖 AI Response")
                    st.markdown(response)
                    
                    # Show query analysis
                    with st.expander("🧠 Query Analysis"):
                        st.write("The system analyzed your query and automatically selected appropriate MCP tools to gather the data needed for a comprehensive response.")
                
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
    
    with tab3:
        st.subheader("Batch Operations")
        st.markdown("Run multiple tools or operations in sequence")
        
        # Batch operation builder
        st.write("**Build a sequence of operations:**")
        
        if 'batch_operations' not in st.session_state:
            st.session_state.batch_operations = []
        
        # Add operation
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            batch_tool = st.selectbox("Tool", available_tools, key="batch_tool")
        with col2:
            batch_name = st.text_input("Operation Name", placeholder="My Operation", key="batch_name")
        with col3:
            if st.button("Add", key="add_batch"):
                if batch_name:
                    st.session_state.batch_operations.append({
                        'name': batch_name,
                        'tool': batch_tool,
                        'status': 'pending'
                    })
                    st.rerun()
        
        # Display current batch
        if st.session_state.batch_operations:
            st.subheader("📝 Batch Queue")
            
            for i, op in enumerate(st.session_state.batch_operations):
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                with col1:
                    st.write(f"**{op['name']}**")
                with col2:
                    st.write(op['tool'])
                with col3:
                    status_color = {"pending": "🟡", "completed": "🟢", "error": "🔴"}
                    st.write(f"{status_color.get(op['status'], '⚪')} {op['status']}")
                with col4:
                    if st.button("❌", key=f"remove_{i}"):
                        st.session_state.batch_operations.pop(i)
                        st.rerun()
            
            # Batch controls
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Run Batch", type="primary"):
                    with st.spinner("Running batch operations..."):
                        for i, op in enumerate(st.session_state.batch_operations):
                            try:
                                # This is a simplified batch execution
                                # In a real implementation, you'd collect parameters for each operation
                                result = f"Batch operation {op['name']} would run {op['tool']}"
                                st.session_state.batch_operations[i]['status'] = 'completed'
                            except Exception as e:
                                st.session_state.batch_operations[i]['status'] = 'error'
                        st.rerun()
            
            with col2:
                if st.button("🗑️ Clear All"):
                    st.session_state.batch_operations = []
                    st.rerun()
    
    with tab4:
        st.subheader("MCP Resources")
        st.markdown("Access structured data resources through MCP")
        
        # Resource selection
        resources = [
            ("argo://profiles/summary", "ARGO Profiles Summary"),
            ("argo://floats/active", "Active ARGO Floats"),
            ("argo://data/schema", "Database Schema")
        ]
        
        selected_resource = st.selectbox(
            "Select Resource",
            resources,
            format_func=lambda x: x[1]
        )
        
        if st.button("Load Resource", type="primary"):
            with st.spinner("Loading resource..."):
                try:
                    resource_data = asyncio.run(st.session_state.mcp_client.read_resource(selected_resource[0]))
                    
                    st.subheader(f"📊 {selected_resource[1]}")
                    
                    # Try to parse and display as JSON
                    try:
                        json_data = json.loads(resource_data)
                        st.json(json_data)
                    except json.JSONDecodeError:
                        st.text_area("Resource Data", value=resource_data, height=400)
                
                except Exception as e:
                    st.error(f"Error loading resource: {str(e)}")
        
        # MCP Documentation
        with st.expander("📖 About MCP Integration"):
            st.markdown("""
            ### Model Context Protocol (MCP) Integration
            
            This ARGO platform now includes MCP (Model Context Protocol) integration, providing:
            
            **🛠️ Advanced Tools:**
            - Structured data queries with parameters
            - Oceanographic calculations and analysis
            - Semantic search capabilities
            - Float trajectory analysis
            - Water mass property calculations
            
            **🤖 AI Enhancement:**
            - Natural language tool selection
            - Automatic parameter extraction
            - Context-aware responses
            - Multi-tool orchestration
            
            **📊 Resource Access:**
            - Real-time database summaries
            - Float status information
            - Schema documentation
            - Structured data exports
            
            **Benefits:**
            - More precise data queries
            - Better integration with AI chat
            - Standardized tool interfaces
            - Enhanced error handling
            """)

if __name__ == "__main__":
    main()