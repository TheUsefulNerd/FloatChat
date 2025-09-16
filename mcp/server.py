"""
MCP Server for ARGO Oceanographic Platform
Provides tools and resources for AI agents to interact with oceanographic data
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from config.settings import load_config
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Import platform components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import DatabaseManager
from data_processing.netcdf_processor import NetCDFProcessor
from rag.groq_rag import GroqRAGSystem
from vector_store.faiss_manager import FAISSManager

logger = logging.getLogger("mcp_argo_server")

class ArgoMCPServer:
    """MCP Server for ARGO Oceanographic Platform"""
    
    def __init__(self):
        self.server = Server("argo-oceanographic")
        self.db_manager = DatabaseManager(config)
        self.netcdf_processor = NetCDFProcessor()
        self.rag_system = None
        self.vector_store = None
        
        # Initialize components
        self._setup_tools()
        self._setup_resources()
    
    def _setup_tools(self):
        """Register MCP tools for oceanographic data analysis"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools for oceanographic data analysis"""
            return [
                Tool(
                    name="query_argo_profiles",
                    description="Query ARGO float profiles by location, date range, or other criteria",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "lat_min": {"type": "number", "description": "Minimum latitude"},
                            "lat_max": {"type": "number", "description": "Maximum latitude"},
                            "lon_min": {"type": "number", "description": "Minimum longitude"},
                            "lon_max": {"type": "number", "description": "Maximum longitude"},
                            "date_start": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                            "date_end": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                            "max_depth": {"type": "number", "description": "Maximum depth in meters"},
                            "limit": {"type": "integer", "description": "Maximum number of profiles", "default": 100}
                        }
                    }
                ),
                Tool(
                    name="analyze_temperature_salinity",
                    description="Analyze temperature and salinity data for specific profiles",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "profile_ids": {"type": "array", "items": {"type": "integer"}, "description": "Profile IDs to analyze"},
                            "analysis_type": {"type": "string", "enum": ["depth_profile", "ts_diagram", "statistics"], "description": "Type of analysis"}
                        },
                        "required": ["profile_ids", "analysis_type"]
                    }
                ),
                Tool(
                    name="search_oceanographic_data",
                    description="Semantic search through oceanographic data using natural language",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Natural language query"},
                            "top_k": {"type": "integer", "description": "Number of results", "default": 5}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_float_trajectory",
                    description="Get trajectory data for specific ARGO floats",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "float_id": {"type": "string", "description": "ARGO float ID"},
                            "cycle_range": {"type": "array", "items": {"type": "integer"}, "description": "Cycle number range [start, end]"}
                        },
                        "required": ["float_id"]
                    }
                ),
                Tool(
                    name="calculate_water_mass_properties",
                    description="Calculate water mass properties from T/S data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "profile_ids": {"type": "array", "items": {"type": "integer"}, "description": "Profile IDs"},
                            "property_type": {"type": "string", "enum": ["density", "potential_temperature", "mixed_layer_depth"], "description": "Property to calculate"}
                        },
                        "required": ["profile_ids", "property_type"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls from AI agents"""
            try:
                if name == "query_argo_profiles":
                    return await self._query_argo_profiles(**arguments)
                elif name == "analyze_temperature_salinity":
                    return await self._analyze_temperature_salinity(**arguments)
                elif name == "search_oceanographic_data":
                    return await self._search_oceanographic_data(**arguments)
                elif name == "get_float_trajectory":
                    return await self._get_float_trajectory(**arguments)
                elif name == "calculate_water_mass_properties":
                    return await self._calculate_water_mass_properties(**arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
            except Exception as e:
                logger.error(f"Tool call error: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    def _setup_resources(self):
        """Register MCP resources for oceanographic data access"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available oceanographic data resources"""
            return [
                Resource(
                    uri="argo://profiles/summary",
                    name="ARGO Profiles Summary",
                    description="Summary statistics of all ARGO profiles in the database",
                    mimeType="application/json"
                ),
                Resource(
                    uri="argo://floats/active",
                    name="Active ARGO Floats",
                    description="List of currently active ARGO floats",
                    mimeType="application/json"
                ),
                Resource(
                    uri="argo://data/schema",
                    name="Database Schema",
                    description="ARGO platform database schema and structure",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read oceanographic data resources"""
            try:
                if uri == "argo://profiles/summary":
                    return await self._get_profiles_summary()
                elif uri == "argo://floats/active":
                    return await self._get_active_floats()
                elif uri == "argo://data/schema":
                    return await self._get_database_schema()
                else:
                    return f"Unknown resource: {uri}"
            except Exception as e:
                logger.error(f"Resource read error: {e}")
                return f"Error reading resource: {str(e)}"
    
    async def _query_argo_profiles(self, **kwargs) -> List[TextContent]:
        """Query ARGO profiles with specified criteria"""
        try:
            # Build query based on parameters
            query = "SELECT * FROM argo_profiles WHERE 1=1"
            params = []
            
            if 'lat_min' in kwargs and 'lat_max' in kwargs:
                query += " AND latitude BETWEEN %s AND %s"
                params.extend([kwargs['lat_min'], kwargs['lat_max']])
            
            if 'lon_min' in kwargs and 'lon_max' in kwargs:
                query += " AND longitude BETWEEN %s AND %s"
                params.extend([kwargs['lon_min'], kwargs['lon_max']])
            
            if 'date_start' in kwargs:
                query += " AND date >= %s"
                params.append(kwargs['date_start'])
            
            if 'date_end' in kwargs:
                query += " AND date <= %s"
                params.append(kwargs['date_end'])
            
            limit = kwargs.get('limit', 100)
            query += f" LIMIT {limit}"
            
            # Execute query
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    results = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
            
            # Format results
            profiles = []
            for row in results:
                profile = dict(zip(columns, row))
                profiles.append(profile)
            
            return [TextContent(
                type="text",
                text=f"Found {len(profiles)} ARGO profiles matching criteria:\\n{json.dumps(profiles, indent=2, default=str)}"
            )]
        
        except Exception as e:
            return [TextContent(type="text", text=f"Error querying profiles: {str(e)}")]
    
    async def _analyze_temperature_salinity(self, profile_ids: List[int], analysis_type: str) -> List[TextContent]:
        """Analyze temperature and salinity data"""
        try:
            query = """
            SELECT am.depth, am.temperature, am.salinity, ap.profile_id
            FROM argo_measurements am
            JOIN argo_profiles ap ON am.profile_id = ap.profile_id
            WHERE ap.profile_id = ANY(%s)
            ORDER BY ap.profile_id, am.depth
            """
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (profile_ids,))
                    results = cursor.fetchall()
            
            if not results:
                return [TextContent(type="text", text="No data found for specified profiles")]
            
            # Process data based on analysis type
            if analysis_type == "depth_profile":
                analysis = self._analyze_depth_profiles(results)
            elif analysis_type == "ts_diagram":
                analysis = self._create_ts_diagram_data(results)
            elif analysis_type == "statistics":
                analysis = self._calculate_statistics(results)
            else:
                return [TextContent(type="text", text=f"Unknown analysis type: {analysis_type}")]
            
            return [TextContent(type="text", text=json.dumps(analysis, indent=2, default=str))]
        
        except Exception as e:
            return [TextContent(type="text", text=f"Error analyzing data: {str(e)}")]
    
    async def _search_oceanographic_data(self, query: str, top_k: int = 5) -> List[TextContent]:
        """Semantic search through oceanographic data"""
        try:
            # Initialize RAG system if not already done
            if not self.rag_system:
                self.rag_system = GroqRAGSystem()
            
            # Perform semantic search
            response = await self.rag_system.query(query)
            
            return [TextContent(type="text", text=response)]
        
        except Exception as e:
            return [TextContent(type="text", text=f"Error in semantic search: {str(e)}")]
    
    async def _get_float_trajectory(self, float_id: str, cycle_range: Optional[List[int]] = None) -> List[TextContent]:
        """Get trajectory data for ARGO float"""
        try:
            query = """
            SELECT latitude, longitude, date, cycle_number
            FROM argo_profiles
            WHERE float_id = %s
            """
            params = [float_id]
            
            if cycle_range:
                query += " AND cycle_number BETWEEN %s AND %s"
                params.extend(cycle_range)
            
            query += " ORDER BY cycle_number"
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    results = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
            
            trajectory = []
            for row in results:
                point = dict(zip(columns, row))
                trajectory.append(point)
            
            return [TextContent(
                type="text",
                text=f"Float {float_id} trajectory ({len(trajectory)} points):\\n{json.dumps(trajectory, indent=2, default=str)}"
            )]
        
        except Exception as e:
            return [TextContent(type="text", text=f"Error getting trajectory: {str(e)}")]
    
    async def _calculate_water_mass_properties(self, profile_ids: List[int], property_type: str) -> List[TextContent]:
        """Calculate water mass properties"""
        try:
            # This is a simplified implementation
            # In a real system, you'd use oceanographic libraries like gsw
            
            query = """
            SELECT depth, temperature, salinity, pressure
            FROM argo_measurements am
            JOIN argo_profiles ap ON am.profile_id = ap.profile_id
            WHERE ap.profile_id = ANY(%s)
            ORDER BY ap.profile_id, depth
            """
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (profile_ids,))
                    results = cursor.fetchall()
            
            if property_type == "density":
                # Simplified density calculation (in reality, use gsw.density)
                calculations = []
                for row in results:
                    depth, temp, sal, pressure = row
                    if temp and sal:
                        # Simplified density approximation
                        density = 1000 + 0.8 * sal - 0.2 * temp + 0.004 * depth
                        calculations.append({
                            "depth": depth,
                            "temperature": temp,
                            "salinity": sal,
                            "calculated_density": density
                        })
                
                return [TextContent(
                    type="text",
                    text=f"Density calculations for {len(calculations)} measurements:\\n{json.dumps(calculations, indent=2)}"
                )]
            
            # Add other property calculations as needed
            return [TextContent(type="text", text=f"Property calculation for {property_type} not yet implemented")]
        
        except Exception as e:
            return [TextContent(type="text", text=f"Error calculating properties: {str(e)}")]
    
    def _analyze_depth_profiles(self, data):
        """Analyze depth profiles"""
        # Group by profile_id and analyze
        profiles = {}
        for row in data:
            depth, temp, sal, profile_id = row
            if profile_id not in profiles:
                profiles[profile_id] = {"depths": [], "temperatures": [], "salinities": []}
            
            if depth is not None:
                profiles[profile_id]["depths"].append(depth)
            if temp is not None:
                profiles[profile_id]["temperatures"].append(temp)
            if sal is not None:
                profiles[profile_id]["salinities"].append(sal)
        
        return {"analysis_type": "depth_profile", "profiles": profiles}
    
    def _create_ts_diagram_data(self, data):
        """Create T-S diagram data"""
        ts_data = []
        for row in data:
            depth, temp, sal, profile_id = row
            if temp is not None and sal is not None:
                ts_data.append({
                    "temperature": temp,
                    "salinity": sal,
                    "depth": depth,
                    "profile_id": profile_id
                })
        
        return {"analysis_type": "ts_diagram", "data": ts_data}
    
    def _calculate_statistics(self, data):
        """Calculate basic statistics"""
        temps = [row[1] for row in data if row[1] is not None]
        sals = [row[2] for row in data if row[2] is not None]
        depths = [row[0] for row in data if row[0] is not None]
        
        stats = {
            "temperature": {
                "count": len(temps),
                "min": min(temps) if temps else None,
                "max": max(temps) if temps else None,
                "mean": sum(temps) / len(temps) if temps else None
            },
            "salinity": {
                "count": len(sals),
                "min": min(sals) if sals else None,
                "max": max(sals) if sals else None,
                "mean": sum(sals) / len(sals) if sals else None
            },
            "depth": {
                "count": len(depths),
                "min": min(depths) if depths else None,
                "max": max(depths) if depths else None,
                "mean": sum(depths) / len(depths) if depths else None
            }
        }
        
        return {"analysis_type": "statistics", "stats": stats}
    
    async def _get_profiles_summary(self) -> str:
        """Get summary of all profiles"""
        try:
            query = """
            SELECT COUNT(*) as total_profiles,
                   MIN(date) as earliest_date,
                   MAX(date) as latest_date,
                   COUNT(DISTINCT float_id) as unique_floats,
                   MIN(latitude) as min_lat,
                   MAX(latitude) as max_lat,
                   MIN(longitude) as min_lon,
                   MAX(longitude) as max_lon
            FROM argo_profiles
            """
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    result = cursor.fetchone()
                    columns = [desc[0] for desc in cursor.description]
            
            summary = dict(zip(columns, result))
            return json.dumps(summary, indent=2, default=str)
        
        except Exception as e:
            return f"Error getting summary: {str(e)}"
    
    async def _get_active_floats(self) -> str:
        """Get list of active floats"""
        try:
            query = """
            SELECT float_id, COUNT(*) as profile_count,
                   MAX(date) as last_profile_date
            FROM argo_profiles
            GROUP BY float_id
            ORDER BY last_profile_date DESC
            LIMIT 50
            """
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    results = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
            
            floats = []
            for row in results:
                float_data = dict(zip(columns, row))
                floats.append(float_data)
            
            return json.dumps(floats, indent=2, default=str)
        
        except Exception as e:
            return f"Error getting active floats: {str(e)}"
    
    async def _get_database_schema(self) -> str:
        """Get database schema information"""
        schema = {
            "tables": {
                "argo_profiles": {
                    "description": "Main ARGO float profile data",
                    "columns": [
                        "profile_id", "float_id", "cycle_number", "date",
                        "latitude", "longitude", "ocean", "profiler_type"
                    ]
                },
                "argo_measurements": {
                    "description": "Measurement data for each profile",
                    "columns": [
                        "measurement_id", "profile_id", "depth", "pressure",
                        "temperature", "salinity", "quality_flags"
                    ]
                },
                "argo_metadata": {
                    "description": "Metadata for ARGO floats and profiles",
                    "columns": [
                        "metadata_id", "profile_id", "parameter_name",
                        "parameter_value", "units"
                    ]
                }
            }
        }
        
        return json.dumps(schema, indent=2)
    
    async def run(self):
        """Run the MCP server"""
        logger.info("Starting ARGO MCP Server...")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="argo-oceanographic",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities=None,
                    ),
                ),
            )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    server = ArgoMCPServer()
    asyncio.run(server.run())