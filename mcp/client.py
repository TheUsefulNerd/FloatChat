"""
MCP Client for ARGO Oceanographic Platform
Connects to MCP server and provides tools for AI agents
"""
from config.settings import load_config
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mcp_argo_client")


class ArgoMCPClient:
    """MCP Client for connecting to ARGO oceanographic tools"""

    def __init__(self):
        self.available_tools: List[Dict] = []
        self.available_resources: List[Dict] = []

    async def connect(self):
        """Connect to the MCP server"""
        try:
            # In a real implementation, you would connect to the MCP server
            # For now, we'll simulate the connection
            logger.info("Connecting to ARGO MCP server...")

            # Simulate available tools
            self.available_tools = [{
                "name":
                "query_argo_profiles",
                "description":
                "Query ARGO float profiles by location, date range, or other criteria"
            }, {
                "name":
                "analyze_temperature_salinity",
                "description":
                "Analyze temperature and salinity data for specific profiles"
            }, {
                "name":
                "search_oceanographic_data",
                "description":
                "Semantic search through oceanographic data using natural language"
            }, {
                "name":
                "get_float_trajectory",
                "description":
                "Get trajectory data for specific ARGO floats"
            }, {
                "name":
                "calculate_water_mass_properties",
                "description":
                "Calculate water mass properties from T/S data"
            }]

            self.available_resources = [{
                "uri": "argo://profiles/summary",
                "name": "ARGO Profiles Summary"
            }, {
                "uri": "argo://floats/active",
                "name": "Active ARGO Floats"
            }, {
                "uri": "argo://data/schema",
                "name": "Database Schema"
            }]

            logger.info(
                f"Connected successfully. Available tools: {len(self.available_tools)}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False

    async def list_tools(self) -> List[Dict]:
        """List available MCP tools"""
        return self.available_tools

    async def list_resources(self) -> List[Dict]:
        """List available MCP resources"""
        return self.available_resources

    async def call_tool(self, tool_name: str, arguments: Dict[str,
                                                              Any]) -> str:
        """Call an MCP tool with given arguments"""
        try:
            # Import here to avoid circular imports
            from database.connection import DatabaseManager

            db_manager = DatabaseManager(config=load_config())

            if tool_name == "query_argo_profiles":
                return await self._query_argo_profiles(db_manager, arguments)
            elif tool_name == "analyze_temperature_salinity":
                return await self._analyze_temperature_salinity(
                    db_manager, arguments)
            elif tool_name == "search_oceanographic_data":
                return await self._search_oceanographic_data(arguments)
            elif tool_name == "get_float_trajectory":
                return await self._get_float_trajectory(db_manager, arguments)
            elif tool_name == "calculate_water_mass_properties":
                return await self._calculate_water_mass_properties(
                    db_manager, arguments)
            else:
                return f"Unknown tool: {tool_name}"

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return f"Error calling tool {tool_name}: {str(e)}"

    async def read_resource(self, uri: str) -> str:
        """Read an MCP resource"""
        try:
            from database.connection import DatabaseManager
            db_manager = DatabaseManager(config=load_config())

            if uri == "argo://profiles/summary":
                return await self._get_profiles_summary(db_manager)
            elif uri == "argo://floats/active":
                return await self._get_active_floats(db_manager)
            elif uri == "argo://data/schema":
                return await self._get_database_schema()
            else:
                return f"Unknown resource: {uri}"

        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            return f"Error reading resource {uri}: {str(e)}"

    async def _query_argo_profiles(self, db_manager,
                                   arguments: Dict[str, Any]) -> str:
        """Query ARGO profiles implementation"""
        try:
            # Build query based on parameters
            query = "SELECT profile_id, float_id, date, latitude, longitude, ocean FROM argo_profiles WHERE 1=1"
            params = []

            if 'lat_min' in arguments and 'lat_max' in arguments:
                query += " AND latitude BETWEEN %s AND %s"
                params.extend([arguments['lat_min'], arguments['lat_max']])

            if 'lon_min' in arguments and 'lon_max' in arguments:
                query += " AND longitude BETWEEN %s AND %s"
                params.extend([arguments['lon_min'], arguments['lon_max']])

            if 'date_start' in arguments:
                query += " AND date >= %s"
                params.append(arguments['date_start'])

            if 'date_end' in arguments:
                query += " AND date <= %s"
                params.append(arguments['date_end'])

            limit = arguments.get('limit', 100)
            query += f" LIMIT {limit}"

            # Execute query
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    results = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]

            # Format results
            profiles = []
            for row in results:
                profile = dict(zip(columns, row))
                profiles.append(profile)

            return f"Found {len(profiles)} ARGO profiles:\\n{json.dumps(profiles, indent=2, default=str)}"

        except Exception as e:
            return f"Error querying profiles: {str(e)}"

    async def _analyze_temperature_salinity(self, db_manager,
                                            arguments: Dict[str, Any]) -> str:
        """Analyze temperature and salinity data"""
        try:
            profile_ids = arguments.get('profile_ids', [])
            analysis_type = arguments.get('analysis_type', 'statistics')

            if not profile_ids:
                return "No profile IDs provided"

            query = """
            SELECT am.depth, am.temperature, am.salinity, ap.profile_id
            FROM argo_measurements am
            JOIN argo_profiles ap ON am.profile_id = ap.profile_id
            WHERE ap.profile_id = ANY(%s)
            ORDER BY ap.profile_id, am.depth
            """

            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (profile_ids, ))
                    results = cursor.fetchall()

            if not results:
                return "No measurement data found for specified profiles"

            # Process data based on analysis type
            if analysis_type == "statistics":
                return self._calculate_basic_statistics(results)
            elif analysis_type == "depth_profile":
                return self._format_depth_profiles(results)
            else:
                return f"Analysis type '{analysis_type}' not implemented"

        except Exception as e:
            return f"Error analyzing data: {str(e)}"

    async def _search_oceanographic_data(self, arguments: Dict[str,
                                                               Any]) -> str:
        """Semantic search through oceanographic data"""
        try:
            query = arguments.get('query', '')
            top_k = arguments.get('top_k', 5)

            if not query:
                return "No search query provided"

            # Initialize RAG system for semantic search
            try:
                from rag.groq_rag import GroqRAGSystem
                rag_system = GroqRAGSystem()
                response = await rag_system.query(query)
                return response
            except Exception as rag_error:
                # Fallback to simple database search
                from database.connection import DatabaseManager
                db_manager = DatabaseManager(config=load_config())

                search_query = """
                SELECT profile_id, float_id, date, latitude, longitude, ocean
                FROM argo_profiles
                WHERE ocean ILIKE %s OR CAST(float_id AS TEXT) ILIKE %s
                LIMIT %s
                """

                search_term = f"%{query}%"
                with db_manager.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(search_query,
                                       (search_term, search_term, top_k))
                        results = cursor.fetchall()
                        columns = [desc[0] for desc in cursor.description]

                profiles = []
                for row in results:
                    profile = dict(zip(columns, row))
                    profiles.append(profile)

                return f"Search results for '{query}':\\n{json.dumps(profiles, indent=2, default=str)}"

        except Exception as e:
            return f"Error in search: {str(e)}"

    async def _get_float_trajectory(self, db_manager,
                                    arguments: Dict[str, Any]) -> str:
        """Get trajectory data for ARGO float"""
        try:
            float_id = arguments.get('float_id')
            cycle_range = arguments.get('cycle_range')

            if not float_id:
                return "No float ID provided"

            query = """
            SELECT latitude, longitude, date, cycle_number
            FROM argo_profiles
            WHERE float_id = %s
            """
            params = [float_id]

            if cycle_range and len(cycle_range) == 2:
                query += " AND cycle_number BETWEEN %s AND %s"
                params.extend(cycle_range)

            query += " ORDER BY cycle_number"

            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    results = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]

            trajectory = []
            for row in results:
                point = dict(zip(columns, row))
                trajectory.append(point)

            return f"Float {float_id} trajectory ({len(trajectory)} points):\\n{json.dumps(trajectory, indent=2, default=str)}"

        except Exception as e:
            return f"Error getting trajectory: {str(e)}"

    async def _calculate_water_mass_properties(
            self, db_manager, arguments: Dict[str, Any]) -> str:
        """Calculate water mass properties"""
        try:
            profile_ids = arguments.get('profile_ids', [])
            property_type = arguments.get('property_type', 'density')

            if not profile_ids:
                return "No profile IDs provided"

            # Get measurement data
            query = """
            SELECT depth, temperature, salinity, pressure
            FROM argo_measurements am
            JOIN argo_profiles ap ON am.profile_id = ap.profile_id
            WHERE ap.profile_id = ANY(%s)
            ORDER BY ap.profile_id, depth
            """

            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (profile_ids, ))
                    results = cursor.fetchall()

            if not results:
                return "No measurement data found"

            if property_type == "density":
                calculations = []
                for row in results:
                    depth, temp, sal, pressure = row
                    if temp and sal:
                        # Simplified density calculation
                        density = 1000 + 0.8 * sal - 0.2 * temp + 0.004 * depth
                        calculations.append({
                            "depth":
                            depth,
                            "temperature":
                            temp,
                            "salinity":
                            sal,
                            "calculated_density":
                            round(density, 3)
                        })

                return f"Density calculations for {len(calculations)} measurements:\\n{json.dumps(calculations, indent=2)}"

            return f"Property calculation for {property_type} not yet implemented"

        except Exception as e:
            return f"Error calculating properties: {str(e)}"

    def _calculate_basic_statistics(self, data) -> str:
        """Calculate basic statistics for T/S data"""
        temps = [row[1] for row in data if row[1] is not None]
        sals = [row[2] for row in data if row[2] is not None]
        depths = [row[0] for row in data if row[0] is not None]

        stats = {
            "temperature": {
                "count": len(temps),
                "min": min(temps) if temps else None,
                "max": max(temps) if temps else None,
                "mean": round(sum(temps) / len(temps), 3) if temps else None
            },
            "salinity": {
                "count": len(sals),
                "min": min(sals) if sals else None,
                "max": max(sals) if sals else None,
                "mean": round(sum(sals) / len(sals), 3) if sals else None
            },
            "depth": {
                "count": len(depths),
                "min": min(depths) if depths else None,
                "max": max(depths) if depths else None,
                "mean": round(sum(depths) / len(depths), 3) if depths else None
            }
        }

        return f"Statistical Analysis:\\n{json.dumps(stats, indent=2)}"

    def _format_depth_profiles(self, data) -> str:
        """Format depth profile data"""
        profiles = {}
        for row in data:
            depth, temp, sal, profile_id = row
            if profile_id not in profiles:
                profiles[profile_id] = []

            profiles[profile_id].append({
                "depth": depth,
                "temperature": temp,
                "salinity": sal
            })

        return f"Depth Profiles for {len(profiles)} profiles:\\n{json.dumps(profiles, indent=2)}"

    async def _get_profiles_summary(self, db_manager) -> str:
        """Get summary of all profiles"""
        try:
            query = """
            SELECT COUNT(*) as total_profiles,
                   MIN(date) as earliest_date,
                   MAX(date) as latest_date,
                   COUNT(DISTINCT float_id) as unique_floats
            FROM argo_profiles
            """

            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    result = cursor.fetchone()
                    columns = [desc[0] for desc in cursor.description]

            summary = dict(zip(columns, result))
            return json.dumps(summary, indent=2, default=str)

        except Exception as e:
            return f"Error getting summary: {str(e)}"

    async def _get_active_floats(self, db_manager) -> str:
        """Get list of active floats"""
        try:
            query = """
            SELECT float_id, COUNT(*) as profile_count,
                   MAX(date) as last_profile_date
            FROM argo_profiles
            GROUP BY float_id
            ORDER BY last_profile_date DESC
            LIMIT 20
            """

            with db_manager.get_connection() as conn:
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
                    "description":
                    "Main ARGO float profile data",
                    "columns": [
                        "profile_id", "float_id", "cycle_number", "date",
                        "latitude", "longitude", "ocean", "profiler_type"
                    ]
                },
                "argo_measurements": {
                    "description":
                    "Measurement data for each profile",
                    "columns": [
                        "measurement_id", "profile_id", "depth", "pressure",
                        "temperature", "salinity", "quality_flags"
                    ]
                },
                "argo_metadata": {
                    "description":
                    "Metadata for ARGO floats and profiles",
                    "columns": [
                        "metadata_id", "profile_id", "parameter_name",
                        "parameter_value", "units"
                    ]
                }
            }
        }

        return json.dumps(schema, indent=2)

    async def disconnect(self):
        """Disconnect from the MCP server"""
        logger.info("Disconnected from MCP server")
