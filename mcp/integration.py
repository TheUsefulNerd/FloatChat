"""
MCP Integration Module for ARGO Platform
Integrates MCP tools with the Groq RAG system for enhanced AI capabilities
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from .client import ArgoMCPClient

logger = logging.getLogger("mcp_integration")

class MCPEnhancedRAG:
    """Enhanced RAG system with MCP tools integration"""
    
    def __init__(self, groq_rag_system=None):
        self.mcp_client = ArgoMCPClient()
        self.groq_rag = groq_rag_system
        self.tools_connected = False
    
    async def initialize(self):
        """Initialize MCP connection and tools"""
        try:
            success = await self.mcp_client.connect()
            if success:
                self.tools_connected = True
                logger.info("MCP tools connected successfully")
                return True
            else:
                logger.warning("Failed to connect MCP tools")
                return False
        except Exception as e:
            logger.error(f"Error initializing MCP: {e}")
            return False
    
    async def process_query(self, user_query: str) -> str:
        """Process user query with MCP tools and RAG"""
        try:
            # Initialize if not already done
            if not self.tools_connected:
                await self.initialize()
            
            # Analyze query to determine if MCP tools are needed
            tool_analysis = self._analyze_query_for_tools(user_query)
            
            if tool_analysis['needs_tools']:
                # Use MCP tools for specific data queries
                tool_result = await self._execute_mcp_tools(tool_analysis['suggested_tools'], user_query)
                
                # Combine tool results with RAG for comprehensive response
                if self.groq_rag:
                    context_enhanced_query = f"Based on this oceanographic data: {tool_result}\\n\\nUser question: {user_query}"
                    rag_response = await self.groq_rag.query(context_enhanced_query)
                    return self._combine_responses(tool_result, rag_response)
                else:
                    return self._format_tool_response(tool_result, user_query)
            else:
                # Use regular RAG for general questions
                if self.groq_rag:
                    return await self.groq_rag.query(user_query)
                else:
                    return "RAG system not available. Please rephrase your query."
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I encountered an error processing your query. Please try again or rephrase your question."
    
    def _analyze_query_for_tools(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine which MCP tools might be useful"""
        query_lower = query.lower()
        
        # Keywords that suggest specific tool usage
        tool_keywords = {
            'query_argo_profiles': [
                'profiles', 'location', 'latitude', 'longitude', 'date range',
                'find profiles', 'search profiles', 'profiles in', 'data from'
            ],
            'analyze_temperature_salinity': [
                'temperature', 'salinity', 't-s', 'analyze', 'depth profile',
                'water properties', 'temperature profile', 'salinity profile'
            ],
            'get_float_trajectory': [
                'trajectory', 'path', 'movement', 'float', 'track', 'route',
                'where did', 'float movement', 'float path'
            ],
            'calculate_water_mass_properties': [
                'density', 'water mass', 'properties', 'calculate', 'derive',
                'water density', 'potential temperature', 'mixed layer'
            ],
            'search_oceanographic_data': [
                'search', 'find', 'look for', 'semantic search', 'similar'
            ]
        }
        
        suggested_tools = []
        for tool_name, keywords in tool_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                suggested_tools.append(tool_name)
        
        # Also check for specific data requests
        needs_tools = bool(suggested_tools) or any([
            'show me' in query_lower,
            'get data' in query_lower,
            'find data' in query_lower,
            'analyze data' in query_lower,
            'profile' in query_lower,
            'measurement' in query_lower
        ])
        
        return {
            'needs_tools': needs_tools,
            'suggested_tools': suggested_tools,
            'confidence': len(suggested_tools) / len(tool_keywords) if suggested_tools else 0.1
        }
    
    async def _execute_mcp_tools(self, suggested_tools: List[str], query: str) -> str:
        """Execute appropriate MCP tools based on query analysis"""
        results = []
        
        try:
            # Extract parameters from query for different tools
            query_params = self._extract_query_parameters(query)
            
            for tool in suggested_tools:
                if tool == 'query_argo_profiles':
                    result = await self.mcp_client.call_tool('query_argo_profiles', query_params)
                    results.append(f"Profile Query Results:\\n{result}")
                
                elif tool == 'search_oceanographic_data':
                    search_params = {'query': query, 'top_k': 10}
                    result = await self.mcp_client.call_tool('search_oceanographic_data', search_params)
                    results.append(f"Semantic Search Results:\\n{result}")
                
                elif tool == 'get_float_trajectory' and 'float_id' in query_params:
                    result = await self.mcp_client.call_tool('get_float_trajectory', query_params)
                    results.append(f"Float Trajectory:\\n{result}")
            
            # If no specific tools were suggested, try a general search
            if not results:
                search_params = {'query': query, 'top_k': 5}
                result = await self.mcp_client.call_tool('search_oceanographic_data', search_params)
                results.append(f"General Search Results:\\n{result}")
            
            return "\\n\\n".join(results)
        
        except Exception as e:
            return f"Error executing MCP tools: {str(e)}"
    
    def _extract_query_parameters(self, query: str) -> Dict[str, Any]:
        """Extract parameters from natural language query"""
        import re
        
        params = {}
        query_lower = query.lower()
        
        # Extract coordinates
        lat_pattern = r'(?:latitude|lat)\\s*(?:between|from)?\\s*(-?\\d+(?:\\.\\d+)?)(?:\\s*(?:to|and)\\s*(-?\\d+(?:\\.\\d+)?))?'
        lon_pattern = r'(?:longitude|lon)\\s*(?:between|from)?\\s*(-?\\d+(?:\\.\\d+)?)(?:\\s*(?:to|and)\\s*(-?\\d+(?:\\.\\d+)?))?'
        
        lat_match = re.search(lat_pattern, query_lower)
        if lat_match:
            params['lat_min'] = float(lat_match.group(1))
            if lat_match.group(2):
                params['lat_max'] = float(lat_match.group(2))
            else:
                params['lat_max'] = params['lat_min'] + 5  # Default range
        
        lon_match = re.search(lon_pattern, query_lower)
        if lon_match:
            params['lon_min'] = float(lon_match.group(1))
            if lon_match.group(2):
                params['lon_max'] = float(lon_match.group(2))
            else:
                params['lon_max'] = params['lon_min'] + 5  # Default range
        
        # Extract dates
        date_pattern = r'(\\d{4}-\\d{2}-\\d{2})'
        dates = re.findall(date_pattern, query)
        if len(dates) >= 1:
            params['date_start'] = dates[0]
        if len(dates) >= 2:
            params['date_end'] = dates[1]
        
        # Extract float ID
        float_pattern = r'float\\s+(\\d+)'
        float_match = re.search(float_pattern, query_lower)
        if float_match:
            params['float_id'] = float_match.group(1)
        
        # Extract profile IDs
        profile_pattern = r'profile\\s+(\\d+(?:,\\s*\\d+)*)'
        profile_match = re.search(profile_pattern, query_lower)
        if profile_match:
            profile_ids = [int(x.strip()) for x in profile_match.group(1).split(',')]
            params['profile_ids'] = profile_ids
        
        # Set default limit
        if 'limit' not in params:
            params['limit'] = 50
        
        return params
    
    def _combine_responses(self, tool_result: str, rag_response: str) -> str:
        """Combine MCP tool results with RAG response"""
        return f"""Based on the oceanographic data analysis:

{rag_response}

**Detailed Data:**
{tool_result}
"""
    
    def _format_tool_response(self, tool_result: str, original_query: str) -> str:
        """Format tool response when RAG is not available"""
        return f"""Here's the oceanographic data analysis for your query: "{original_query}"

{tool_result}

This data is retrieved directly from the ARGO float database and processed using specialized oceanographic tools.
"""
    
    async def get_available_tools(self) -> List[Dict]:
        """Get list of available MCP tools"""
        if not self.tools_connected:
            await self.initialize()
        
        return await self.mcp_client.list_tools()
    
    async def get_available_resources(self) -> List[Dict]:
        """Get list of available MCP resources"""
        if not self.tools_connected:
            await self.initialize()
        
        return await self.mcp_client.list_resources()
    
    async def get_database_summary(self) -> str:
        """Get a summary of available data"""
        try:
            summary = await self.mcp_client.read_resource("argo://profiles/summary")
            return f"Database Summary:\\n{summary}"
        except Exception as e:
            return f"Error getting database summary: {str(e)}"
    
    async def call_specific_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a specific MCP tool with given arguments"""
        if not self.tools_connected:
            await self.initialize()
        
        return await self.mcp_client.call_tool(tool_name, arguments)
    
    async def disconnect(self):
        """Disconnect from MCP services"""
        await self.mcp_client.disconnect()
        self.tools_connected = False
        logger.info("MCP integration disconnected")

class MCPToolHelper:
    """Helper class for MCP tool interactions in Streamlit"""
    
    @staticmethod
    def get_tool_descriptions() -> Dict[str, str]:
        """Get human-readable descriptions of available tools"""
        return {
            "query_argo_profiles": "Search for ARGO float profiles by location, date, or other criteria",
            "analyze_temperature_salinity": "Analyze temperature and salinity data from specific profiles",
            "search_oceanographic_data": "Perform semantic search through oceanographic data",
            "get_float_trajectory": "Get the movement trajectory of a specific ARGO float",
            "calculate_water_mass_properties": "Calculate oceanographic properties like density and water mass characteristics"
        }
    
    @staticmethod
    def format_tool_parameters(tool_name: str) -> Dict[str, Any]:
        """Get parameter schema for a specific tool"""
        schemas = {
            "query_argo_profiles": {
                "lat_min": {"type": "number", "description": "Minimum latitude (-90 to 90)"},
                "lat_max": {"type": "number", "description": "Maximum latitude (-90 to 90)"},
                "lon_min": {"type": "number", "description": "Minimum longitude (-180 to 180)"},
                "lon_max": {"type": "number", "description": "Maximum longitude (-180 to 180)"},
                "date_start": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                "date_end": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                "limit": {"type": "integer", "description": "Maximum number of results (default: 100)"}
            },
            "analyze_temperature_salinity": {
                "profile_ids": {"type": "list", "description": "List of profile IDs to analyze"},
                "analysis_type": {"type": "select", "options": ["statistics", "depth_profile", "ts_diagram"]}
            },
            "get_float_trajectory": {
                "float_id": {"type": "string", "description": "ARGO float ID"},
                "cycle_range": {"type": "list", "description": "Optional: cycle number range [start, end]"}
            },
            "calculate_water_mass_properties": {
                "profile_ids": {"type": "list", "description": "List of profile IDs"},
                "property_type": {"type": "select", "options": ["density", "potential_temperature", "mixed_layer_depth"]}
            }
        }
        return schemas.get(tool_name, {})