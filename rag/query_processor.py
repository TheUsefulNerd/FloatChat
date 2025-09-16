import re
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Process natural language queries and extract structured information for database queries
    """
    
    def __init__(self):
        self.parameter_synonyms = {
            'temperature': ['temp', 'temperature', 'thermal', 'warming', 'cooling', 'heat'],
            'salinity': ['sal', 'salinity', 'salt', 'saltiness', 'psu'],
            'pressure': ['pressure', 'press', 'depth pressure'],
            'depth': ['depth', 'deep', 'shallow', 'meters', 'level'],
            'oxygen': ['oxygen', 'o2', 'dissolved oxygen', 'doxy'],
            'nitrate': ['nitrate', 'nitrogen', 'nutrient', 'no3'],
            'ph': ['ph', 'acidity', 'alkalinity', 'acid'],
            'chlorophyll': ['chlorophyll', 'chla', 'phytoplankton', 'algae', 'primary production']
        }
        
        self.location_patterns = {
            'latitude': [
                r'(\d+(?:\.\d+)?)\s*°?\s*[nN]',
                r'(\d+(?:\.\d+)?)\s*[nN]',
                r'north\s+(\d+(?:\.\d+)?)',
                r'latitude\s+(\d+(?:\.\d+)?)'
            ],
            'longitude': [
                r'(\d+(?:\.\d+)?)\s*°?\s*[eE]',
                r'(\d+(?:\.\d+)?)\s*[eE]',
                r'east\s+(\d+(?:\.\d+)?)',
                r'longitude\s+(\d+(?:\.\d+)?)'
            ]
        }
        
        self.date_patterns = [
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # YYYY-MM-DD or YYYY/MM/DD
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',  # MM-DD-YYYY or MM/DD/YYYY
            r'([jJ]anuary|[fF]ebruary|[mM]arch|[aA]pril|[mM]ay|[jJ]une|[jJ]uly|[aA]ugust|[sS]eptember|[oO]ctober|[nN]ovember|[dD]ecember)\s+(\d{4})',
            r'(\d{4})',  # Just year
            r'([jJ]an|[fF]eb|[mM]ar|[aA]pr|[mM]ay|[jJ]un|[jJ]ul|[aA]ug|[sS]ep|[oO]ct|[nN]ov|[dD]ec)\s+(\d{4})'
        ]
        
        self.region_names = {
            'arabian sea': {'lat_range': (10, 25), 'lon_range': (50, 80)},
            'indian ocean': {'lat_range': (-40, 25), 'lon_range': (20, 120)},
            'pacific ocean': {'lat_range': (-60, 60), 'lon_range': (120, -60)},
            'atlantic ocean': {'lat_range': (-60, 70), 'lon_range': (-80, 20)},
            'mediterranean sea': {'lat_range': (30, 46), 'lon_range': (-6, 36)},
            'red sea': {'lat_range': (12, 30), 'lon_range': (32, 43)},
            'equator': {'lat_range': (-5, 5), 'lon_range': (-180, 180)},
            'tropics': {'lat_range': (-23.5, 23.5), 'lon_range': (-180, 180)},
            'polar': {'lat_range': (66.5, 90), 'lon_range': (-180, 180)},
        }
        
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a natural language query and extract structured information
        """
        query_lower = query.lower()
        
        analysis = {
            'original_query': query,
            'query_type': self._determine_query_type(query_lower),
            'parameters': self._extract_parameters(query_lower),
            'location': self._extract_location(query),
            'time_range': self._extract_time_range(query_lower),
            'float_ids': self._extract_float_ids(query),
            'depth_range': self._extract_depth_range(query_lower),
            'comparison': self._detect_comparison(query_lower),
            'aggregation': self._detect_aggregation(query_lower),
            'filters': self._extract_filters(query_lower)
        }
        
        logger.info(f"Query analysis: {analysis['query_type']} query for {analysis['parameters']}")
        return analysis
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query being asked"""
        
        if any(word in query for word in ['show', 'display', 'plot', 'visualize', 'graph']):
            return 'visualization'
        elif any(word in query for word in ['compare', 'difference', 'versus', 'vs']):
            return 'comparison'
        elif any(word in query for word in ['near', 'nearby', 'closest', 'around', 'within']):
            return 'location_search'
        elif any(word in query for word in ['count', 'how many', 'number of']):
            return 'count'
        elif any(word in query for word in ['average', 'mean', 'median', 'maximum', 'minimum', 'statistics']):
            return 'statistics'
        elif any(word in query for word in ['trend', 'change', 'over time', 'temporal']):
            return 'temporal_analysis'
        elif any(word in query for word in ['profile', 'profiles', 'depth']):
            return 'profile_analysis'
        else:
            return 'general_search'
    
    def _extract_parameters(self, query: str) -> List[str]:
        """Extract oceanographic parameters mentioned in the query"""
        found_parameters = []
        
        for parameter, synonyms in self.parameter_synonyms.items():
            if any(synonym in query for synonym in synonyms):
                found_parameters.append(parameter)
        
        # If no specific parameters found, assume general query
        if not found_parameters:
            found_parameters = ['temperature', 'salinity']  # Default parameters
        
        return found_parameters
    
    def _extract_location(self, query: str) -> Dict[str, Any]:
        """Extract location information from the query"""
        location_info = {}
        
        # Check for specific coordinates
        for coord_type, patterns in self.location_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    location_info[coord_type] = float(match.group(1))
        
        # Check for named regions
        for region_name, bounds in self.region_names.items():
            if region_name in query.lower():
                location_info['region'] = region_name
                location_info['lat_range'] = bounds['lat_range']
                location_info['lon_range'] = bounds['lon_range']
                break
        
        # Extract radius for proximity searches
        radius_match = re.search(r'within\s+(\d+)\s*(km|kilometers|miles)', query, re.IGNORECASE)
        if radius_match:
            radius = float(radius_match.group(1))
            if 'miles' in radius_match.group(2).lower():
                radius *= 1.60934  # Convert miles to km
            location_info['radius_km'] = radius
        
        return location_info
    
    def _extract_time_range(self, query: str) -> Dict[str, Any]:
        """Extract time range information from the query"""
        time_info = {}
        
        # Extract specific dates
        dates_found = []
        for pattern in self.date_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if isinstance(match, tuple):
                    dates_found.extend([m for m in match if m])
                else:
                    dates_found.append(match)
        
        # Parse dates
        parsed_dates = []
        for date_str in dates_found:
            try:
                # Try different date formats
                for fmt in ['%Y-%m-%d', '%m-%d-%Y', '%Y/%m/%d', '%m/%d/%Y', '%Y']:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        parsed_dates.append(parsed_date)
                        break
                    except ValueError:
                        continue
            except:
                continue
        
        if parsed_dates:
            time_info['start_date'] = min(parsed_dates)
            time_info['end_date'] = max(parsed_dates)
        
        # Handle relative time expressions
        if 'last' in query:
            if 'month' in query:
                time_info['end_date'] = datetime.now()
                time_info['start_date'] = datetime.now() - timedelta(days=30)
            elif 'year' in query:
                time_info['end_date'] = datetime.now()
                time_info['start_date'] = datetime.now() - timedelta(days=365)
            elif 'week' in query:
                time_info['end_date'] = datetime.now()
                time_info['start_date'] = datetime.now() - timedelta(days=7)
        
        # Extract specific months/years
        month_match = re.search(r'([jJ]anuary|[fF]ebruary|[mM]arch|[aA]pril|[mM]ay|[jJ]une|[jJ]uly|[aA]ugust|[sS]eptember|[oO]ctober|[nN]ovember|[dD]ecember)\s+(\d{4})', query)
        if month_match:
            month_name = month_match.group(1).lower().capitalize()
            year = int(month_match.group(2))
            month_num = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12
            }.get(month_name, 1)
            
            time_info['start_date'] = datetime(year, month_num, 1)
            if month_num == 12:
                time_info['end_date'] = datetime(year + 1, 1, 1)
            else:
                time_info['end_date'] = datetime(year, month_num + 1, 1)
        
        return time_info
    
    def _extract_float_ids(self, query: str) -> List[str]:
        """Extract float IDs from the query"""
        float_ids = []
        
        # Pattern for float IDs (usually numeric)
        float_patterns = [
            r'float\s+(\d+)',
            r'platform\s+(\d+)',
            r'(\d{7,10})'  # 7-10 digit numbers (typical float IDs)
        ]
        
        for pattern in float_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            float_ids.extend(matches)
        
        return list(set(float_ids))  # Remove duplicates
    
    def _extract_depth_range(self, query: str) -> Dict[str, float]:
        """Extract depth range information"""
        depth_info = {}
        
        # Patterns for depth ranges
        depth_patterns = [
            r'(\d+)\s*-\s*(\d+)\s*m',  # 100-500m
            r'(\d+)\s*to\s*(\d+)\s*meter',  # 100 to 500 meters
            r'depth\s+(\d+)\s*m',  # depth 100m
            r'(\d+)\s*meter',  # 100 meters
            r'surface',  # surface
            r'deep',  # deep
        ]
        
        for pattern in depth_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if pattern == r'surface':
                    depth_info['max_depth'] = 100
                elif pattern == r'deep':
                    depth_info['min_depth'] = 1000
                elif len(match.groups()) == 2:
                    depth_info['min_depth'] = float(match.group(1))
                    depth_info['max_depth'] = float(match.group(2))
                elif len(match.groups()) == 1:
                    depth_info['target_depth'] = float(match.group(1))
                break
        
        return depth_info
    
    def _detect_comparison(self, query: str) -> Dict[str, Any]:
        """Detect if the query involves comparison"""
        comparison_info = {}
        
        if any(word in query for word in ['compare', 'comparison', 'versus', 'vs', 'difference']):
            comparison_info['is_comparison'] = True
            
            # Try to extract what's being compared
            if 'between' in query:
                comparison_info['type'] = 'between_regions_or_times'
            elif any(word in query for word in ['before', 'after']):
                comparison_info['type'] = 'temporal'
            else:
                comparison_info['type'] = 'general'
        else:
            comparison_info['is_comparison'] = False
        
        return comparison_info
    
    def _detect_aggregation(self, query: str) -> Dict[str, Any]:
        """Detect aggregation requirements"""
        aggregation_info = {}
        
        aggregation_words = {
            'average': 'mean',
            'mean': 'mean',
            'median': 'median',
            'maximum': 'max',
            'minimum': 'min',
            'sum': 'sum',
            'count': 'count',
            'total': 'sum'
        }
        
        for word, agg_type in aggregation_words.items():
            if word in query:
                aggregation_info['type'] = agg_type
                aggregation_info['required'] = True
                break
        
        if not aggregation_info:
            aggregation_info['required'] = False
        
        return aggregation_info
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract various filter conditions"""
        filters = {}
        
        # Quality filters
        if any(word in query for word in ['good', 'quality', 'valid']):
            filters['quality_flag'] = {'operator': '<=', 'value': 2}
        elif any(word in query for word in ['bad', 'invalid', 'poor']):
            filters['quality_flag'] = {'operator': '>', 'value': 2}
        
        # Value range filters
        value_patterns = [
            r'(\w+)\s+greater\s+than\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+>\s*(\d+(?:\.\d+)?)',
            r'(\w+)\s+less\s+than\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+<\s*(\d+(?:\.\d+)?)',
            r'(\w+)\s+between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in value_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    param, value = match
                    if 'greater' in pattern or '>' in pattern:
                        filters[param] = {'operator': '>', 'value': float(value)}
                    elif 'less' in pattern or '<' in pattern:
                        filters[param] = {'operator': '<', 'value': float(value)}
                elif len(match) == 3:
                    param, min_val, max_val = match
                    filters[param] = {'operator': 'between', 'min': float(min_val), 'max': float(max_val)}
        
        return filters
    
    def build_database_filters(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Convert query analysis to database filter parameters"""
        db_filters = {}
        
        # Location filters
        location = analysis.get('location', {})
        if 'latitude' in location:
            db_filters['lat'] = location['latitude']
        if 'longitude' in location:
            db_filters['lon'] = location['longitude']
        if 'lat_range' in location:
            db_filters['min_lat'], db_filters['max_lat'] = location['lat_range']
        if 'lon_range' in location:
            db_filters['min_lon'], db_filters['max_lon'] = location['lon_range']
        if 'radius_km' in location:
            db_filters['radius_km'] = location['radius_km']
        
        # Time filters
        time_range = analysis.get('time_range', {})
        if 'start_date' in time_range:
            db_filters['start_date'] = time_range['start_date']
        if 'end_date' in time_range:
            db_filters['end_date'] = time_range['end_date']
        
        # Float ID filters
        float_ids = analysis.get('float_ids', [])
        if float_ids:
            db_filters['float_ids'] = float_ids
        
        # Depth filters
        depth_range = analysis.get('depth_range', {})
        db_filters.update(depth_range)
        
        # Parameter value filters
        filters = analysis.get('filters', {})
        for param, filter_condition in filters.items():
            if param in ['temperature', 'salinity', 'pressure', 'oxygen', 'nitrate', 'ph', 'chlorophyll']:
                db_filters[f'{param}_filter'] = filter_condition
        
        return db_filters
    
    def suggest_query_improvements(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest improvements to make the query more specific"""
        suggestions = []
        
        if not analysis.get('location'):
            suggestions.append("Consider specifying a geographic region or coordinates")
        
        if not analysis.get('time_range'):
            suggestions.append("Consider adding a time range (e.g., 'in 2023' or 'last 6 months')")
        
        if len(analysis.get('parameters', [])) > 3:
            suggestions.append("Consider focusing on 1-2 specific parameters for clearer results")
        
        if analysis.get('query_type') == 'general_search':
            suggestions.append("Try using specific action words like 'show', 'compare', or 'analyze'")
        
        return suggestions
