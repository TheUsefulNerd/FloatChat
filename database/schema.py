"""
Database schema definitions for ARGO oceanographic data
"""

from typing import Dict, Any, List

# Table schemas for ARGO data
ARGO_PROFILES_SCHEMA = {
    'table_name': 'argo_profiles',
    'columns': {
        'id': 'SERIAL PRIMARY KEY',
        'float_id': 'VARCHAR(50) NOT NULL',
        'cycle_number': 'INTEGER',
        'latitude': 'DECIMAL(10, 6)',
        'longitude': 'DECIMAL(10, 6)',
        'measurement_date': 'TIMESTAMP',
        'platform_number': 'VARCHAR(50)',
        'data_center': 'VARCHAR(10)',
        'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
        'file_hash': 'VARCHAR(64) UNIQUE'
    },
    'indexes': [
        'CREATE INDEX IF NOT EXISTS idx_argo_profiles_float_id ON argo_profiles(float_id)',
        'CREATE INDEX IF NOT EXISTS idx_argo_profiles_date ON argo_profiles(measurement_date)',
        'CREATE INDEX IF NOT EXISTS idx_argo_profiles_location ON argo_profiles(latitude, longitude)'
    ]
}

ARGO_MEASUREMENTS_SCHEMA = {
    'table_name': 'argo_measurements',
    'columns': {
        'id': 'SERIAL PRIMARY KEY',
        'profile_id': 'INTEGER REFERENCES argo_profiles(id) ON DELETE CASCADE',
        'pressure': 'DECIMAL(10, 4)',
        'temperature': 'DECIMAL(10, 4)',
        'salinity': 'DECIMAL(10, 4)',
        'depth': 'DECIMAL(10, 4)',
        'oxygen': 'DECIMAL(10, 4)',
        'nitrate': 'DECIMAL(10, 4)',
        'ph': 'DECIMAL(10, 4)',
        'chlorophyll': 'DECIMAL(10, 4)',
        'quality_flag': 'INTEGER DEFAULT 1'
    },
    'indexes': [
        'CREATE INDEX IF NOT EXISTS idx_argo_measurements_profile ON argo_measurements(profile_id)',
        'CREATE INDEX IF NOT EXISTS idx_argo_measurements_depth ON argo_measurements(depth)'
    ]
}

ARGO_METADATA_SCHEMA = {
    'table_name': 'argo_metadata',
    'columns': {
        'id': 'SERIAL PRIMARY KEY',
        'profile_id': 'INTEGER REFERENCES argo_profiles(id) ON DELETE CASCADE',
        'parameter_name': 'VARCHAR(100)',
        'parameter_value': 'TEXT',
        'parameter_units': 'VARCHAR(50)'
    },
    'indexes': [
        'CREATE INDEX IF NOT EXISTS idx_argo_metadata_profile ON argo_metadata(profile_id)',
        'CREATE INDEX IF NOT EXISTS idx_argo_metadata_parameter ON argo_metadata(parameter_name)'
    ]
}

# Standard ARGO parameters mapping
ARGO_PARAMETER_MAPPING = {
    'TEMP': {
        'name': 'temperature',
        'units': 'degree_Celsius',
        'long_name': 'Sea water temperature'
    },
    'PSAL': {
        'name': 'salinity',
        'units': 'psu',
        'long_name': 'Practical salinity'
    },
    'PRES': {
        'name': 'pressure',
        'units': 'decibar',
        'long_name': 'Sea water pressure'
    },
    'DOXY': {
        'name': 'oxygen',
        'units': 'micromole/kg',
        'long_name': 'Dissolved oxygen'
    },
    'NITRATE': {
        'name': 'nitrate',
        'units': 'micromole/kg',
        'long_name': 'Nitrate'
    },
    'PH_IN_SITU_TOTAL': {
        'name': 'ph',
        'units': '1',
        'long_name': 'pH'
    },
    'CHLA': {
        'name': 'chlorophyll',
        'units': 'mg/m3',
        'long_name': 'Chlorophyll-A'
    }
}

# Quality control flags
QUALITY_FLAGS = {
    1: 'Good data',
    2: 'Probably good data',
    3: 'Bad data that are potentially correctable',
    4: 'Bad data',
    5: 'Value changed',
    6: 'Not used',
    7: 'Not used',
    8: 'Estimated value',
    9: 'Missing value'
}

def get_create_table_sql(schema: Dict[str, Any]) -> str:
    """Generate CREATE TABLE SQL from schema definition"""
    table_name = schema['table_name']
    columns = schema['columns']
    
    column_defs = []
    for col_name, col_type in columns.items():
        column_defs.append(f"{col_name} {col_type}")
    
    sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        {', '.join(column_defs)}
    );
    """
    
    return sql

def get_all_schemas() -> List[Dict[str, Any]]:
    """Get all table schemas"""
    return [
        ARGO_PROFILES_SCHEMA,
        ARGO_MEASUREMENTS_SCHEMA,
        ARGO_METADATA_SCHEMA
    ]

def validate_measurement_data(measurement: Dict[str, Any]) -> bool:
    """Validate measurement data against schema"""
    required_fields = ['pressure', 'temperature', 'salinity']
    
    for field in required_fields:
        if field not in measurement:
            return False
    
    # Check for reasonable value ranges
    if measurement.get('temperature') is not None:
        temp = float(measurement['temperature'])
        if temp < -5 or temp > 50:  # Reasonable ocean temperature range
            return False
    
    if measurement.get('salinity') is not None:
        sal = float(measurement['salinity'])
        if sal < 0 or sal > 50:  # Reasonable salinity range
            return False
    
    if measurement.get('pressure') is not None:
        pres = float(measurement['pressure'])
        if pres < 0 or pres > 10000:  # Reasonable pressure range (0-10000 dbar)
            return False
    
    return True

def standardize_parameter_name(param_name: str) -> str:
    """Standardize ARGO parameter names"""
    return ARGO_PARAMETER_MAPPING.get(param_name, {}).get('name', param_name.lower())

def get_parameter_units(param_name: str) -> str:
    """Get standard units for ARGO parameters"""
    return ARGO_PARAMETER_MAPPING.get(param_name, {}).get('units', '')

def get_parameter_long_name(param_name: str) -> str:
    """Get long name for ARGO parameters"""
    return ARGO_PARAMETER_MAPPING.get(param_name, {}).get('long_name', param_name)
