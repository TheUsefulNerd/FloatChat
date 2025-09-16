import psycopg2
import psycopg2.extras
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from config.settings import get_database_connection_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages PostgreSQL database connections and operations for ARGO data
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection_string = get_database_connection_string(config)
        self.connection = None
        self._connect()
        self._initialize_schema()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(self.connection_string)
            self.connection.autocommit = True
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def _initialize_schema(self):
        """Create tables if they don't exist"""
        try:
            with self.connection.cursor() as cursor:
                # Create ARGO profiles table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS argo_profiles (
                        id SERIAL PRIMARY KEY,
                        float_id VARCHAR(50) NOT NULL,
                        cycle_number INTEGER,
                        latitude DECIMAL(10, 6),
                        longitude DECIMAL(10, 6),
                        measurement_date TIMESTAMP,
                        platform_number VARCHAR(50),
                        data_center VARCHAR(10),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        file_hash VARCHAR(64) UNIQUE
                    );
                """)
                
                # Create measurements table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS argo_measurements (
                        id SERIAL PRIMARY KEY,
                        profile_id INTEGER REFERENCES argo_profiles(id) ON DELETE CASCADE,
                        pressure DECIMAL(10, 4),
                        temperature DECIMAL(10, 4),
                        salinity DECIMAL(10, 4),
                        depth DECIMAL(10, 4),
                        oxygen DECIMAL(10, 4),
                        nitrate DECIMAL(10, 4),
                        ph DECIMAL(10, 4),
                        chlorophyll DECIMAL(10, 4),
                        quality_flag INTEGER DEFAULT 1
                    );
                """)
                
                # Create metadata table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS argo_metadata (
                        id SERIAL PRIMARY KEY,
                        profile_id INTEGER REFERENCES argo_profiles(id) ON DELETE CASCADE,
                        parameter_name VARCHAR(100),
                        parameter_value TEXT,
                        parameter_units VARCHAR(50)
                    );
                """)
                
                # Create indexes for better performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_argo_profiles_float_id ON argo_profiles(float_id);
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_argo_profiles_date ON argo_profiles(measurement_date);
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_argo_profiles_location ON argo_profiles(latitude, longitude);
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_argo_measurements_profile ON argo_measurements(profile_id);
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_argo_measurements_depth ON argo_measurements(depth);
                """)
                
                logger.info("Database schema initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {str(e)}")
            raise
    
    def insert_profile(self, profile_data: Dict[str, Any]) -> int:
        """Insert a new ARGO profile and return the profile ID"""
        try:
            with self.connection.cursor() as cursor:
                insert_query = """
                    INSERT INTO argo_profiles 
                    (float_id, cycle_number, latitude, longitude, measurement_date, 
                     platform_number, data_center, file_hash)
                    VALUES (%(float_id)s, %(cycle_number)s, %(latitude)s, %(longitude)s, 
                            %(measurement_date)s, %(platform_number)s, %(data_center)s, %(file_hash)s)
                    RETURNING id;
                """
                cursor.execute(insert_query, profile_data)
                profile_id = cursor.fetchone()[0]
                logger.info(f"Inserted profile with ID: {profile_id}")
                return profile_id
                
        except psycopg2.IntegrityError as e:
            if "file_hash" in str(e):
                logger.warning(f"Profile already exists with hash: {profile_data.get('file_hash')}")
                return self.get_profile_id_by_hash(profile_data['file_hash'])
            else:
                raise
        except Exception as e:
            logger.error(f"Failed to insert profile: {str(e)}")
            raise
    
    def insert_measurements(self, profile_id: int, measurements: List[Dict[str, Any]]):
        """Insert measurements for a profile"""
        try:
            with self.connection.cursor() as cursor:
                # Add profile_id to each measurement
                for measurement in measurements:
                    measurement['profile_id'] = profile_id
                
                insert_query = """
                    INSERT INTO argo_measurements 
                    (profile_id, pressure, temperature, salinity, depth, oxygen, nitrate, ph, chlorophyll, quality_flag)
                    VALUES (%(profile_id)s, %(pressure)s, %(temperature)s, %(salinity)s, %(depth)s, 
                            %(oxygen)s, %(nitrate)s, %(ph)s, %(chlorophyll)s, %(quality_flag)s);
                """
                cursor.executemany(insert_query, measurements)
                logger.info(f"Inserted {len(measurements)} measurements for profile {profile_id}")
                
        except Exception as e:
            logger.error(f"Failed to insert measurements: {str(e)}")
            raise
    
    def get_profile_id_by_hash(self, file_hash: str) -> Optional[int]:
        """Get profile ID by file hash"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT id FROM argo_profiles WHERE file_hash = %s", (file_hash,))
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to get profile by hash: {str(e)}")
            return None
    
    def get_profiles(self, limit: int = 100, offset: int = 0, filters: Dict[str, Any] = None) -> pd.DataFrame:
        """Get ARGO profiles with optional filters"""
        try:
            base_query = """
                SELECT id, float_id, cycle_number, latitude, longitude, measurement_date, 
                       platform_number, data_center, created_at
                FROM argo_profiles
            """
            
            where_conditions = []
            params = []
            
            if filters:
                if filters.get('float_id'):
                    where_conditions.append("float_id = %s")
                    params.append(filters['float_id'])
                
                if filters.get('start_date'):
                    where_conditions.append("measurement_date >= %s")
                    params.append(filters['start_date'])
                
                if filters.get('end_date'):
                    where_conditions.append("measurement_date <= %s")
                    params.append(filters['end_date'])
                
                if filters.get('min_lat') is not None:
                    where_conditions.append("latitude >= %s")
                    params.append(filters['min_lat'])
                
                if filters.get('max_lat') is not None:
                    where_conditions.append("latitude <= %s")
                    params.append(filters['max_lat'])
                
                if filters.get('min_lon') is not None:
                    where_conditions.append("longitude >= %s")
                    params.append(filters['min_lon'])
                
                if filters.get('max_lon') is not None:
                    where_conditions.append("longitude <= %s")
                    params.append(filters['max_lon'])
            
            if where_conditions:
                base_query += " WHERE " + " AND ".join(where_conditions)
            
            base_query += " ORDER BY measurement_date DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            
            return pd.read_sql_query(base_query, self.connection, params=params)
            
        except Exception as e:
            logger.error(f"Failed to get profiles: {str(e)}")
            return pd.DataFrame()
    
    def get_measurements_by_profile(self, profile_id: int) -> pd.DataFrame:
        """Get all measurements for a specific profile"""
        try:
            query = """
                SELECT pressure, temperature, salinity, depth, oxygen, nitrate, ph, chlorophyll, quality_flag
                FROM argo_measurements
                WHERE profile_id = %s
                ORDER BY depth
            """
            return pd.read_sql_query(query, self.connection, params=[profile_id])
            
        except Exception as e:
            logger.error(f"Failed to get measurements for profile {profile_id}: {str(e)}")
            return pd.DataFrame()
    
    def get_total_records(self) -> int:
        """Get total number of profiles in the database"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM argo_profiles")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to get total records: {str(e)}")
            return 0
    
    def search_profiles_by_location(self, lat: float, lon: float, radius_km: float = 100) -> pd.DataFrame:
        """Search profiles within a radius of a given location"""
        try:
            # Using Haversine formula approximation
            query = """
                SELECT id, float_id, cycle_number, latitude, longitude, measurement_date,
                       (6371 * acos(cos(radians(%s)) * cos(radians(latitude)) * 
                        cos(radians(longitude) - radians(%s)) + sin(radians(%s)) * 
                        sin(radians(latitude)))) AS distance_km
                FROM argo_profiles
                WHERE (6371 * acos(cos(radians(%s)) * cos(radians(latitude)) * 
                       cos(radians(longitude) - radians(%s)) + sin(radians(%s)) * 
                       sin(radians(latitude)))) <= %s
                ORDER BY distance_km
                LIMIT 50
            """
            params = [lat, lon, lat, lat, lon, lat, radius_km]
            return pd.read_sql_query(query, self.connection, params=params)
            
        except Exception as e:
            logger.error(f"Failed to search profiles by location: {str(e)}")
            return pd.DataFrame()
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the database"""
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Basic counts
                cursor.execute("SELECT COUNT(*) as total_profiles FROM argo_profiles")
                total_profiles = cursor.fetchone()['total_profiles']
                
                cursor.execute("SELECT COUNT(*) as total_measurements FROM argo_measurements")
                total_measurements = cursor.fetchone()['total_measurements']
                
                cursor.execute("SELECT COUNT(DISTINCT float_id) as unique_floats FROM argo_profiles")
                unique_floats = cursor.fetchone()['unique_floats']
                
                # Date range
                cursor.execute("""
                    SELECT MIN(measurement_date) as earliest_date, 
                           MAX(measurement_date) as latest_date 
                    FROM argo_profiles
                """)
                date_range = cursor.fetchone()
                
                # Geographic coverage
                cursor.execute("""
                    SELECT MIN(latitude) as min_lat, MAX(latitude) as max_lat,
                           MIN(longitude) as min_lon, MAX(longitude) as max_lon
                    FROM argo_profiles
                """)
                geo_coverage = cursor.fetchone()
                
                return {
                    'total_profiles': total_profiles,
                    'total_measurements': total_measurements,
                    'unique_floats': unique_floats,
                    'date_range': {
                        'earliest': date_range['earliest_date'],
                        'latest': date_range['latest_date']
                    },
                    'geographic_coverage': {
                        'min_latitude': geo_coverage['min_lat'],
                        'max_latitude': geo_coverage['max_lat'],
                        'min_longitude': geo_coverage['min_lon'],
                        'max_longitude': geo_coverage['max_lon']
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get summary statistics: {str(e)}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
