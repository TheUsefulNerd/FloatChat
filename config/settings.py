import os
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables with fallback defaults
    """
    
    config = {
        # Groq API Configuration
        'groq_api_key': os.getenv('GROQ_API_KEY', ''),
        'groq_model': os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant'),
        
        # PostgreSQL Configuration
        'database_url': os.getenv('DATABASE_URL', ''),
        'db_host': os.getenv('PGHOST', 'localhost'),
        'db_port': int(os.getenv('PGPORT', '5432')),
        'db_name': os.getenv('PGDATABASE', 'argo_data'),
        'db_user': os.getenv('PGUSER', 'postgres'),
        'db_password': os.getenv('PGPASSWORD', ''),
        
        # Application Configuration
        'session_secret': os.getenv('SESSION_SECRET', 'default_session_secret'),
        'upload_max_size': int(os.getenv('UPLOAD_MAX_SIZE', '200')),  # MB
        
        # Vector Store Configuration
        'vector_dimension': int(os.getenv('VECTOR_DIMENSION', '384')),
        'faiss_index_path': os.getenv('FAISS_INDEX_PATH', './data/faiss_index'),
        
        # Data Processing Configuration
        'temp_data_path': os.getenv('TEMP_DATA_PATH', './data/temp'),
        'processed_data_path': os.getenv('PROCESSED_DATA_PATH', './data/processed'),
        
        # Visualization Configuration
        'map_default_zoom': int(os.getenv('MAP_DEFAULT_ZOOM', '2')),
        'plot_height': int(os.getenv('PLOT_HEIGHT', '500')),
        'plot_width': int(os.getenv('PLOT_WIDTH', '800')),
    }
    
    # Validate critical configurations
    if not config['groq_api_key']:
        print("Warning: GROQ_API_KEY not configured. AI chat functionality will be limited.")
    
    if not config['database_url'] and not all([config['db_host'], config['db_name'], config['db_user']]):
        print("Warning: Database configuration incomplete. Some features may not work.")
    
    # Create necessary directories
    os.makedirs(config['temp_data_path'], exist_ok=True)
    os.makedirs(config['processed_data_path'], exist_ok=True)
    os.makedirs(os.path.dirname(config['faiss_index_path']), exist_ok=True)
    
    return config

def get_database_connection_string(config: Dict[str, Any]) -> str:
    """
    Build database connection string from configuration
    """
    if config['database_url']:
        return config['database_url']
    
    return f"postgresql://{config['db_user']}:{config['db_password']}@{config['db_host']}:{config['db_port']}/{config['db_name']}"

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate that all required configuration is present
    """
    required_fields = ['groq_api_key', 'db_host', 'db_name', 'db_user']
    
    for field in required_fields:
        if not config.get(field):
            print(f"Error: Required configuration field '{field}' is missing")
            return False
    
    return True
