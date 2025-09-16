"""
Utility functions and helpers for the ARGO Oceanographic Platform
"""

__version__ = "1.0.0"
__author__ = "ARGO Platform Team"

# Import commonly used utilities
from .helpers import (
    format_data_for_display,
    create_download_link,
    validate_coordinates,
    calculate_distance,
    format_parameter_value,
    get_parameter_info
)

__all__ = [
    'format_data_for_display',
    'create_download_link', 
    'validate_coordinates',
    'calculate_distance',
    'format_parameter_value',
    'get_parameter_info'
]
