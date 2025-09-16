import faiss
import numpy as np
import pickle
import os
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import hashlib
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSManager:
    """
    Manages FAISS vector database for ARGO profile metadata and summaries
    """
    
    def __init__(self, index_path: str = "./data/faiss_index", dimension: int = 256):
        self.index_path = index_path
        self.dimension = dimension
        self.index = None
        self.metadata = []
        self.vocabulary = {}
        self.vocab_size = 1000  # Limited vocabulary for simple hashing
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Initialize simple text encoder
        self._initialize_encoder()
        
        # Load existing index or create new one
        self._load_or_create_index()
    
    def _initialize_encoder(self):
        """Initialize a simple text encoder using feature hashing"""
        try:
            # Create common oceanographic terms vocabulary
            self.oceanographic_terms = {
                'temperature', 'salinity', 'pressure', 'depth', 'oxygen', 'nitrate', 'ph',
                'chlorophyll', 'float', 'profile', 'cycle', 'latitude', 'longitude',
                'north', 'south', 'east', 'west', 'ocean', 'sea', 'water', 'measurement',
                'data', 'surface', 'deep', 'shallow', 'high', 'low', 'warm', 'cold',
                'argo', 'bgc', 'biogeochemical', 'mixed', 'layer', 'thermocline'
            }
            logger.info("Initialized simple text encoder")
        except Exception as e:
            logger.error(f"Failed to initialize encoder: {str(e)}")
            raise
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create a new one"""
        try:
            if os.path.exists(f"{self.index_path}.index") and os.path.exists(f"{self.index_path}.metadata"):
                # Load existing index
                self.index = faiss.read_index(f"{self.index_path}.index")
                with open(f"{self.index_path}.metadata", 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded existing FAISS index with {len(self.metadata)} vectors")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                self.metadata = []
                logger.info("Created new FAISS index")
        except Exception as e:
            logger.error(f"Failed to load/create FAISS index: {str(e)}")
            # Create fresh index as fallback
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
    
    def save_index(self):
        """Save the FAISS index and metadata to disk"""
        try:
            faiss.write_index(self.index, f"{self.index_path}.index")
            with open(f"{self.index_path}.metadata", 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Saved FAISS index with {len(self.metadata)} vectors")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {str(e)}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to vector representation using simple feature hashing"""
        try:
            # Clean and tokenize text
            text_lower = text.lower()
            text_clean = re.sub(r'[^a-zA-Z0-9\s\.]', ' ', text_lower)
            tokens = text_clean.split()
            
            # Create feature vector using simple hashing
            vector = np.zeros(self.dimension, dtype='float32')
            
            # Hash each token to vector dimensions
            for token in tokens:
                if len(token) > 2:  # Skip very short tokens
                    # Simple hash function
                    hash_val = hash(token) % self.dimension
                    vector[hash_val] += 1.0
                    
                    # Add weight for oceanographic terms
                    if token in self.oceanographic_terms:
                        vector[hash_val] += 2.0
            
            # Add bi-grams for better context
            for i in range(len(tokens) - 1):
                if len(tokens[i]) > 2 and len(tokens[i+1]) > 2:
                    bigram = f"{tokens[i]}_{tokens[i+1]}"
                    hash_val = hash(bigram) % self.dimension
                    vector[hash_val] += 0.5
            
            # Normalize vector
            if np.linalg.norm(vector) > 0:
                vector = vector / np.linalg.norm(vector)
            
            return vector.astype('float32')
            
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            return np.zeros(self.dimension, dtype='float32')
    
    def add_profile(self, profile_summary: Dict[str, Any], profile_id: int):
        """Add a profile summary to the vector database"""
        try:
            # Create searchable text from profile summary
            text_parts = []
            
            # Add summary text if available
            if 'summary_text' in profile_summary:
                text_parts.append(profile_summary['summary_text'])
            
            # Add location information
            lat = profile_summary.get('latitude', 0)
            lon = profile_summary.get('longitude', 0)
            text_parts.append(f"Location: {lat:.2f}°N, {lon:.2f}°E")
            
            # Add temporal information
            if 'measurement_date' in profile_summary:
                date = profile_summary['measurement_date']
                if hasattr(date, 'strftime'):
                    text_parts.append(f"Date: {date.strftime('%Y-%m-%d')}")
                else:
                    text_parts.append(f"Date: {date}")
            
            # Add float information
            float_id = profile_summary.get('float_id', 'unknown')
            cycle_num = profile_summary.get('cycle_number', 0)
            text_parts.append(f"Float: {float_id}, Cycle: {cycle_num}")
            
            # Add parameter statistics
            if 'statistics' in profile_summary:
                stats = profile_summary['statistics']
                for param, param_stats in stats.items():
                    if isinstance(param_stats, dict) and 'mean' in param_stats:
                        text_parts.append(f"{param}: {param_stats['mean']:.2f}")
            
            # Combine all text parts
            search_text = ". ".join(text_parts)
            
            # Encode to vector
            vector = self.encode_text(search_text)
            
            # Add to index
            self.index.add(vector.reshape(1, -1))
            
            # Store metadata
            metadata_entry = {
                'profile_id': profile_id,
                'search_text': search_text,
                'summary': profile_summary,
                'vector_index': len(self.metadata)  # Index position in FAISS
            }
            self.metadata.append(metadata_entry)
            
            logger.info(f"Added profile {profile_id} to vector database")
            
        except Exception as e:
            logger.error(f"Failed to add profile to vector database: {str(e)}")
    
    def search(self, query: str, k: int = 10, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Search for similar profiles using natural language query"""
        try:
            if len(self.metadata) == 0:
                return []
            
            # Encode query
            query_vector = self.encode_text(query)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_vector.reshape(1, -1), min(k, len(self.metadata)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold and idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['similarity_score'] = float(score)
                    results.append(result)
            
            logger.info(f"Search query: '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vector database: {str(e)}")
            return []
    
    def search_by_location(self, latitude: float, longitude: float, radius_km: float = 100, k: int = 10) -> List[Dict[str, Any]]:
        """Search for profiles near a specific location"""
        try:
            # Create location-based query
            query = f"oceanographic data near {latitude:.2f} degrees north {longitude:.2f} degrees east within {radius_km} kilometers"
            
            # Get all results first
            all_results = self.search(query, k=len(self.metadata), threshold=0.0)
            
            # Filter by actual geographic distance
            filtered_results = []
            for result in all_results:
                summary = result.get('summary', {})
                result_lat = summary.get('latitude', 0)
                result_lon = summary.get('longitude', 0)
                
                # Calculate approximate distance (Haversine formula simplified)
                lat_diff = (result_lat - latitude) * 111  # Approximate km per degree
                lon_diff = (result_lon - longitude) * 111 * np.cos(np.radians(latitude))
                distance_km = np.sqrt(lat_diff**2 + lon_diff**2)
                
                if distance_km <= radius_km:
                    result['distance_km'] = distance_km
                    filtered_results.append(result)
            
            # Sort by distance and return top k
            filtered_results.sort(key=lambda x: x['distance_km'])
            return filtered_results[:k]
            
        except Exception as e:
            logger.error(f"Failed to search by location: {str(e)}")
            return []
    
    def search_by_parameter(self, parameter: str, min_value: float = None, max_value: float = None, k: int = 10) -> List[Dict[str, Any]]:
        """Search for profiles with specific parameter ranges"""
        try:
            # Create parameter-based query
            query_parts = [f"profiles with {parameter} measurements"]
            
            if min_value is not None:
                query_parts.append(f"{parameter} greater than {min_value}")
            
            if max_value is not None:
                query_parts.append(f"{parameter} less than {max_value}")
            
            query = " ".join(query_parts)
            
            # Search using text query
            results = self.search(query, k=len(self.metadata), threshold=0.0)
            
            # Filter by actual parameter values
            filtered_results = []
            for result in results:
                summary = result.get('summary', {})
                stats = summary.get('statistics', {})
                
                if parameter in stats:
                    param_stats = stats[parameter]
                    if isinstance(param_stats, dict) and 'mean' in param_stats:
                        param_value = param_stats['mean']
                        
                        # Check if value is within range
                        if (min_value is None or param_value >= min_value) and \
                           (max_value is None or param_value <= max_value):
                            filtered_results.append(result)
            
            return filtered_results[:k]
            
        except Exception as e:
            logger.error(f"Failed to search by parameter: {str(e)}")
            return []
    
    def search_by_date_range(self, start_date: str, end_date: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for profiles within a date range"""
        try:
            # Create date-based query
            query = f"oceanographic measurements between {start_date} and {end_date}"
            
            # Search using text query
            results = self.search(query, k=len(self.metadata), threshold=0.0)
            
            # Filter by actual dates
            from datetime import datetime
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            filtered_results = []
            for result in results:
                summary = result.get('summary', {})
                measurement_date = summary.get('measurement_date')
                
                if measurement_date:
                    if isinstance(measurement_date, str):
                        try:
                            measurement_dt = datetime.strptime(measurement_date[:10], '%Y-%m-%d')
                        except:
                            continue
                    else:
                        measurement_dt = measurement_date
                    
                    if start_dt <= measurement_dt <= end_dt:
                        filtered_results.append(result)
            
            return filtered_results[:k]
            
        except Exception as e:
            logger.error(f"Failed to search by date range: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        try:
            if not self.metadata:
                return {'total_profiles': 0}
            
            # Count profiles by various attributes
            float_ids = set()
            dates = []
            locations = []
            
            for entry in self.metadata:
                summary = entry.get('summary', {})
                
                if 'float_id' in summary:
                    float_ids.add(summary['float_id'])
                
                if 'measurement_date' in summary:
                    dates.append(summary['measurement_date'])
                
                if 'latitude' in summary and 'longitude' in summary:
                    locations.append((summary['latitude'], summary['longitude']))
            
            stats = {
                'total_profiles': len(self.metadata),
                'unique_floats': len(float_ids),
                'date_range': {
                    'earliest': min(dates) if dates else None,
                    'latest': max(dates) if dates else None
                },
                'geographic_coverage': {
                    'latitudes': [loc[0] for loc in locations],
                    'longitudes': [loc[1] for loc in locations]
                } if locations else {}
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get vector database statistics: {str(e)}")
            return {'total_profiles': 0}
    
    def remove_profile(self, profile_id: int):
        """Remove a profile from the vector database"""
        try:
            # Find the profile in metadata
            indices_to_remove = []
            for i, entry in enumerate(self.metadata):
                if entry.get('profile_id') == profile_id:
                    indices_to_remove.append(i)
            
            if not indices_to_remove:
                logger.warning(f"Profile {profile_id} not found in vector database")
                return
            
            # Remove from metadata (reverse order to maintain indices)
            for idx in reversed(indices_to_remove):
                del self.metadata[idx]
            
            # Note: FAISS doesn't support efficient removal of individual vectors
            # For production use, consider using a different vector database like Chroma
            # For now, we'll rebuild the index
            self._rebuild_index()
            
            logger.info(f"Removed profile {profile_id} from vector database")
            
        except Exception as e:
            logger.error(f"Failed to remove profile from vector database: {str(e)}")
    
    def _rebuild_index(self):
        """Rebuild the FAISS index from existing metadata"""
        try:
            # Create new index
            new_index = faiss.IndexFlatIP(self.dimension)
            
            # Re-encode all metadata entries
            vectors = []
            for entry in self.metadata:
                search_text = entry.get('search_text', '')
                vector = self.encode_text(search_text)
                vectors.append(vector)
            
            if vectors:
                vectors_array = np.array(vectors)
                new_index.add(vectors_array)
            
            self.index = new_index
            logger.info("Rebuilt FAISS index")
            
        except Exception as e:
            logger.error(f"Failed to rebuild FAISS index: {str(e)}")
    
    def clear_index(self):
        """Clear all data from the vector database"""
        try:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            logger.info("Cleared vector database")
        except Exception as e:
            logger.error(f"Failed to clear vector database: {str(e)}")
