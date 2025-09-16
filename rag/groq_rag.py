import os
from typing import List, Dict, Any, Optional
import logging
from groq import Groq
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroqRAG:
    """
    Retrieval-Augmented Generation system using Groq for ultra-fast LLM inference
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable.")
        
        # Initialize Groq clients
        self.groq_client = Groq(api_key=self.api_key)
        self.chat_model = ChatGroq(
            api_key=self.api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1024
        )
        
        self.system_prompt = self._create_system_prompt()
        
    def _create_system_prompt(self) -> str:
        """Create the system prompt for oceanographic data queries"""
        return """You are an expert oceanographic data analyst with deep knowledge of ARGO float data and marine science. 

Your role is to help users understand and analyze oceanographic data from ARGO floats. ARGO floats are autonomous profiling instruments that measure temperature, salinity, pressure, and sometimes biogeochemical parameters like oxygen, nitrate, pH, and chlorophyll in the world's oceans.

Key knowledge areas:
- ARGO float operations and data collection methods
- Oceanographic parameters: temperature, salinity, pressure, density, oxygen, nutrients
- Ocean physics: mixed layer depth, thermocline, halocline, water masses
- Biogeochemical cycles and marine ecosystems
- Data quality assessment and interpretation
- Geographic oceanography and regional characteristics

When answering questions:
1. Provide scientifically accurate information
2. Explain oceanographic concepts clearly for different expertise levels
3. Reference specific data when available in the context
4. Suggest appropriate visualizations or analyses
5. Highlight data quality considerations
6. Use proper oceanographic terminology

If you're asked to generate SQL queries or data analysis code, ensure it's appropriate for oceanographic data structures and follows best practices for scientific data analysis.

Always be helpful, accurate, and educational in your responses."""

    def generate_sql_query(self, user_question: str, database_schema: Dict[str, Any]) -> str:
        """Generate SQL query from natural language question"""
        try:
            schema_description = self._format_schema_description(database_schema)
            
            prompt = f"""
Given this database schema for ARGO oceanographic data:

{schema_description}

Generate a PostgreSQL query to answer this question: "{user_question}"

Rules:
1. Use only the tables and columns shown in the schema
2. Include appropriate WHERE clauses for filtering
3. Use JOINs when data from multiple tables is needed
4. Include ORDER BY and LIMIT clauses when appropriate
5. Handle NULL values appropriately
6. Use oceanographically meaningful constraints (e.g., reasonable temperature ranges)

Return only the SQL query without explanations:
"""
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=512
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the response to extract just the SQL
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1].strip()
            
            logger.info(f"Generated SQL query for: {user_question}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Failed to generate SQL query: {str(e)}")
            return ""
    
    def _format_schema_description(self, schema: Dict[str, Any]) -> str:
        """Format database schema for prompt"""
        schema_text = "Database Tables:\n\n"
        
        tables_info = {
            'argo_profiles': {
                'description': 'Main profile information for each ARGO float deployment',
                'columns': [
                    'id (PRIMARY KEY)',
                    'float_id (VARCHAR) - ARGO float identifier',
                    'cycle_number (INTEGER) - Profile cycle number',
                    'latitude (DECIMAL) - Measurement latitude',
                    'longitude (DECIMAL) - Measurement longitude', 
                    'measurement_date (TIMESTAMP) - Date/time of measurement',
                    'platform_number (VARCHAR) - Platform identifier',
                    'data_center (VARCHAR) - Data center code'
                ]
            },
            'argo_measurements': {
                'description': 'Individual measurements at different depths',
                'columns': [
                    'id (PRIMARY KEY)',
                    'profile_id (INTEGER) - Foreign key to argo_profiles',
                    'pressure (DECIMAL) - Water pressure in decibars',
                    'temperature (DECIMAL) - Water temperature in Celsius',
                    'salinity (DECIMAL) - Practical salinity in PSU',
                    'depth (DECIMAL) - Depth in meters',
                    'oxygen (DECIMAL) - Dissolved oxygen in micromole/kg',
                    'nitrate (DECIMAL) - Nitrate in micromole/kg',
                    'ph (DECIMAL) - pH value',
                    'chlorophyll (DECIMAL) - Chlorophyll-a in mg/m3',
                    'quality_flag (INTEGER) - Data quality flag (1=good, 4=bad)'
                ]
            }
        }
        
        for table_name, table_info in tables_info.items():
            schema_text += f"Table: {table_name}\n"
            schema_text += f"Description: {table_info['description']}\n"
            schema_text += "Columns:\n"
            for column in table_info['columns']:
                schema_text += f"  - {column}\n"
            schema_text += "\n"
        
        return schema_text
    
    def answer_question_with_context(self, question: str, retrieved_data: List[Dict[str, Any]]) -> str:
        """Answer question using retrieved data as context"""
        try:
            # Format retrieved data for context
            context = self._format_retrieved_data(retrieved_data)
            
            prompt = f"""
Based on the following ARGO oceanographic data:

{context}

Please answer this question: "{question}"

Provide a comprehensive answer that:
1. Directly addresses the question
2. Explains relevant oceanographic concepts
3. References specific data points when available
4. Suggests additional analysis if appropriate
5. Notes any limitations or data quality considerations

Be scientifically accurate and educational in your response.
"""
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1024
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer for question: {question}")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to answer question with context: {str(e)}")
            return "I apologize, but I encountered an error while processing your question. Please try again."
    
    def _format_retrieved_data(self, data: List[Dict[str, Any]]) -> str:
        """Format retrieved data for use in prompts"""
        if not data:
            return "No specific data found for this query."
        
        formatted_parts = []
        
        for i, item in enumerate(data[:5], 1):  # Limit to top 5 results
            part = f"Profile {i}:\n"
            
            # Profile metadata
            summary = item.get('summary', {})
            if 'float_id' in summary:
                part += f"  Float ID: {summary['float_id']}\n"
            if 'latitude' in summary and 'longitude' in summary:
                part += f"  Location: {summary['latitude']:.2f}°N, {summary['longitude']:.2f}°E\n"
            if 'measurement_date' in summary:
                part += f"  Date: {summary['measurement_date']}\n"
            
            # Statistics
            if 'statistics' in summary:
                stats = summary['statistics']
                part += "  Measurements:\n"
                for param, param_stats in stats.items():
                    if isinstance(param_stats, dict):
                        mean_val = param_stats.get('mean', 'N/A')
                        min_val = param_stats.get('min', 'N/A')
                        max_val = param_stats.get('max', 'N/A')
                        part += f"    {param.title()}: {mean_val:.2f} (range: {min_val:.2f} - {max_val:.2f})\n"
            
            # Search text summary
            if 'search_text' in item:
                part += f"  Summary: {item['search_text']}\n"
            
            formatted_parts.append(part)
        
        return "\n".join(formatted_parts)
    
    def suggest_visualizations(self, question: str, data_type: str = None) -> List[str]:
        """Suggest appropriate visualizations based on the question"""
        try:
            prompt = f"""
For this oceanographic data question: "{question}"

Suggest 3-5 appropriate data visualizations that would help answer this question effectively.

Consider these visualization types:
- Geographic maps (float trajectories, spatial distributions)
- Depth profiles (temperature-salinity, depth-time plots)
- Time series plots
- Scatter plots (parameter relationships)
- Contour plots (sections, climatologies)
- Statistical plots (histograms, box plots)

For each suggestion, briefly explain why it would be useful.

Format your response as a numbered list.
"""
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=512
            )
            
            suggestions = response.choices[0].message.content.strip()
            
            # Parse suggestions into a list
            suggestion_lines = [line.strip() for line in suggestions.split('\n') if line.strip()]
            return suggestion_lines
            
        except Exception as e:
            logger.error(f"Failed to suggest visualizations: {str(e)}")
            return ["Geographic map showing float locations", "Depth profile plots", "Time series analysis"]
    
    def explain_oceanographic_concept(self, concept: str) -> str:
        """Explain oceanographic concepts for educational purposes"""
        try:
            prompt = f"""
Explain the oceanographic concept of "{concept}" in a clear and educational way.

Include:
1. Definition and basic explanation
2. How it relates to ARGO float measurements
3. Why it's important in oceanography
4. How it can be observed or calculated from data
5. Real-world examples or applications

Make the explanation accessible to users with varying levels of oceanographic knowledge.
"""
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1024
            )
            
            explanation = response.choices[0].message.content.strip()
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to explain concept: {str(e)}")
            return f"I apologize, but I couldn't provide an explanation for '{concept}' at this time."
    
    def generate_analysis_code(self, analysis_type: str, parameters: List[str]) -> str:
        """Generate Python code for specific oceanographic analyses"""
        try:
            prompt = f"""
Generate Python code for performing this oceanographic analysis: "{analysis_type}"

Parameters to analyze: {', '.join(parameters)}

Assume the data is in a pandas DataFrame called 'measurements_df' with columns:
- depth, pressure, temperature, salinity, oxygen, nitrate, ph, chlorophyll
- quality_flag (1=good, 4=bad)

And a DataFrame 'profiles_df' with:
- latitude, longitude, measurement_date, float_id

Generate clean, well-commented Python code that:
1. Filters for good quality data
2. Performs the requested analysis
3. Creates appropriate visualizations
4. Follows oceanographic best practices

Include necessary imports and explain the methodology.
"""
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=1024
            )
            
            code = response.choices[0].message.content.strip()
            return code
            
        except Exception as e:
            logger.error(f"Failed to generate analysis code: {str(e)}")
            return "# Error generating analysis code"
    
    def chat_with_history(self, current_question: str, chat_history: List[Dict[str, str]]) -> str:
        """Chat with conversation history context"""
        try:
            # Build conversation history
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add chat history
            for exchange in chat_history[-5:]:  # Last 5 exchanges for context
                if 'user' in exchange:
                    messages.append({"role": "user", "content": exchange['user']})
                if 'assistant' in exchange:
                    messages.append({"role": "assistant", "content": exchange['assistant']})
            
            # Add current question
            messages.append({"role": "user", "content": current_question})
            
            response = self.groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1024
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            logger.error(f"Failed to chat with history: {str(e)}")
            return "I apologize, but I encountered an error. Please try rephrasing your question."
    
    async def query(self, question: str, context_data: List[Dict[str, Any]] = None) -> str:
        """Async query method for MCP integration"""
        try:
            if context_data:
                return self.answer_question_with_context(question, context_data)
            else:
                # Simple query without context
                response = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": question}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=1024
                )
                
                answer = response.choices[0].message.content.strip()
                return answer
        except Exception as e:
            logger.error(f"Failed to process query: {str(e)}")
            return "I apologize, but I encountered an error while processing your question."

# Create alias for compatibility with MCP integration
GroqRAGSystem = GroqRAG
