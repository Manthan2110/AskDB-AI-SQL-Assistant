# Enhanced Flask Backend for Text-to-SQL AI Assistant
# Optimized version with better error handling and data parsing

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
import json
import logging
from typing import List, Dict, Any, Optional

# Import your existing LangChain components
from langchain.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app, origins=["http://localhost:5000", "http://127.0.0.1:5000"])

# ============================================
# Configuration
# ============================================
class Config:
    # Database Configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '3306')
    DB_USERNAME = os.getenv('DB_USERNAME', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'root')
    DB_SCHEMA = os.getenv('DB_SCHEMA', 'text_to_sql')
    
    # API Configuration
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyB4d5FAmv1i6Rgt2meMEgd4g0UiBA5Lhuc')
    LLM_MODEL = os.getenv('LLM_MODEL', 'gemini-2.5-flash')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.3'))
    
    # Application Configuration
    MAX_HISTORY_SIZE = int(os.getenv('MAX_HISTORY_SIZE', '50'))
    MAX_TABLE_RESULTS = int(os.getenv('MAX_TABLE_RESULTS', '100'))

# ============================================
# Database Setup
# ============================================
try:
    mysql_uri = f"mysql+pymysql://{Config.DB_USERNAME}:{Config.DB_PASSWORD}@{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_SCHEMA}"
    db = SQLDatabase.from_uri(mysql_uri, sample_rows_in_table_info=2)
    logger.info(f"✅ Connected to database: {Config.DB_SCHEMA}")
except Exception as e:
    logger.error(f"❌ Database connection failed: {e}")
    db = None

# ============================================
# LLM Setup
# ============================================
try:
    llm = ChatGoogleGenerativeAI(
        model=Config.LLM_MODEL,
        api_key=Config.GOOGLE_API_KEY,
        temperature=Config.LLM_TEMPERATURE
    )
    logger.info(f"✅ LLM initialized: {Config.LLM_MODEL}")
except Exception as e:
    logger.error(f"❌ LLM initialization failed: {e}")
    llm = None

# ============================================
# Prompt Templates
# ============================================
SQL_TEMPLATE = """You are an expert SQL query generator. Based on the table schema below, generate a syntactically correct SQL query that answers the user's question.

IMPORTANT RULES:
1. Only return the SQL query without any explanation or additional text
2. Use ONLY the table names and column names shown in the schema
3. Write the query in a single line without line breaks
4. Ensure the query is syntactically valid for MySQL
5. Use appropriate JOINs when querying multiple tables
6. Use WHERE clauses to filter data when needed
7. If the question asks for aggregation, use appropriate aggregate functions (COUNT, SUM, AVG, etc.)
8. If the question asks for ordering, use ORDER BY
9. If the question asks for limiting results, use LIMIT
10. Be careful with column names and table names - they are case sensitive

Database Schema:
{schema}

User Question: {question}

SQL Query:"""

SUMMARY_TEMPLATE = """You are an expert data analyst. Your task is to summarize SQL query results in a simple, natural, and user-friendly explanation.

Question asked by user:
{question}

SQL Query executed:
{query}

Raw SQL results:
{results}

You are a helpful data analyst and communication expert. Your task is to explain SQL query results to users in a clear, conversational, and friendly way. The explanation should read naturally and be concise yet insightful.

Follow these rules carefully:
1. Write as if you are talking to a non-technical person. Avoid all database or SQL jargon such as “SELECT,” “WHERE,” or “JOIN.”
2. Summarize the results using natural, human-friendly language — focus on what the data reveals, not how it was retrieved.
3. Always mention key figures, comparisons, or patterns found in the data (like totals, averages, or trends)
4. If there are many records, summarize the highlights instead of listing everything. Mention the most important few, followed by a general overview.
5. If no records were found, politely explain that there are no matching results.
6. Keep the response short, helpful, and fluent — aim for 2–4 sentences.
7. Use a positive, analytical tone, similar to a friendly data analyst or dashboard assistant.
8. When applicable, conclude with an interpretive insight, e.g., “This suggests that sales improved in Q2,” or “This indicates low customer activity in this region.”

Your response:"""

# Create prompt templates
sql_prompt = ChatPromptTemplate.from_template(SQL_TEMPLATE)
summary_prompt = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)

# ============================================
# Helper Functions
# ============================================
def get_database_schema():
    """Get database schema information"""
    if not db:
        return ""
    try:
        return db.get_table_info()
    except Exception as e:
        logger.error(f"Error getting schema: {e}")
        return ""

def create_langchain_chains():
    """Create LangChain processing chains"""
    if not db or not llm:
        return None, None
        
    try:
        # SQL Generation Chain
        sql_chain = (
            RunnablePassthrough.assign(schema=lambda _: get_database_schema())
            | sql_prompt
            | llm.bind(stop=["\n"])
            | StrOutputParser()
        )
        
        # Summary Generation Chain  
        summary_chain = summary_prompt | llm | StrOutputParser()
        
        return sql_chain, summary_chain
    except Exception as e:
        logger.error(f"Error creating chains: {e}")
        return None, None

# Initialize chains
sql_chain, summary_chain = create_langchain_chains()

# ============================================
# Query History Management
# ============================================
query_history = []

def add_to_history(question: str, sql_query: str, results_count: int, timestamp: str):
    """Add query to history with size limit"""
    query_history.append({
        'id': len(query_history) + 1,
        'question': question,
        'sql': sql_query,
        'results_count': results_count,
        'timestamp': timestamp
    })
    
    # Maintain history size limit
    while len(query_history) > Config.MAX_HISTORY_SIZE:
        query_history.pop(0)

# ============================================
# Data Processing Functions  
# ============================================
def parse_sql_results(raw_results: str) -> List[Dict[str, Any]]:
    """
    Enhanced parsing of SQL results from your LangChain implementation
    """
    if not raw_results or str(raw_results).strip() == '':
        return []
    
    try:
        # Convert to string if not already
        results_str = str(raw_results).strip()
        
        # Handle empty results
        if results_str in ['', 'None', '[]']:
            return []
        
        # If it looks like a list of tuples or structured data
        if results_str.startswith('[(') and results_str.endswith(')]'):
            # Parse tuple format: [('value1', 'value2'), ('value3', 'value4')]
            return parse_tuple_results(results_str)
        
        # If it's a simple list
        if results_str.startswith('[') and results_str.endswith(']'):
            try:
                # Try to evaluate as Python literal
                import ast
                parsed = ast.literal_eval(results_str)
                if isinstance(parsed, list):
                    return format_list_results(parsed)
            except (ValueError, SyntaxError):
                pass
        
        # Handle single values or simple text results
        return [{'result': results_str}]
        
    except Exception as e:
        logger.error(f"Error parsing SQL results: {e}")
        return [{'result': str(raw_results)}]

def parse_tuple_results(results_str: str) -> List[Dict[str, Any]]:
    """Parse results in tuple format"""
    try:
        import ast
        parsed = ast.literal_eval(results_str)
        
        if not parsed:
            return []
        
        # If we have tuple results, convert to dict format
        formatted_results = []
        for i, row in enumerate(parsed):
            if isinstance(row, (tuple, list)):
                row_dict = {}
                for j, value in enumerate(row):
                    row_dict[f'column_{j+1}'] = value
                formatted_results.append(row_dict)
            else:
                formatted_results.append({'value': row})
        
        return formatted_results[:Config.MAX_TABLE_RESULTS]
        
    except Exception as e:
        logger.error(f"Error parsing tuple results: {e}")
        return [{'result': results_str}]

def format_list_results(parsed_list: List) -> List[Dict[str, Any]]:
    """Format list results for display"""
    if not parsed_list:
        return []
    
    formatted = []
    for item in parsed_list:
        if isinstance(item, dict):
            formatted.append(item)
        elif isinstance(item, (list, tuple)):
            row_dict = {}
            for i, val in enumerate(item):
                row_dict[f'column_{i+1}'] = val
            formatted.append(row_dict)
        else:
            formatted.append({'value': item})
    
    return formatted[:Config.MAX_TABLE_RESULTS]

def extract_table_info_from_schema(schema_info: str) -> List[Dict[str, Any]]:
    """Enhanced table information extraction"""
    tables = []
    
    if not schema_info:
        # Return known tables from your schema as fallback
        return get_fallback_tables()
    
    try:
        # Split by table sections
        sections = schema_info.split('\n\n')
        
        for section in sections:
            if 'TABLE' in section.upper():
                table_info = parse_table_section(section)
                if table_info:
                    tables.append(table_info)
        
        # If no tables parsed, return fallback
        if not tables:
            return get_fallback_tables()
            
        return tables
        
    except Exception as e:
        logger.error(f"Error extracting table info: {e}")
        return get_fallback_tables()

def parse_table_section(section: str) -> Optional[Dict[str, Any]]:
    """Parse individual table section"""
    lines = [line.strip() for line in section.split('\n') if line.strip()]
    
    table_name = None
    columns = []
    
    for line in lines:
        # Look for table name
        if 'TABLE' in line.upper():
            parts = line.split()
            for i, part in enumerate(parts):
                if part.upper() == 'TABLE' and i + 1 < len(parts):
                    table_name = parts[i + 1].strip('`')
                    break
        
        # Look for column definitions
        elif table_name and line and not line.startswith('--'):
            # Simple column parsing
            parts = line.split()
            if len(parts) >= 2:
                col_name = parts[0].strip('`')
                col_type = parts[1].upper()
                
                columns.append({
                    'name': col_name,
                    'type': col_type
                })
    
    if table_name:
        return {
            'name': table_name,
            'columns': columns,
            'row_count': 'N/A'
        }
    
    return None

def get_fallback_tables() -> List[Dict[str, Any]]:
    """Return known tables when parsing fails"""
    return [
        {
            'name': 'budget',
            'columns': [
                {'name': 'ProductNames', 'type': 'TEXT'},
                {'name': 'Budget', 'type': 'INTEGER'}
            ],
            'row_count': 'N/A'
        },
        {
            'name': 'customers', 
            'columns': [
                {'name': 'CustomerNameIndex', 'type': 'INTEGER'},
                {'name': 'CustomerNames', 'type': 'TEXT'}
            ],
            'row_count': 'N/A'
        },
        {
            'name': 'orders',
            'columns': [
                {'name': 'OrderNumber', 'type': 'TEXT'},
                {'name': 'OrderDate', 'type': 'TEXT'},
                {'name': 'CustomerNameIndex', 'type': 'INTEGER'},
                {'name': 'ProductIndex', 'type': 'INTEGER'},
                {'name': 'ProductQuantity', 'type': 'INTEGER'},
                {'name': 'UnitPrice', 'type': 'DOUBLE'},
                {'name': 'LineTotal', 'type': 'DOUBLE'}
            ],
            'row_count': 'N/A'
        },
        {
            'name': 'products',
            'columns': [
                {'name': 'ProductIndex', 'type': 'INTEGER'},
                {'name': 'ProductNames', 'type': 'TEXT'}
            ],
            'row_count': 'N/A'
        },
        {
            'name': 'regions',
            'columns': [
                {'name': 'name', 'type': 'TEXT'},
                {'name': 'state', 'type': 'TEXT'}
            ],
            'row_count': 'N/A'
        },
        {
            'name': 'stateregions',
            'columns': [
                {'name': 'StateCode', 'type': 'TEXT'},
                {'name': 'State', 'type': 'TEXT'},
                {'name': 'Region', 'type': 'TEXT'}
            ],
            'row_count': 'N/A'
        }
    ]

# ============================================
# Routes
# ============================================

@app.route('/')
def index():
    """Render main chat interface"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def process_query():
    """Main endpoint for processing natural language queries"""
    if not db or not sql_chain or not summary_chain:
        return jsonify({
            'success': False,
            'error': 'System not properly initialized. Check database and API key configuration.'
        }), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        question = data.get('question', '').strip()
        if not question:
            return jsonify({
                'success': False,
                'error': 'Question cannot be empty'
            }), 400
        
        logger.info(f"Processing query: {question}")
        
        # Step 1: Generate SQL query
        sql_query = sql_chain.invoke({"question": question})
        logger.info(f"Generated SQL: {sql_query}")
        
        # Step 2: Execute SQL query
        raw_results = db.run(sql_query)
        logger.info(f"Raw results: {raw_results}")
        
        # Step 3: Parse results for frontend
        parsed_results = parse_sql_results(raw_results)
        
        # Step 4: Generate natural language summary
        summary = summary_chain.invoke({
            "question": question,
            "query": sql_query,
            "results": str(raw_results)
        })
        
        # Step 5: Save to history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        add_to_history(question, sql_query, len(parsed_results), timestamp)
        
        return jsonify({
            'success': True,
            'sql_query': sql_query,
            'results': parsed_results,
            'summary': summary,
            'timestamp': timestamp,
            'results_count': len(parsed_results)
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({
            'success': False,
            'error': f'Query processing failed: {str(e)}'
        }), 500

@app.route('/api/schema', methods=['GET'])
def get_database_schema_endpoint():
    """Get database schema information"""
    if not db:
        return jsonify({
            'success': False,
            'error': 'Database not connected'
        }), 500
    
    try:
        schema_info = get_database_schema()
        tables = extract_table_info_from_schema(schema_info)
        
        return jsonify({
            'success': True,
            'database': Config.DB_SCHEMA,
            'connection_status': 'connected',
            'tables': tables,
            'table_count': len(tables)
        })
        
    except Exception as e:
        logger.error(f"Error getting schema: {e}")
        return jsonify({
            'success': False,
            'error': f'Schema retrieval failed: {str(e)}'
        }), 500

@app.route('/api/history', methods=['GET'])
def get_query_history():
    """Get recent query history"""
    try:
        # Return last 20 queries in reverse order (most recent first)
        recent_history = list(reversed(query_history[-20:]))
        
        return jsonify({
            'success': True,
            'history': recent_history,
            'total_queries': len(query_history)
        })
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({
            'success': False,
            'error': f'History retrieval failed: {str(e)}'
        }), 500

@app.route('/api/execute', methods=['POST'])
def execute_custom_sql():
    """Execute a custom SQL query (optional advanced feature)"""
    if not db:
        return jsonify({
            'success': False,
            'error': 'Database not connected'
        }), 500
    
    try:
        data = request.get_json()
        sql_query = data.get('sql', '').strip()
        
        if not sql_query:
            return jsonify({
                'success': False,
                'error': 'SQL query cannot be empty'
            }), 400
        
        # Basic SQL injection protection
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        if any(keyword in sql_query.upper() for keyword in dangerous_keywords):
            return jsonify({
                'success': False,
                'error': 'Only SELECT queries are allowed for security reasons'
            }), 400
        
        # Execute query
        raw_results = db.run(sql_query)
        parsed_results = parse_sql_results(raw_results)
        
        return jsonify({
            'success': True,
            'results': parsed_results,
            'results_count': len(parsed_results)
        })
        
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        return jsonify({
            'success': False,
            'error': f'SQL execution failed: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'database_connected': db is not None,
        'llm_initialized': llm is not None,
        'timestamp': datetime.now().isoformat()
    })

# ============================================
# Error Handlers
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ============================================
# Main Application
# ============================================

if __name__ == '__main__':
    # Startup checks
    print("🚀 Starting Text-to-SQL AI Assistant...")
    print(f"📊 Database: {Config.DB_SCHEMA} ({'✅ Connected' if db else '❌ Not Connected'})")
    print(f"🤖 LLM: {Config.LLM_MODEL} ({'✅ Ready' if llm else '❌ Not Ready'})")
    print(f"🌐 Server: http://localhost:5000")
    
    if not db:
        print("⚠️  Warning: Database not connected. Check your configuration.")
    if not llm:
        print("⚠️  Warning: LLM not initialized. Check your API key.")
    
    # Run the application
    app.run(
        debug=os.getenv('FLASK_DEBUG', 'True').lower() == 'true',
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 5000))
    )