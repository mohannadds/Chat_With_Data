import json
import uuid
import pandas as pd
import re
import requests
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Request, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType
from pymongo import MongoClient
from datetime import datetime
import functools
import time

# Initialize FastAPI app
app = FastAPI(
    title="Data Analysis API",
    description="API for analyzing data with LLM-powered pandas agent",
    version="1.0.0"
)

# Add CORS middleware - THIS IS THE KEY ADDITION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - replace with specific origins in production
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Configure static file serving and templates
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Initialize MongoDB client
MONGO_URI = "mongodb://localhost:27017"  # Update with your MongoDB URI
client = MongoClient(MONGO_URI, connect=False)  # Lazy connection to improve startup time
db = client["JsonChat"]
sessions_collection = db["sessions"]

# Create TTL index to automatically expire old sessions in MongoDB
# This will delete documents after 24 hours (86400 seconds)
try:
    sessions_collection.create_index("created_at", expireAfterSeconds=86400)
except Exception as e:
    print(f"Warning: Could not create TTL index: {str(e)}")

# Session cache class with LRU functionality
class SessionCache:
    def __init__(self, expiry_minutes=30, max_size=100):
        self.sessions = {}
        self.last_accessed = {}
        self.expiry_seconds = expiry_minutes * 60
        self.max_size = max_size
    
    def get(self, session_id):
        """Get a session from cache and update its last accessed time"""
        if session_id in self.sessions:
            self.last_accessed[session_id] = datetime.now()
            return self.sessions[session_id]
        return None
    
    def set(self, session_id, session_data):
        """Add or update a session in the cache"""
        # Enforce max size before adding new item
        if session_id not in self.sessions and len(self.sessions) >= self.max_size:
            self.enforce_max_size()
            
        self.sessions[session_id] = session_data
        self.last_accessed[session_id] = datetime.now()
    
    def delete(self, session_id):
        """Remove a session from the cache"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            del self.last_accessed[session_id]
            return True
        return False
    
    def cleanup_expired(self):
        """Remove sessions that haven't been accessed recently"""
        now = datetime.now()
        expired_sessions = [
            sid for sid, last_access in self.last_accessed.items()
            if (now - last_access).total_seconds() > self.expiry_seconds
        ]
        for sid in expired_sessions:
            del self.sessions[sid]
            del self.last_accessed[sid]
        return len(expired_sessions)
    
    def enforce_max_size(self):
        """Remove oldest sessions if cache exceeds maximum size"""
        if len(self.sessions) <= self.max_size:
            return 0
            
        # Sort sessions by last access time (oldest first)
        sorted_sessions = sorted(
            self.last_accessed.items(),
            key=lambda x: x[1]
        )
        
        # Calculate how many sessions to remove to get under max size
        sessions_to_remove = len(self.sessions) - self.max_size
        
        # Remove oldest sessions
        removed = 0
        for i in range(sessions_to_remove):
            if i < len(sorted_sessions):
                sid = sorted_sessions[i][0]
                del self.sessions[sid]
                del self.last_accessed[sid]
                removed += 1
                
        return removed

# Initialize the session cache
session_cache = SessionCache(expiry_minutes=30, max_size=100)

# Create a custom TTL cache implementation
class TimedCache:
    def __init__(self, ttl_seconds=300):
        self.cache = {}
        self.timestamps = {}
        self.ttl = ttl_seconds
    
    def get(self, key):
        """Get a value from cache if it exists and is not expired"""
        if key in self.cache:
            now = time.time()
            if now - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                # Remove expired entry
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key, value):
        """Add or update a value in the cache"""
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.timestamps.clear()

# Initialize a cache for ollama models with 5-minute TTL
ollama_models_cache = TimedCache(ttl_seconds=300)

# Background task for periodic cleanup
async def periodic_cleanup():
    while True:
        try:
            # Clean up sessions every 10 minutes
            await asyncio.sleep(600)
            count = session_cache.cleanup_expired()
            print(f"Cleaned up {count} expired sessions from memory")
        except Exception as e:
            print(f"Error during session cleanup: {str(e)}")

# Start the cleanup task when the app starts
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_cleanup())

# Clean up MongoDB connection when app shuts down
@app.on_event("shutdown")
async def shutdown_event():
    client.close()
    print("MongoDB connection closed")

# Define Pydantic models for request/response validation
class SessionResponse(BaseModel):
    session_id: str
    message: str
    dataframe_shape: List[int]
    columns: List[str]
    model_name: str

# NEW: Define a Pydantic model for the query request
class QueryRequest(BaseModel):
    session_id: str
    question: str

class QueryResponse(BaseModel):
    session_id: str
    question: str
    response: str
    previous_question: Optional[str] = None
    previous_response: Optional[str] = None
    model_name: str

class InfoResponse(BaseModel):
    session_id: str
    dataframe_shape: List[int]
    columns: List[str]
    dtypes: Dict[str, str]
    sample: List[Dict[str, Any]]
    model_name: str

class EndSessionResponse(BaseModel):
    session_id: str
    message: str

class ConversationHistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, str]]

class OllamaModelsResponse(BaseModel):
    models: List[Dict[str, Any]]

# Performance decorator for logging execution time
def timing_decorator(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

# Function to clean up the agent's responses - optimized version
def clean_agent_response(response):
    """
    Clean up the agent's response to make it more user-friendly.
    Optimized for performance.
    """
    if not response:
        return "I couldn't process your request. Please try again."
    
    # Quick check before using regex
    if "Number of" in response:
        # Try to find and return just the number statement
        number_match = re.search(r"Number of[^:]*: \d+", response)
        if number_match:
            return number_match.group(0)
    
    # Cleanup pipeline - batch regexes together
    patterns = [
        (r"Thought:.*?(?=\n\n|$)", ""),
        (r"Action Input:\s*```python.*?```", "")
    ]
    
    cleaned = response
    for pattern, replacement in patterns:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.DOTALL)
    
    # If the response became empty after cleaning
    if not cleaned.strip():
        # Check for print statements first (most common)
        print_match = re.search(r'print\(f"([^"]+)"\)', response)
        if print_match:
            return print_match.group(1).replace('{', '').replace('}', '')
        
        # Look for any numbers in the response
        number_match = re.search(r'\d+', response)
        if number_match:
            return f"The result is {number_match.group(0)}"
            
        return "Analysis complete. Please let me know if you need any specific information."
    
    return cleaned.strip()

# Custom error handler for parsing errors - optimized
def handle_parsing_errors(error):
    """Custom handler for parsing errors with optimized extraction"""
    error_str = str(error)
    
    # Quick check before more expensive operations
    if "Could not parse LLM output: `" in error_str:
        # Extract the content between the backticks
        start_idx = error_str.find("Could not parse LLM output: `") + len("Could not parse LLM output: `")
        end_idx = error_str.rfind("`\nFor troubleshooting")
        
        if end_idx > start_idx:
            # Return the actual LLM output
            return error_str[start_idx:end_idx]
    
    return "I had trouble understanding. Could you rephrase your question or provide more details?"

# Fixed implementation for getting Ollama models using the custom cache
def get_installed_ollama_models():
    """Get a list of all installed Ollama models with custom caching"""
    cache_key = "ollama_models"
    cached_models = ollama_models_cache.get(cache_key)
    
    if cached_models is not None:
        return cached_models
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2.0)
        if response.status_code == 200:
            models = response.json().get("models", [])
            ollama_models_cache.set(cache_key, models)
            return models
        else:
            raise HTTPException(status_code=response.status_code, 
                              detail=f"Failed to get models from Ollama API: {response.text}")
    except requests.RequestException as e:
        raise HTTPException(status_code=500, 
                          detail=f"Error connecting to Ollama API: {str(e)}")
#deepseek-r1:14b-qwen-distill-q4_K_M
#qwen2.5:32b-instruct-q2_K
def create_agent(df, model_name="qwen2.5:32b-instruct-q2_K"):
    """Create a Pandas DataFrame agent with specified Ollama model"""
    try:
        # Optimize by pre-calculating column list and datatypes
        column_info = ", ".join([f"{col} ({df[col].dtype})" for col in df.columns])
        
        llm = Ollama(
            model=model_name,
            temperature=0.1,  # Lower temperature for more factual responses
            top_k=40,
            num_ctx=8000
        )
        
        # Optimize agent creation with specific config for faster execution
        return create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,  # Changed to False for improved performance
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            max_iterations=12,  # Balanced between accuracy and speed
            early_stopping=True,
            handle_parsing_errors=handle_parsing_errors,
            allow_dangerous_code=True,
            prefix=f"""You are an AI assistant specializing in data analysis. The dataframe has {len(df.columns)} columns: {column_info}.
Please follow these guidelines:
 
 1. Language selection:
    - Respond in the ARABIC or ENGLISH, language as the user's question (Arabic or English NO OTHER LANGUAGES).
    - You're FLUENT in both English and Arabic.
 2. Data analysis focus:
    - Prioritize clear explanations and methodologies
    - Verify your calculations and double-check answers
    - When in doubt, query the dataframe directly
    - Handle complex queries carefully and  double-check everytime
    - Provide numbers with the answers everytime
 4. Response format:
    - Provide clear, direct answers without showing internal thinking
    - Never include code
    - Verify answers accurately reflect the dataframe data
    - explain your answers
5. Only Answer about the data not any other topic. 
"""
        )
    except Exception as e:
        print(f"Error creating agent with model {model_name}: {str(e)}")
        raise

# Optimized data summary creation
def create_data_summary(df):
    """Generate a concise summary of the dataframe - optimized version"""
    # Pre-compute all the needed info at once
    shape = df.shape
    columns = df.columns.tolist()
    dtypes = {col: str(df[col].dtype) for col in columns}
    
    # Select only the most important columns if there are many
    if len(columns) > 10:
        sample_cols = columns[:10]
        sample_note = f" (showing first 10 of {len(columns)})"
    else:
        sample_cols = columns
        sample_note = ""
    
    # Build the summary string efficiently
    summary_parts = [
        f"DataFrame shape: {shape[0]} rows × {shape[1]} columns.",
        f"Columns{sample_note}: {', '.join(sample_cols)}.",
    ]
    
    # Add sample data (limited to first 2 rows for performance)
    sample_data = df.head(2).to_dict(orient="records")
    if sample_data:
        sample_str = "Sample data: " + json.dumps(sample_data[0], default=str)
        if len(sample_str) > 500:  # Truncate if too long
            sample_str = sample_str[:497] + "..."
        summary_parts.append(sample_str)
    
    return "\n".join(summary_parts)

# Optimized query execution with retry
def run_with_retry(agent, df, question, conversation_history=None, max_retries=1):
    """Execute query with retry mechanism and data grounding - optimized"""
    # Create a minimal but effective data summary
    data_summary = create_data_summary(df)
    
    # Only include the last exchange for efficiency
    context = ""
    if conversation_history and len(conversation_history) > 0:
        last_exchange = conversation_history[-1]
        context = f"Previous question: {last_exchange.get('question', '')}\n"
        context += f"Previous answer: {last_exchange.get('response', '')}\n\n"
    
    # Combine data summary, context, and question efficiently
    full_query = f"{data_summary}\n{context}Current question: {question}"
    
    # Try with shorter retry loop for performance
    for attempt in range(max_retries + 1):
        try:
            return agent.run(full_query)
        except Exception as e:
            if attempt == max_retries:
                # Last attempt failed, handle error
                return handle_parsing_errors(e)
            # Simplify the query for the retry
            full_query = f"{data_summary}\nPlease answer directly: {question}"
    
    return "I'm unable to process this question. Please try rephrasing it."

# Route to serve the HTML interface
@app.get("/", response_class=HTMLResponse)
async def get_html(request: Request):
    with open(BASE_DIR / "templates" / "index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/api/models", response_model=OllamaModelsResponse)
async def list_models():
    """API endpoint to list all installed Ollama models"""
    try:
        models = get_installed_ollama_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data", response_model=SessionResponse)
@timing_decorator
async def receive_data(
    json_data: List[Dict[str, Any]],
    model_name: str = Query("qwen2.5:32b-instruct-q2_K", description="Name of Ollama model to use")
):
    """API endpoint to receive JSON data and create a dataframe."""
    try:
        # Quick validation for model existence
        available_models = get_installed_ollama_models()
        model_names = [model["name"] for model in available_models]
        
        if model_name not in model_names:
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{model_name}' not found. Available models: {', '.join(model_names)}"
            )
        
        # Generate session ID first for logging
        session_id = str(uuid.uuid4())
        print(f"Processing data for new session {session_id}")
        
        # Convert JSON to pandas DataFrame - optimize for speed with fewer checks
        try:
            df = pd.DataFrame(json_data)
        except Exception as df_error:
            print(f"DataFrame creation error: {str(df_error)}")
            raise HTTPException(status_code=400, detail=f"Invalid data format: {str(df_error)}")
        
        # Create pandas agent with specified model
        agent = create_agent(df, model_name)
        
        # Store the agent in the session cache
        session_cache.set(session_id, {
            "agent": agent,
            "dataframe": df,
            "model_name": model_name
        })
        
        # MODIFIED: Store only metadata in MongoDB, not the full dataframe
        session_data = {
            "session_id": session_id,
            # Remove dataframe storage from MongoDB
            "columns": df.columns.tolist(),
            "shape": list(df.shape),
            "model_name": model_name,
            "created_at": datetime.utcnow(),
            "queries": [],
            "conversation_history": [],
            "query_count": 0,
            # Add a small sample for reference (first 5 rows only)
            "sample_data": json.loads(df.head(5).to_json(orient="records"))
        }
        
        # Async insert to DB
        sessions_collection.insert_one(session_data)
        
        return {
            "session_id": session_id,
            "message": "Data processed successfully",
            "dataframe_shape": list(df.shape),
            "columns": df.columns.tolist(),
            "model_name": model_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in receive_data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# CHANGED: Changed from GET to POST and updated parameter handling
@app.post("/api/query", response_model=QueryResponse)
@timing_decorator
async def query_data(
    request: QueryRequest = Body(..., description="Query request with session ID and question")
):
    """API endpoint to query the dataframe using natural language."""
    session_id = request.session_id
    question = request.question
    
    print(f"Processing query for session {session_id}: {question[:50]}...")
    
    # Quick check if session exists in MongoDB
    session_data = sessions_collection.find_one(
        {"session_id": session_id},
        {"_id": 0, "sample_data": 0}  # Exclude large fields for faster query
    )
    if not session_data:
        raise HTTPException(status_code=404, detail="Invalid or expired session ID")
    
    try:
        # Update the query count - batched with other updates later
        query_count = session_data.get("query_count", 0) + 1
        
        # Get from cache or fail - MODIFIED to remove MongoDB dataframe fallback
        cached_session = session_cache.get(session_id)
        model_name = session_data.get("model_name", "qwen2.5:32b-instruct-q2_K")
        
        if cached_session and "dataframe" in cached_session:
            df = cached_session["dataframe"]
            agent = cached_session["agent"]
            
            # Refresh every 10 queries but only if needed
            if query_count % 10 == 0:
                print(f"Refreshing agent for session {session_id}")
                agent = create_agent(df, model_name)
                session_cache.set(session_id, {
                    "agent": agent,
                    "dataframe": df,
                    "model_name": model_name
                })
        else:
            # MODIFIED: Instead of getting from MongoDB, return error suggesting to reload data
            raise HTTPException(
                status_code=404, 
                detail="Session data not available in memory. Please reload your data to continue."
            )
        
        # Get minimal conversation history - just last exchange for context
        conversation_history = session_data.get("conversation_history", [])
        previous_question = None
        previous_response = None
        
        if conversation_history:
            last_exchange = conversation_history[-1]
            previous_question = last_exchange.get("question")
            previous_response = last_exchange.get("response")
        
        # Process the query with optimized retry approach
        try:
            start_time = time.time()
            raw_response = run_with_retry(agent, df, question, conversation_history, max_retries=1)
            query_time = time.time() - start_time
            print(f"Query processing took {query_time:.2f} seconds")
            
            # Clean up the response
            response = clean_agent_response(raw_response)
        except Exception as agent_error:
            print(f"Agent error: {str(agent_error)}")
            response = "I encountered an error. Please try rephrasing your question."
            raw_response = str(agent_error)
        
        # Prepare updates for MongoDB - batch them together
        timestamp = datetime.utcnow()
        
        # Create conversation entry
        conversation_entry = {
            "question": question,
            "response": response,
            "timestamp": timestamp
        }
        
        # Create query entry (with more details)
        query_entry = {
            "question": question,
            "response": response,
            "raw_response": raw_response if 'raw_response' in locals() else "Error occurred",
            "timestamp": timestamp
        }
        
        # Batch update to MongoDB
        sessions_collection.update_one(
            {"session_id": session_id},
            {"$push": {
                "queries": query_entry,
                "conversation_history": conversation_entry
            },
            "$set": {"query_count": query_count}}
        )
        
        return {
            "session_id": session_id,
            "question": question,
            "response": response,
            "previous_question": previous_question,
            "previous_response": previous_response,
            "model_name": model_name
        }
        
    except Exception as e:
        print(f"Error in query_data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/info", response_model=InfoResponse)
async def get_info(
    session_id: str = Query(..., description="Session ID received from /api/data endpoint")
):
    """Get information about the dataframe for a given session"""
    # First try to get minimal session data
    session_data = sessions_collection.find_one(
        {"session_id": session_id},
        {"_id": 0, "shape": 1, "columns": 1, "model_name": 1, "sample_data": 1}
    )
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Invalid or expired session ID")
    
    # MODIFIED: Try to get from cache, otherwise use sample data from MongoDB
    cached_session = session_cache.get(session_id)
    if cached_session and "dataframe" in cached_session:
        df = cached_session["dataframe"]
        sample_data = json.loads(df.head(5).to_json(orient="records"))
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    else:
        # Use the sample data stored in MongoDB instead of full dataframe
        sample_data = session_data.get("sample_data", [])
        # Create a mini dataframe from sample for dtypes
        if sample_data:
            mini_df = pd.DataFrame(sample_data)
            dtypes = {col: str(dtype) for col, dtype in mini_df.dtypes.items()}
        else:
            dtypes = {col: "unknown" for col in session_data.get("columns", [])}
    
    return {
        "session_id": session_id,
        "dataframe_shape": session_data.get("shape", [0, 0]),
        "columns": session_data.get("columns", []),
        "dtypes": dtypes,
        "sample": sample_data,
        "model_name": session_data.get("model_name", "qwen2.5:32b-instruct-q2_K")
    }

@app.get("/api/conversation_history", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    session_id: str = Query(..., description="Session ID"),
    limit: int = Query(10, description="Maximum exchanges to return")
):
    """Retrieve conversation history for a given session."""
    # More efficient query that only gets the needed fields
    session_data = sessions_collection.find_one(
        {"session_id": session_id},
        {"_id": 0, "conversation_history": 1}
    )
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Invalid or expired session ID")
    
    # Get limited history
    conversation_history = session_data.get("conversation_history", [])
    limited_history = conversation_history[-limit:] if conversation_history else []
    
    return {
        "session_id": session_id,
        "history": limited_history
    }

@app.delete("/api/end_session", response_model=EndSessionResponse)
async def end_session(
    session_id: str = Query(..., description="Session ID to delete")
):
    """End a session by deleting it from MongoDB."""
    # Minimal check if session exists
    session_exists = sessions_collection.count_documents({"session_id": session_id}, limit=1)
    if not session_exists:
        raise HTTPException(status_code=404, detail="Invalid or expired session ID")
    
    try:
        # Delete session from MongoDB
        sessions_collection.delete_one({"session_id": session_id})
        
        # Remove from in-memory cache
        session_cache.delete(session_id)
        
        return {
            "session_id": session_id,
            "message": "Session successfully deleted"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ADDED: New endpoint to handle session recovery if memory cache is lost
@app.post("/api/reload_session", response_model=SessionResponse)
@timing_decorator
async def reload_session(
    json_data: List[Dict[str, Any]],
    session_id: str = Query(..., description="Session ID to recover"),
    model_name: str = Query("qwen2.5:32b-instruct-q2_K", description="Model name to use")
):
    """Reload a session's data if it was lost from memory cache."""
    # Check if session exists in MongoDB
    session_data = sessions_collection.find_one({"session_id": session_id})
    if not session_data:
        raise HTTPException(status_code=404, detail="Invalid session ID")
    
    try:
        # Convert JSON to pandas DataFrame
        df = pd.DataFrame(json_data)
        
        # Create pandas agent with specified model
        agent = create_agent(df, model_name)
        
        # Store in session cache
        session_cache.set(session_id, {
            "agent": agent,
            "dataframe": df,
            "model_name": model_name
        })
        
        # Update session in MongoDB with new metadata
        sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": {
                "columns": df.columns.tolist(),
                "shape": list(df.shape),
                "model_name": model_name,
                "sample_data": json.loads(df.head(5).to_json(orient="records"))
            }}
        )
        
        return {
            "session_id": session_id,
            "message": "Session data reloaded successfully",
            "dataframe_shape": list(df.shape),
            "columns": df.columns.tolist(),
            "model_name": model_name
        }
        
    except Exception as e:
        print(f"Error in reload_session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

from a2wsgi import ASGIMiddleware

# Expose the WSGI callable for wfastcgi/IIS
wsgi_app = ASGIMiddleware(app)

if __name__ == "__main__":
    import uvicorn
    # Use multiple workers for better concurrency
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        workers=8,  # Increased from 2
        log_level="info",
        timeout_keep_alive=65
    )