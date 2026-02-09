from flask import Flask, request, jsonify
import asyncio
import json
import logging
import traceback
from datetime import datetime
import os

# Try to load environment variables, handle missing dotenv gracefully
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables should be set manually.")

app = Flask(__name__)
app.debug = True
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
EMBEDDINGS_DEPLOYMENT = os.environ.get("EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002")

async def process_chat_request(messages):
    """
    Process chat request - simplified version that will work without full SK imports
    """
    try:
        # Extract user query from messages
        user_query = None
        for message in messages:
            if message.get("role") == "user":
                user_query = message.get("content")
                break
        
        if not user_query:
            raise ValueError("No user query found in messages")

        logger.info(f"Processing query: {user_query}")

        # Try to import and use the actual chat_with_docs function
        try:
            from chat_with_docs import chat_with_docs
            # Call the actual function
            result = await chat_with_docs(messages)
            return result
        except ImportError as e:
            logger.warning(f"Could not import chat_with_docs: {e}")
            # Return a mock response with helpful information
            return {
                "content": f"I received your query: '{user_query}'. However, the full chat_with_docs functionality is not available due to missing dependencies. Please install: semantic-kernel, azure-ai-projects, openai, azure-search-documents, and other required packages.",
                "search_query": user_query,
                "original_query": user_query,
                "documents_found": 0,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "note": "This is a placeholder response. Install missing dependencies to enable full RAG functionality."
            }

    except Exception as e:
        logger.error(f"Error in process_chat_request: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    API endpoint for chat with documents
    """
    try:
        # Get the input data from the request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Invalid input, "query" is required'}), 400

        query = data['query']
        logger.info(f"[/api/chat] Received query: {query}")

        # Format messages for the chat function
        messages = [{"role": "user", "content": query}]

        # Run the async function
        try:
            # Use asyncio.run for clean event loop management
            response = asyncio.run(process_chat_request(messages))
        except RuntimeError:
            # Handle case where event loop is already running
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(process_chat_request(messages))
            finally:
                loop.close()

        logger.info(f"[/api/chat] Response status: {response.get('status', 'unknown')}")

        # Check if there was an error
        if response.get("status") == "error":
            return jsonify({
                'error': response["error"],
                'timestamp': response.get("timestamp", datetime.now().isoformat())
            }), 500

        # Return successful response
        return jsonify({
            'content': response["content"],
            'role': 'assistant',
            'search_query': response.get("search_query", query),
            'original_query': response.get("original_query", query),
            'documents_found': response.get("documents_found", 0),
            'timestamp': response.get("timestamp", datetime.now().isoformat()),
            'status': 'success',
            'note': response.get("note", "")
        })

    except Exception as e:
        logger.error(f"[/api/chat] Unexpected error: {str(e)}")
        logger.error(f"[/api/chat] Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    try:
        health_status = {
            'status': 'healthy',
            'service': 'chat_with_docs_api',
            'timestamp': datetime.now().isoformat(),
            'environment': {
                'azure_openai_endpoint': bool(AZURE_OPENAI_ENDPOINT),
                'azure_openai_key': bool(AZURE_OPENAI_KEY),
                'embeddings_deployment': bool(EMBEDDINGS_DEPLOYMENT)
            },
            'dependencies': {
                'flask': True,  # We know this works since we're running
                'dotenv': 'python-dotenv' in str(globals()),
                'chat_with_docs_available': False  # Will be True once imports work
            }
        }
        
        # Test if we can import chat_with_docs
        try:
            from chat_with_docs import chat_with_docs
            health_status['dependencies']['chat_with_docs_available'] = True
        except ImportError:
            health_status['dependencies']['chat_with_docs_available'] = False
            
        return jsonify(health_status)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/info', methods=['GET'])
def info():
    """
    API information endpoint
    """
    return jsonify({
        'service': 'Chat with Documents API',
        'description': 'RAG-based chat API using Azure AI Search and Semantic Kernel',
        'version': '1.0.0',
        'azure_services': [
            'Azure OpenAI',
            'Azure AI Search', 
            'Azure AI Inference'
        ],
        'endpoints': {
            '/api/chat': {
                'method': 'POST',
                'description': 'Main chat endpoint',
                'accepts': {'query': 'string'},
                'returns': 'AI response with grounding documents'
            },
            '/api/health': {
                'method': 'GET',
                'description': 'Health check endpoint'
            },
            '/api/info': {
                'method': 'GET', 
                'description': 'API information'
            }
        },
        'required_packages': [
            'flask',
            'python-dotenv',
            'semantic-kernel',
            'azure-ai-projects',
            'openai',
            'azure-search-documents',
            'azure-identity',
            'azure-core'
        ],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint
    """
    return jsonify({
        'message': 'Chat with Documents API is running',
        'endpoints': ['/api/chat', '/api/health', '/api/info'],
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("Starting Chat with Documents API...")
    logger.info(f"Azure OpenAI Endpoint: {AZURE_OPENAI_ENDPOINT}")
    logger.info(f"Embeddings Deployment: {EMBEDDINGS_DEPLOYMENT}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
