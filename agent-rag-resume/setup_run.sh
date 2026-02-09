cd /home/rkadmin/azureai-chatapp/agent-rag-resume/

# Run the main Streamlit app with uv
echo "Starting Azure RAG Chat with uv..."

# Check which app to run
if [ -f "agent_rag_resume/ai_foundry_agent_sk.py" ]; then
    echo "Running Semantic Kernel version..."
    uv run streamlit run agent_rag_resume/ai_foundry_agent_sk.py
elif [ -f "agent_rag_resume/azure_rag_chat_ai_foundry_agent_style_python_single_file_app.py" ]; then
    echo "Running original Azure RAG Chat app..."
    uv run streamlit run agent_rag_resume/azure_rag_chat_ai_foundry_agent_style_python_single_file_app.py
else
    echo "Error: No Streamlit app found in agent_rag_resume/"
    exit 1
fi

# Install/sync dependencies
uv sync

# Run Streamlit apps
uv run streamlit run agent_rag_resume/azure_rag_chat_ai_foundry_agent_style_python_single_file_app.py

# Run as module
uv run python -m agent_rag_resume

# Development dependencies  
uv add --dev pytest black ruff

# Build package
uv build


# old non-uv setup
python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

streamlit run azure_rag_chat_ai_foundry_agent_style_python_single_file_app.py

deactivate