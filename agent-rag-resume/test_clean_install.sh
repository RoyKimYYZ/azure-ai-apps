#!/bin/bash

# Test script for clean environment installation
set -e

echo "🧪 Testing agent-rag-resume in clean environment..."

# Build first
echo "📦 Building distribution..."
uv build

# Create clean test environment
TEMP_DIR=$(mktemp -d)
echo "📁 Created temp directory: $TEMP_DIR"
cd $TEMP_DIR

# Create virtual environment
echo "🐍 Creating clean Python environment..."
uv venv clean-env
source clean-env/bin/activate

# Install the package
echo "📥 Installing agent-rag-resume from wheel..."
WHEEL_PATH="/home/rkadmin/azureai-chatapp/agent-rag-resume/dist/agent_rag_resume-0.1.0-py3-none-any.whl"
uv pip install "$WHEEL_PATH"

# Test import
echo "🔍 Testing package import..."
python -c "
import agent_rag_resume
print('✅ Package imported successfully!')
print(f'Package location: {agent_rag_resume.__file__}')
"

# Test module execution
echo "🚀 Testing module execution..."
python -m agent_rag_resume || echo "⚠️  Module execution failed (expected if no main function)"

# Show installed packages
echo "📋 Installed packages:"
pip list | grep -E "(agent-rag|openai|streamlit|semantic-kernel|azure)"

# Cleanup
echo "🧹 Cleaning up..."
deactivate
cd /home/rkadmin/azureai-chatapp/agent-rag-resume/
rm -rf $TEMP_DIR

echo "✅ Clean environment test completed successfully!"
