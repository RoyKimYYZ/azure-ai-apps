

# check if python3 is installed
which python3
python3 --version

# login to azure
az login --use-device-code


# create virtual environment if venv does not exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv  
else
  echo "Virtual environment already exists."
  
# activate virtual environment
source venv/bin/activate
deactivate

pip install --upgrade pip
pip install  -r requirements.txt

# install packages explicitly if not installed via requirements.txt
pip install dotenv azure.identity
pip install azure-search-documents==11.4.0b11 tiktoken

pip install matplotlib langchain-community==0.0.16 pypdf
pip install azure-ai-projects
pip install pandas
pip install azure-ai-inference

python config.py

python upload-data-blobstorage.py

pip install azure-search-documents --upgrade

# creates resume-index and upload 13 documents
python create_search_index_resume.py

# https://learn.microsoft.com/en-us/azure/search/search-howto-indexing-azure-blob-storage?source=recommendations
# https://learn.microsoft.com/en-us/azure/search/search-howto-indexing-azure-blob-storage?source=recommendations#configure-and-run-the-blob-indexer

# Obsolete. just for reference
#pip install azure-search-documents==11.4.0b11 
#python create_index_run_indexer.py 

# Run AI Search Indexer job against resume-index 
python create_skillset_run_indexerv2.py

pip install opentelemetry-api


python search_resumes.py --query "I am looking for a data scientist with experience in machine learning and Python programming. Can you help me find someone?"


pip install azure-monitor-opentelemetry
python chat_with_resumes.py --query "I am working on a project that requires a data scientist with experience in machine learning and R programming. Can you help me find someone?" --enable-telemetry
python chat_with_resumes.py --query "I am working on a tech project that requires a cloud engineer with skills and knowledge in Azure and Terraform. Preferably more than 2 years of working experience. Can you help me find someone?" --enable-telemetry

pip install flask


# Test the API using curl or Postman
# Purpose: test the API endpoints using curl or Postman
# The API server will listen on port 5000
# The API server will use the same environment variables as the main
python chat_with_resumes_api.py

# Test the API using curl
# In another terminal window with the same env variables and venv activated
# Test the API using curl or Postman
# Purpose: test the API endpoints using curl or Postman
curl -X POST "http://localhost:5001 /api/chat" -H "Content-Type: application/json" -d '{"query": "I am working on a tech project that requires a cloud engineer with skills and knowledge in Azure and Terraform. Preferably more than 2 years of working experience. Can you help me find someone?"}'


# Optinally, you can run the API server in background
# Run the API server in background, in another terminal window with the same env variables and venv activated
# Test the API using curl or Postman
# Purpose: test the API endpoints using curl or Postman
# The API server will listen on port 5001
# The API server will use the same environment variables as the main app
python search_resumes_api.py