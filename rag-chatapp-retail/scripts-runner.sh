source .venv/bin/activate

pip install azure-ai-projects azure-ai-inference azure-identity 
python -m pip install -r requirements.txt

# Simple chatbot test against Azure AI Foundry SDK
python chat.py

# need to create Azure AI Search and set as connected resources in Azure AI Foundry projets

# Create a new search index and uploads the data from the CSV file
python create_search_index.py --index-name "example-index" \
    --csv-file "assets/products.csv"

python get_product_documents.py --query "what is cost of Hiking Shoes?"

python get_product_documents.py --query "I need a new tent for 4 people, what would you recommend?"


python chat_with_products.py --query "I need a new tent for 4 people, what would you recommend?" 

python chat_with_products.py --query "I need a hiking shoes, what is price?" 
python chat_with_products.py --query "I need TrailWalker Hiking Shoe, what is price?" 

pip install azure-monitor-opentelemetry
python chat_with_products.py --query "I need a new tent for 4 people, what would you recommend?" --enable-telemetry
pip install azure_ai-evaluation

python evaluate.py