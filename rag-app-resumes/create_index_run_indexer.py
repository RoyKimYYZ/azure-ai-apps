from azure.search.documents.indexes import (
    SearchIndexClient, 
    SearchIndexerClient
)
from config import azure_openai_key, azure_openai_embedding_deployment_id, azure_openai_endpoint, search_endpoint, search_credential, blob_connection_string, blob_container, search_index, search_datasource, search_skillset, search_indexer, resume_doc_index_name
from lib.common import (
    create_search_index,
    create_search_datasource,
    create_search_skillset,
    create_search_indexer
)

search_index_client = SearchIndexClient(endpoint=search_endpoint, credential=search_credential)
index = create_search_index(
    resume_doc_index_name,
    azure_openai_endpoint,
    azure_openai_embedding_deployment_id,
    azure_openai_key
)

print( resume_doc_index_name )
print( azure_openai_endpoint )
print( azure_openai_embedding_deployment_id )
print( azure_openai_key )
print( search_datasource )
print( f"blob container {blob_container}"  )
print( f"skillset {search_skillset}" )

search_skillset
search_index_client.create_or_update_index(index)

search_indexer_client = SearchIndexerClient(endpoint=search_endpoint, credential=search_credential)

data_source = create_search_datasource(
    search_datasource,
    blob_connection_string,
    blob_container
)

search_indexer_client.create_or_update_data_source_connection(data_source)

# Create a skillset that uses the OpenAI skill to embed text
embedding_skillset = create_search_skillset(
    skillset_name=search_skillset,
    index_name=resume_doc_index_name,
    azure_openai_endpoint=azure_openai_endpoint,
    azure_openai_embedding_deployment_id=azure_openai_embedding_deployment_id,
    azure_openai_key=azure_openai_key,
    text_split_mode='pages', # split text into pages
    maximum_page_length=2000, # 2000 words per page
    page_overlap_length=500, # 500 words overlap between pages
    
)

    
search_indexer_client.create_or_update_skillset(embedding_skillset)

# Create an indexer that uses the data source, index, and skillset
indexer = create_search_indexer(
    indexer_name=search_indexer,
    index_name=resume_doc_index_name,
    datasource_name=search_datasource,
    skillset_name=search_skillset,
) # field mapping to metadata_storage_name

# indexer = SearchIndexer(
#         name=search_indexer,
#         data_source_name=search_datasource,
#         target_index_name=search_index,
#         skillset_name=search_skillset
#     )
# skillset = SearchIndexerSkillset(  
#     name=skillset_name,  
#     description="Skillset to chunk documents and generating embeddings",  
#     skills=skills,  
#     index_projection=index_projections,
#     cognitive_services_account=cognitive_services_account
# )
  
  
search_indexer_client.create_or_update_indexer(indexer)
search_indexer_client.run_indexer(search_indexer)

print("Running indexer")
