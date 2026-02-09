import os
# Purpose: Create an Azure Search index, data source, skillset, and indexer for a blob storage container.
# resume-index
# data source: container resume-samples
# skillset: resume-skillset 

from azure.search.documents.indexes.models import (
    SplitSkill,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    AzureOpenAIEmbeddingSkill,
    EntityRecognitionSkill,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    IndexProjectionMode,
    SearchIndexerSkillset,
    CognitiveServicesAccountKey,
    SearchIndexer,
    FieldMapping,
    IndexingParameters,
    IndexingParametersConfiguration
)
from azure.search.documents.indexes import SearchIndexerClient
from azure.identity import DefaultAzureCredential
from azure.identity import get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
from config import ( azure_openai_key, 
    azure_openai_embedding_deployment_id, 
    azure_openai_endpoint, 
    search_endpoint, search_credential, 
    blob_connection_string, blob_container, 
    search_index, search_datasource, 
    search_skillset)

from lib.common import (
    create_search_index,
    create_search_datasource,
    create_search_skillset,
    create_search_indexer
)


credential = DefaultAzureCredential()

AZURE_SEARCH_SERVICE: str = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT", "https://<your-search-service>.search.windows.net")
AZURE_SEARCH_KEY: str = os.environ.get("AZURE_SEARCH_KEY", "<YOUR_AZURE_SEARCH_KEY>")
AZURE_OPENAI_ACCOUNT: str = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://<your-openai-resource>.openai.azure.com")
AZURE_OPENAI_KEY: str = os.environ.get("AZURE_OPENAI_KEY", "<YOUR_AZURE_OPENAI_KEY>")
AZURE_AI_MULTISERVICE_ACCOUNT: str = os.environ.get("AZURE_AI_MULTISERVICE_ACCOUNT", "https://<your-ai-services>.cognitiveservices.azure.com")
AZURE_AI_MULTISERVICE_KEY: str = os.environ.get("AZURE_AI_MULTISERVICE_KEY", "<YOUR_AZURE_AI_MULTISERVICE_KEY>")
AZURE_STORAGE_CONNECTION: str = os.environ.get("AZURE_STORAGE_CONNECTION", "<YOUR_AZURE_STORAGE_CONNECTION_STRING>")


# Create a skillset  
skillset_name = "resume-skillset"

# This splitskill is used to chunk the documents into pages
# It splits the text into pages of 2000 words with a 500 word overlap
# The split skill is used to chunk the documents into pages
# It splits the text into pages of 2000 words with a 500 word overlap
split_skill = SplitSkill(  
    description="Split skill to chunk documents",  
    text_split_mode="pages", 
    context="/document",  # /document is the root context
    maximum_page_length=2000,  
    page_overlap_length=500,  
    inputs=[  
        InputFieldMappingEntry(name="text", source="/document/content"),   # content is the field to be split
    ],
    outputs=[  
        OutputFieldMappingEntry(name="textItems", target_name="pages")  # pages is the field to store the split text
    ],  
)  

embedding_skill = AzureOpenAIEmbeddingSkill(  
    description="Skill to generate embeddings via Azure OpenAI",  
    context="/document/pages/*",  
    resource_url=azure_openai_endpoint,  
    api_key=azure_openai_key,
    deployment_name="text-embedding-ada-002",  
    model_name="text-embedding-ada-002",
    dimensions=1536,
    inputs=[  
        InputFieldMappingEntry(name="text", source="/document/pages/*"),  
    ],  
    outputs=[  
        OutputFieldMappingEntry(name="embedding", target_name="text_vector")  
    ],  
)

entity_skill = EntityRecognitionSkill(
    description="Skill to recognize entities in text",
    context="/document/pages/*",
    categories=["Location"],
    default_language_code="en",
    inputs=[
        InputFieldMappingEntry(name="text", source="/document/pages/*")
    ],
    outputs=[
        OutputFieldMappingEntry(name="locations", target_name="locations")
    ]
)

print(f"search index name:" + search_index)

index_projections = SearchIndexerIndexProjection(  
    selectors=[  
        SearchIndexerIndexProjectionSelector(  
            target_index_name=search_index,  
            parent_key_field_name="parent_id",  
            source_context="/document/pages/*",  
            mappings=[  
                InputFieldMappingEntry(name="chunk", source="/document/pages/*"),  
                InputFieldMappingEntry(name="text_vector", source="/document/pages/*/text_vector"),
                InputFieldMappingEntry(name="locations", source="/document/pages/*/locations"),  
                InputFieldMappingEntry(name="title", source="/document/metadata_storage_name"),  
                InputFieldMappingEntry(name="metadata_storage_name", source="/document/metadata_storage_name"),  
                InputFieldMappingEntry(name="metadata_storage_path", source="/document/metadata_storage_path"),  
                InputFieldMappingEntry(name="last_modified", source="/document/metadata_storage_last_modified"),  # Example for metadata property
                InputFieldMappingEntry(name="author", source="/document/author"),  # Custom metadata key. DO NOT DO FIELDMAPPING in Indexer
                InputFieldMappingEntry(name="document_type", source="/document/document_type")  # Custom metadata key. DO NOT DO FIELDMAPPING in Indexer
                
            ],  
        ),  
    ],  
    parameters=SearchIndexerIndexProjectionsParameters(  
        projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS  
    ),  
)

cognitive_services_account = CognitiveServicesAccountKey(key=AZURE_AI_MULTISERVICE_KEY)

skills = [split_skill, embedding_skill, entity_skill]

skillset = SearchIndexerSkillset(  
    name=skillset_name,  
    description="Skillset to chunk documents and generating embeddings",  
    skills=skills,  # List of skills to be used in the skillset including text split skill
    index_projection=index_projections,
    cognitive_services_account=cognitive_services_account,
)
  
client = SearchIndexerClient(endpoint=AZURE_SEARCH_SERVICE, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
client.create_or_update_skillset(skillset)  
print(f"Skillset {skillset.name} created")  

indexer_name = "resume-indexerv2"

print(f"Indexer {indexer_name}")  
print(f"Data Source {search_datasource}")  

# Blob Indexing: https://learn.microsoft.com/en-us/azure/search/search-howto-indexing-azure-blob-storage?source=recommendations#indexing-blob-metadata

indexer = SearchIndexer(
        name=indexer_name,
        data_source_name=search_datasource,
        target_index_name=search_index,
        skillset_name=skillset_name,
        parameters = IndexingParameters( # https://medium.com/@tanishk_rane/automating-azure-ai-search-workflows-with-python-342644c639a6
            configuration=IndexingParametersConfiguration(
                data_to_extract="contentAndMetadata",
                parsing_mode="default",
                image_action="generateNormalizedImages", 
                allow_skillset_to_read_file_data=True,
                query_timeout=None
            ),
            max_failed_items=-1,
            max_failed_items_per_batch=-1,
            batch_size=1
        ),
        # field_mappings=[FieldMapping(source_field_name="metadata_storage_name", target_field_name="title"),
        #                 # FieldMapping(source_field_name="/document/author", target_field_name="author"), WARNING: THIS DOESN'T WORK. MAP IN PROJECTION.                       
        #                 ],
    )


client.create_or_update_indexer(indexer)
client.reset_indexer(indexer_name)
client.run_indexer(indexer_name)

print("Running indexer")