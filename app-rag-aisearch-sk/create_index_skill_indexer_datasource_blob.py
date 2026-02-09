import os # 4/6 - not used. 
# Purpose: Create an 
#   Azure Search index, 
#   data source, 
#   skillset, and 
#   indexer for a blob storage container.

from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataSourceConnection,
    SearchIndexerDataContainer,
    SearchField,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    AzureOpenAIVectorizerParameters,
    AzureOpenAIEmbeddingSkill,
    SplitSkill,
    FieldMapping,
    IndexingParameters,
    IndexingParametersConfiguration,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    SearchIndexerSkillset
)

## CONFIGURATION
# azure openai
azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://<your-openai-resource>.openai.azure.com")
azure_openai_key = os.environ.get("AZURE_OPENAI_KEY", "<YOUR_AZURE_OPENAI_KEY>")
# azure search
search_endpoint = "https://rk-ai-search.search.windows.net"
# index
index_name = "real-estate-index"
# vectorizer deployment name
vectorizer_deployment_name = "text-embedding-ada-002"
# blob storage
blob_container = "resume-samples"   



# Create the search index
search_index = SearchIndex(
        name=index_name,
        fields=[
            SearchField(
                name="chunk_id",
                type=SearchFieldDataType.String,
                key=True,
                hidden=False,
                filterable=True,
                sortable=True,
                facetable=False,
                searchable=True,
                analyzer_name="keyword"
            ),
            SearchField(
                name="parent_id",
                type=SearchFieldDataType.String,
                hidden=False,
                filterable=True,
                sortable=True,
                facetable=False,
                searchable=True
            ),
            SearchField(
                name="chunk",
                type=SearchFieldDataType.String,
                hidden=False,
                filterable=False,
                sortable=False,
                facetable=False,
                searchable=True
            ),
            SearchField(
                name="title",
                type=SearchFieldDataType.String,
                hidden=False,
                filterable=False,
                sortable=False,
                facetable=False,
                searchable=True
            ),
            SearchField(
                name="chunk_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                hidden=False,
                filterable=False,
                sortable=False,
                facetable=False,
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="myHnswProfile"
            ),
            SimpleField(name="filepath", type=SearchFieldDataType.String),
            SimpleField(name="url", type=SearchFieldDataType.String),
            SearchField(
                name="address",
                type=SearchFieldDataType.ComplexType,
                fields=[
                SearchableField(name="address", type=SearchFieldDataType.String, searchable=True, retrievable=True),
                SearchableField(name="city", type=SearchFieldDataType.String, searchable=True, retrievable=True),
                SimpleField(name="postalCode", type=SearchFieldDataType.String, retrievable=True),
                SearchableField(name="province", type=SearchFieldDataType.String, searchable=True, retrievable=True)
                ]
            ),
            SimpleField(name="metadata_storage_name", type=SearchFieldDataType.String, retrievable=True),
            SimpleField(name="metadata_storage_path", type=SearchFieldDataType.String, retrievable=True),
            SimpleField(name="metadata_storage_size", type=SearchFieldDataType.Int64, retrievable=True),
            SimpleField(name="metadata_storage_last_modified", type=SearchFieldDataType.DateTimeOffset, retrievable=True),
            SimpleField(name="last_modified", type=SearchFieldDataType.DateTimeOffset, retrievable=True),
            SimpleField(name="document_type", type=SearchFieldDataType.String, retrievable=True),
            SimpleField(name="year", type=SearchFieldDataType.String, retrievable=True),
            SimpleField(name="locations", type=SearchFieldDataType.String, retrievable=True)
        ],
        vector_search=VectorSearch(
            profiles=[
                VectorSearchProfile(  
                    name="myHnswProfile",  
                    algorithm_configuration_name="myHnsw", 
                    vectorizer_name="myOpenAI",  
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(name="myHnsw"), # name must match the name in the profile
            ],
            vectorizers=[
                AzureOpenAIVectorizer(  
                    vectorizer_name="myOpenAI",  
                    kind="azureOpenAI",  
                    parameters=AzureOpenAIVectorizerParameters(  
                        resource_url=azure_openai_endpoint, 
                        api_key=azure_openai_key, 
                        deployment_name="text-embedding-ada-002",
                        model_name="text-embedding-ada-002"
                    ),
                ),  
            ]
        )
    )

# Create the skillset
def create_search_skillset(skillset_name, cognitive_services_key):
    return SearchIndexerSkillset(
        name=skillset_name,
        description="Skillset to map filename to title",
        skills=[],
        cognitive_services_account=CognitiveServicesAccountKey(key=cognitive_services_key),
        field_mappings=[
            {
                "sourceFieldName": "metadata_storage_name",
                "targetFieldName": "title"
            }
        ]
    )

# Create the indexer
def create_search_indexer(indexer_name, index_name, datasource_name, skillset_name):
    return SearchIndexer(
        name=indexer_name,
        data_source_name=datasource_name,
        target_index_name=index_name,
        skillset_name=skillset_name
    )

# Main code
search_index_client = SearchIndexClient(endpoint=search_endpoint, credential=search_credential)
search_indexer_client = SearchIndexerClient(endpoint=search_endpoint, credential=search_credential)

# Create and upload the index
index = create_search_index(search_index)
search_index_client.create_or_update_index(index)

# Create and upload the data source
data_source = create_search_datasource(search_datasource, blob_connection_string, blob_container)
search_indexer_client.create_or_update_data_source_connection(data_source)

# Create and upload the skillset
skillset = create_search_skillset(search_skillset, azure_openai_key)
search_indexer_client.create_or_update_skillset(skillset)

# Create and upload the indexer
indexer = create_search_indexer(search_indexer, search_index, search_datasource, search_skillset)
search_indexer_client.create_or_update_indexer(indexer)

# Run the indexer
search_indexer_client.run_indexer(search_indexer)
print("Indexer is running...")