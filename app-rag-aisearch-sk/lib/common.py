from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataSourceConnection,
    SearchIndexerDataContainer,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    #HnswVectorSearchAlgorithmConfiguration, # depends on azure-search-documents==11.4.0b11
    HnswAlgorithmConfiguration,
    AzureOpenAIVectorizerParameters,
    # AzureOpenAIParameters,
    AzureOpenAIEmbeddingSkill,
    SplitSkill,
    FieldMapping,
    IndexingParameters,
    IndexingParametersConfiguration,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    #HnswParameters,
    #VectorSearchAlgorithmConfiguration
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    SearchIndexerSkillset
)
# Required to use the preview SDK
# from azure.search.documents.indexes._generated.models import (
#     SearchIndexerSkillset,
#     AzureOpenAIVectorizer,
#     # AzureOpenAIParameters,
#     SearchIndexerIndexProjections,
#     SearchIndexerIndexProjectionSelector,
#     SearchIndexerIndexProjectionsParameters,
#     InputFieldMappingEntry,
#     OutputFieldMappingEntry
# )
import tiktoken
import matplotlib.pyplot as plt
import math
import numpy as np

def create_search_index(index_name, azure_openai_endpoint, azure_openai_embedding_deployment_id, azure_openai_key=None):
    return SearchIndex(
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
                name="vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                hidden=False,
                filterable=False,
                sortable=False,
                facetable=False,
                searchable=True,
                vector_search_dimensions=1536,
                #vector_search_profile="profile",
                vector_search_profile_name="myHnswProfile"
            )
        ],
        vector_search=VectorSearch(
            profiles=[
                # VectorSearchProfile( # from original code and older version
                #     name="profile",
                #     #algorithm="hnsw-algorithm",
                #     algorithm_configuration_name="myHnsw", 
                #     vectorizer_name="azure-openai-vectorizer",
                # ),
                VectorSearchProfile(  
                    name="myHnswProfile",  
                    algorithm_configuration_name="myHnsw", 
                    vectorizer_name="myOpenAI",  
                )
            ],
            algorithms=[
                #HnswVectorSearchAlgorithmConfiguration(name="hnsw-algorithm") # depends on azure-search-documents==11.4.0b11 
                HnswAlgorithmConfiguration(name="myHnsw"), # name must match the name in the profile
            ],
            vectorizers=[
                # AzureOpenAIVectorizer(
                #         name="azure-openai-vectorizer",
                #         azure_open_ai_parameters=AzureOpenAIParameters(
                #             resource_uri=azure_openai_endpoint,
                #             deployment_id=azure_openai_embedding_deployment_id,
                #             api_key=azure_openai_key # Optional if using RBAC authentication
                #         )
                #     )
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

def create_search_datasource(datasource_name, connection_string, container_name):
    return SearchIndexerDataSourceConnection(
        name=datasource_name,
        type="azureblob",
        connection_string=connection_string,
        container=SearchIndexerDataContainer(
            name=container_name
        )
    )

def create_search_skillset(
        skillset_name,
        index_name,
        azure_openai_endpoint,
        azure_openai_embedding_deployment_id,
        azure_openai_key=None,
        text_split_mode='pages',
        maximum_page_length=2000,
        page_overlap_length=500):
    return SearchIndexerSkillset(
        name=skillset_name,
        skills=[
            SplitSkill(
                name="Text Splitter",
                default_language_code="en",
                text_split_mode=text_split_mode,
                maximum_page_length=maximum_page_length,
                page_overlap_length=page_overlap_length,
                context="/document",
                inputs=[
                    InputFieldMappingEntry(
                        name="text",
                        source="/document/content"
                    )
                ],
                outputs=[
                    OutputFieldMappingEntry(
                        name="textItems",
                        target_name="pages"
                    )
                ]
            ),
            AzureOpenAIEmbeddingSkill(
                name="Embeddings",
                resource_url=azure_openai_endpoint,
                deployment_name=azure_openai_embedding_deployment_id,
                
                api_key=azure_openai_key, # Optional if using RBAC authentication
                context="/document/pages/*",
                inputs=[
                    InputFieldMappingEntry(
                        name="text",
                        source="/document/pages/*"
                    )
                ],
                outputs=[
                    OutputFieldMappingEntry(
                        name="embedding",
                        target_name="vector"
                    )
                ]
            )
        ],
        index_projection=SearchIndexerIndexProjection(
            selectors=[
                SearchIndexerIndexProjectionSelector(
                    target_index_name=index_name,
                    parent_key_field_name="parent_id",
                    source_context="/document/pages/*",
                    mappings=[
                        InputFieldMappingEntry(
                            name="chunk",
                            source="/document/pages/*"
                        ),
                        InputFieldMappingEntry(
                            name="vector",
                            source="/document/pages/*/vector"
                        ),
                        InputFieldMappingEntry(
                            name="title",
                            source="/document/metadata_storage_name"
                        ),
                        # InputFieldMappingEntry(name="document_type", 
                        #                        source="/document/metadata_storage_properties/document_type")  # Custom metadata key
                    ]
                )
            ],
            parameters=SearchIndexerIndexProjectionsParameters(projection_mode="skipIndexingParentDocuments")
        )
    )

def create_search_indexer(
    indexer_name,
    skillset_name,
    datasource_name,
    index_name):
    return SearchIndexer(
        name=indexer_name,
        data_source_name=datasource_name,
        target_index_name=index_name,
        skillset_name=skillset_name,
        # parameters = IndexingParameters( # https://medium.com/@tanishk_rane/automating-azure-ai-search-workflows-with-python-342644c639a6
        #     configuration=IndexingParametersConfiguration(
        #         data_to_extract="contentAndMetadata",
        #         parsing_mode="default",
        #         image_action="generateNormalizedImages",
        #         allow_skillset_to_read_file_data=True,
        #         query_timeout=None
        #     ),
        #     max_failed_items=-1,
        #     max_failed_items_per_batch=-1,
        #     batch_size=1
        # ),
        field_mappings=[FieldMapping(source_field_name="metadata_storage_name", target_field_name="title")],
        # field_mappings=[
        #     FieldMapping( # https://github.com/Azure/azure-sdk-for-python/issues/34529
        #         source_field_name="metadata_storage_name", 
        #         target_field_name="title"   
        #     )
        # ]
    )

def get_chunks(search_client):
    results = search_client.search(search_text="*", top=100000, select="chunk_id,chunk")
    chunks = {}
    for result in results:
        id = int(result["chunk_id"].split("_")[3])
        chunks[id] = result["chunk"]
    return [chunks[id] for id in sorted(chunks.keys())]

def get_encoding_name(model="gpt-3.5-turbo"):
    return tiktoken.encoding_for_model(model).name

def get_token_length(text, model="gpt-3.5-turbo"):
    return len(tiktoken.encoding_for_model(model).encode(text))

def plot_chunk_histogram(chunks, length_fn, title, xlabel, ylabel="Chunk Count"):
    def round_to_lowest_multiple(number, multiple):
        return (number // multiple) * multiple

    def round_to_highest_multiple(number, multiple):
        return math.ceil(number / multiple) * multiple

    ys = [length_fn(chunk) for chunk in chunks]
    min_y = min(ys)
    max_y = max(ys)
    bins=25
    n, _, _ = plt.hist(ys, edgecolor="black", bins=bins) 
    # Set y-axis limits to remove the gap at the top
    max_freq = max(n)
    plt.ylim(0, max_freq)

    # Spacing for ticks on x-axis and x-axis limits to remove gaps
    tick_step = max(int(round_to_lowest_multiple((max_y-min_y)/5, 100)), 100)
    max_xtick = round_to_highest_multiple(max_y, tick_step)
    xticks = list(np.arange(start=round_to_lowest_multiple(min_y, tick_step), stop=round_to_highest_multiple(max_xtick, tick_step), step=tick_step))
    if max_xtick and xticks[-1] != max_xtick:
        xticks.append(max_xtick)
    plt.xticks(xticks)
    plt.xlim(round_to_lowest_multiple(min_y, tick_step), max_xtick)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()