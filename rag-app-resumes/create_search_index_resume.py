## This script creates a search index in Azure AI Search using the Azure SDK for Python.
# Make sure azure ai search connection is up to date in azure foundry project
# - check resumes.cv file in assets folder and is passed in as default or cmd line argument


import os
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from config import get_logger
import pandas as pd
from azure.search.documents.indexes.models import (
    SemanticSearch,
    SearchField,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    HnswParameters,
    VectorSearchAlgorithmMetric,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    SearchIndex,
)


# initialize logging object
logger = get_logger(__name__)

# create a project client using environment variables loaded from the .env file
project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
)

# create a vector embeddings client that will be used to generate vector embeddings
embeddings = project.inference.get_embeddings_client()

# use the project client to get the default search connection
search_connection = project.connections.get_default(
    connection_type=ConnectionType.AZURE_AI_SEARCH, include_credentials=True
)

# Create a search index client using the search connection
# This client will be used to create and delete search indexes
index_client = SearchIndexClient(
    endpoint=search_connection.endpoint_url, credential=AzureKeyCredential(key=search_connection.key)
)


def create_index_definition(index_name: str, model: str) -> SearchIndex:
    dimensions = 1536  # text-embedding-ada-002
    if model == "text-embedding-3-large":
        dimensions = 3072

    # The fields we want to index. The "embedding" field is a vector field that will
    # be used for vector search.
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String), # SimpleField is used to define basic fields in a search index. These fields can store simple data types such as strings, integers, booleans, etc.
        SearchableField(name="title", type=SearchFieldDataType.String, searchable=True, retrievable=True),
        SearchableField(name="firstName", type=SearchFieldDataType.String, searchable=True, retrievable=True),
        SearchableField(name="lastName", type=SearchFieldDataType.String, searchable=True, retrievable=True),
        SearchableField(name="profession", type=SearchFieldDataType.String, searchable=True, retrievable=True, filterable=True),
        SearchableField(name="parent_id", type=SearchFieldDataType.String, searchable=True, retrievable=True, filterable=True),
        SearchableField(name="resume", type=SearchFieldDataType.String),
        SimpleField(name="filepath", type=SearchFieldDataType.String,),
        SimpleField(name="url", type=SearchFieldDataType.String),
        SearchField(
            name="address",
            type=SearchFieldDataType.ComplexType,
            fields=[
            SearchableField(name="address", type=SearchFieldDataType.String, searchable=True, retrievable=True),
            SearchableField(name="city", type=SearchFieldDataType.String, searchable=True, retrievable=True),
            SimpleField(name="postalCode", type=SearchFieldDataType.String, retrievable=True),
            SearchableField(name="province", type=SearchFieldDataType.String, searchable=True, retrievable=True),
            ],
        ),
        SimpleField(name="content", type=SearchFieldDataType.String, retrievable=True),
        SimpleField(name="resumeContent", type=SearchFieldDataType.String, retrievable=True),
        SearchField(
            name="resumeVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            # Size of the vector created by the text-embedding-ada-002 model.
            vector_search_dimensions=dimensions,
            vector_search_profile_name="myHnswProfile",
        ),
        SimpleField(name="metadata_storage_name", type=SearchFieldDataType.String, retrievable=True),
        SimpleField(name="metadata_storage_path", type=SearchFieldDataType.String, retrievable=True),
        SimpleField(name="metadata_storage_size", type=SearchFieldDataType.Int64, retrievable=True),
        SimpleField(name="metadata_storage_last_modified", type=SearchFieldDataType.DateTimeOffset, retrievable=True),
        SimpleField(name="last_modified", type=SearchFieldDataType.DateTimeOffset, retrievable=True),
        SimpleField(name="document_type", type=SearchFieldDataType.String, retrievable=True),
        SimpleField(name="author", type=SearchFieldDataType.String, retrievable=True),
        SimpleField(name="locations", type=SearchFieldDataType.String, retrievable=True),
        SearchField(name="chunk_id",type=SearchFieldDataType.String,
                key=True,
                hidden=False,
                filterable=True,
                sortable=True,
                facetable=False,
                searchable=True,
                analyzer_name="keyword"
            ),
            SearchField(name="chunk", type=SearchFieldDataType.String, hidden=False,
                filterable=False, sortable=False, facetable=False, searchable=True
            ),
            SearchField(
                name="text_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                hidden=False,
                filterable=False,
                sortable=False,
                facetable=False,
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile="myHnswProfile",
                vector_search_profile_name="myHnswProfile",
                
            )
    ]

    # The "content" field should be prioritized for semantic ranking.
    # Semantic Configuration is used to define the fields that should be prioritized for semantic ranking.
    semantic_config = SemanticConfiguration(
        name="default",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="firstName"),
            keywords_fields=[],
            content_fields=[SemanticField(field_name="resume")],
        ),
    )

    # For vector search, we want to use the HNSW (Hierarchical Navigable Small World)
    # algorithm (a type of approximate nearest neighbor search algorithm) with cosine
    # distance.
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4,
                    ef_construction=1000,
                    ef_search=1000,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            ),
            ExhaustiveKnnAlgorithmConfiguration(
                name="myExhaustiveKnn",
                kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                parameters=ExhaustiveKnnParameters(metric=VectorSearchAlgorithmMetric.COSINE),
            ),
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
                vectorizer_name="myOpenAI",  # This is the name of the vectorizer we created earlier
            ),
            VectorSearchProfile(
                name="myExhaustiveKnnProfile",
                algorithm_configuration_name="myExhaustiveKnn",
            ),
        ],
        ## Vectorizers are used to convert text into vector embeddings that can be used for vector search.       
        # Vectorizers are used at query time, but specified in index definitions, and referenced on vector fields through a vector profile. The Azure OpenAI vectorizer is called AzureOpenAIVectorizer in the API.
        vectorizers=[  
            AzureOpenAIVectorizer(  
                vectorizer_name="myOpenAI",  
                kind="azureOpenAI",   ## The kind of vectorizer to use. This should be "azureOpenAI" for Azure OpenAI vectorizers.
                parameters=AzureOpenAIVectorizerParameters(  
                    resource_url=os.environ.get("AZURE_OPENAI_ENDPOINT", "https://<your-openai-resource>.openai.azure.com"),  # TODO - update with your resource URL
                    deployment_name="text-embedding-ada-002",
                    model_name="text-embedding-ada-002",
                    api_key=os.environ.get("AZURE_OPENAI_KEY", "<YOUR_AZURE_OPENAI_KEY>"), #TODO
                ),
            ),  
        ],  
    )

    # Create the semantic settings with the configuration
    # semantic_search = SemanticSearch(configurations=[semantic_config])

    # Create the search index definition
    return SearchIndex(
        name=index_name,
        fields=fields,
        #semantic_search=semantic_search, # turn off semantic ranker and use vectorizer
        vector_search=vector_search,
    )
    
# define a function for indexing a csv file, that adds each row as a document
# and generates vector embeddings for the specified content_column
def create_docs_from_csv(path: str, content_column: str, model: str) -> list[dict[str, any]]:
    products = pd.read_csv(path)
    items = []
    for product in products.to_dict("records"):
        resume = product[content_column]
        id = str(product["id"])
        firstName = product["firstName"]
        lastName = product["lastName"]
        profession = product["profession"]
        addressCity = product["city"]
        
        url = f"/products/{firstName.lower().replace(' ', '-')}"
        emb = embeddings.embed(input=resume, model=model)
        rec = {
            "id": id,
            "resume": resume,
            "filepath": f"{firstName.lower().replace(' ', '-')}",
            "firstName": firstName,
            "lastName": lastName,
            "profession": profession,
            "address": {
                "city": addressCity
            },
            "resumeContent": resume,
            "url": url,
            "resumeVector": emb.data[0].embedding,
            'chunk_id': id
        }
        items.append(rec)

    return items


def create_index_from_csv(index_name, csv_file):
    # If a search index already exists, delete it:
    try:
        index_definition = index_client.get_index(index_name)
        index_client.delete_index(index_name)
        logger.info(f"🗑️  Found existing index named '{index_name}', and deleted it")
    except Exception:
        pass

    # create an empty search index
    index_definition = create_index_definition(index_name, model=os.environ["EMBEDDINGS_MODEL"])
    index_client.create_index(index_definition)

    # create documents from the products.csv file, generating vector embeddings for the "description" column
    docs = create_docs_from_csv(path=csv_file, content_column="resume", model=os.environ["EMBEDDINGS_MODEL"])

    # Add the documents to the index using the Azure AI Search client
    search_client = SearchClient(
        endpoint=search_connection.endpoint_url,
        index_name=index_name,
        credential=AzureKeyCredential(key=search_connection.key),
    )
    
    # comment out for future use
    # rec = {
    #         "id": id,
    #         "content": content,
    #         "filepath": f"{title.lower().replace(' ', '-')}",
    #         "title": title,
    #         "url": url,
    #         "contentVector": emb.data[0].embedding,
    #     }

    search_client.upload_documents(docs)
    logger.info(f"➕ Uploaded {len(docs)} documents to '{index_name}' index")
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-name",
        type=str,
        help="index name to use when creating the AI Search index",
        default=os.environ["AISEARCH_INDEX_RESUME"],
    )
    parser.add_argument(
        "--csv-file", type=str, help="path to data for creating search index", default="assets/resumes.csv"
    )
    
    
    print(os.environ["AIPROJECT_CONNECTION_STRING"])
    
    args = parser.parse_args()
    index_name = args.index_name
    csv_file = args.csv_file

    create_index_from_csv(index_name, csv_file)