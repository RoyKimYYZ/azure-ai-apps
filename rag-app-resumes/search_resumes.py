import os # 
from pathlib import Path
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from config import ASSET_PATH, get_logger
from azure.ai.inference.prompts import PromptTemplate
from azure.search.documents.models import VectorizedQuery
from opentelemetry import trace
import json

# import debugpy
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()

# initialize logging and tracing objects
logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

# create a project client using environment variables loaded from the .env file
project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
)

# create a vector embeddings client that will be used to generate vector embeddings
chat = project.inference.get_chat_completions_client()
embeddings = project.inference.get_embeddings_client()

# use the project client to get the default search connection
search_connection = project.connections.get_default(
    connection_type=ConnectionType.AZURE_AI_SEARCH, include_credentials=True
)

# Create a search index client using the search connection
# This client will be used to create and delete search indexes
search_client = SearchClient(
    index_name="resume-index",  #os.environ["AISEARCH_INDEX_NAME"],
    endpoint=search_connection.endpoint_url,
    credential=AzureKeyCredential(key=search_connection.key),
)

@tracer.start_as_current_span(name="search_resumes")
def search_resumes(messages: list, context: dict = None) -> dict:
    if context is None:
        context = {}

    # get the top number of search results to return
    overrides = context.get("overrides", {})
    top = overrides.get("top", 5)

    # generate a search query from the chat messages
    intent_prompty = PromptTemplate.from_prompty(Path(ASSET_PATH) / "intent_mapping.prompty")

    #print(f"intent mapping prompt: \n{intent_prompty}")

    #print(f"messages: {messages}")
    #print(intent_prompty.parameters)
    # generate a search query from the chat messages
    # the intent mapping prompt is a chat prompt that takes the chat messages and generates a search query      
    try:
        intent_mapping_response = chat.complete(
            model=os.environ["INTENT_MAPPING_MODEL"],
            messages=intent_prompty.create_messages(conversation=messages),
            **intent_prompty.parameters,
        )
    # except azure.core.exceptions.HttpResponseError as e:
    #     logger.error(f"HTTP request error during search: {e}")
    #     raise
    except Exception as e:
        logger.error(f"Error during intent mapping: {e}")
        raise

    search_query = intent_mapping_response.choices[0].message.content
    logger.debug(f"🧠 Intent mapping: {search_query}")

    # generate a vector representation of the search query
    embedding = embeddings.embed(model=os.environ["EMBEDDINGS_MODEL"], input=search_query)
    search_vector = embedding.data[0].embedding
    ##print(f"search vector embedding: \n {search_vector}")
    
    # search the index for products matching the search query
    vector_query = VectorizedQuery(vector=search_vector, k_nearest_neighbors=top, fields="text_vector")

    search_results = search_client.search(
        search_text=search_query, vector_queries=[vector_query], select=["id", "firstName", "lastName", "chunk", "resumeContent", "filepath", "title", "url"]
    )

    documents = [
        {
            "id": result["id"],
            "firstName": result["firstName"],
            "lastName": result["lastName"],
            "content": result["chunk"],
            #"filepath": result["filepath"],
            "title": result["title"],
            #"url": result["url"],
        }
        for result in search_results
    ]
    
    # Print the contents of the documents variable
    # print("Retrieved documents:")
    # for doc in documents:
    #     print(doc)

    # add results to the provided context
    if "thoughts" not in context:
        context["thoughts"] = []

    # add thoughts and documents to the context object so it can be returned to the caller
    context["thoughts"].append(
        {
            "title": "Generated search query",
            "description": search_query,
        }
    )

    if "grounding_data" not in context:
        context["grounding_data"] = []
    context["grounding_data"].append(documents)

    logger.debug(f"📄 {len(documents)} documents retrieved: {documents}")
    
    # Pretty print the documents variable to the console
    #print("Retrieved documents (pretty-printed):")
    #print(json.dumps(documents, indent=4, ensure_ascii=False).replace("\\n", "\n"))
    return documents

if __name__ == "__main__":
    import logging
    import argparse
    import json

    # set logging level to debug when running this module directly
    logger.setLevel(logging.DEBUG)

    # load command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        help="Query to use to search product",
        default="I need to hire a software engineer with experience in Python and Azure.",
    )

    args = parser.parse_args()
    query = args.query

    result = search_resumes(messages=[{"role": "user", "content": query}])