import os # 
from pathlib import Path
from pprint import pprint
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from config import ASSET_PATH, get_logger
from azure.ai.inference.prompts import PromptTemplate
from azure.search.documents.models import VectorizedQuery
from opentelemetry import trace
from semantic_kernel.functions import kernel_function
import json
from dotenv import load_dotenv

class AISearchPlugin:
    """
    Description: Search docs in AI Search index.
    """
    logger = get_logger(__name__)
    tracer = trace.get_tracer(__name__)

    # Load environment variables from .env file 
    load_dotenv()
    print(os.environ["AIPROJECT_ENDPOINT"])
    
    # create a project client using environment variables loaded from the .env file
    project = AIProjectClient(
            endpoint=os.environ["AIPROJECT_ENDPOINT"], 
            credential=DefaultAzureCredential()
        )
    
        # create a vector embeddings client that will be used to generate vector embeddings
        #self.chat_client = self.project.inference.get_chat_completions_client()
        # self.embeddings_client = self.project.inference.get_azure_openai_client(
        #     api_version="2024-02-15-preview"
        # )

        # use the project client to get the default search connection
        # self.search_connection = self.project.connections.get_default(
        #     connection_type=ConnectionType.AZURE_AI_SEARCH, include_credentials=True
        # )

        # Create a search index client using the search connection
        # This client will be used to create and delete search indexes
        # self.search_client = SearchClient(
        #     index_name="resume-index",  # os.environ["AISEARCH_INDEX_NAME"],
        #     endpoint=self.search_connection.endpoint_url,
        #     credential=AzureKeyCredential(key=self.search_connection.key),
        #     )

    # @tracer.start_as_current_span(name="search_resumes")
    @staticmethod
    @kernel_function(description="Search docs in AI Search index", name="SearchResumes")
    def search_resumes(kernel, search_query:str, embedding_query, context=None) -> list:
        logger = get_logger(__name__)
        tracer = trace.get_tracer(__name__)

        # Load environment variables from .env file 
        load_dotenv()
        # print(os.environ["AIPROJECT_ENDPOINT"])
        
        # create a project client using environment variables loaded from the .env file
        # project = AIProjectClient(
        #         endpoint=os.environ["AIPROJECT_ENDPOINT"], 
        #         credential=DefaultAzureCredential()
        #     )
        # use the project client to get the default search connection
        # Error: project does not exist
        # search_connection = project.connections.get_default(
        #     connection_type=ConnectionType.AZURE_AI_SEARCH, 
        #     include_credentials=True
        # )
        
        # Create a search index client using the search connection
        # This client will be used to create and delete search indexes
        search_client = SearchClient(
            index_name="resume-index",  #os.environ["AISEARCH_INDEX_NAME"],
            endpoint=os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"],  # search_connection.endpoint_url,
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),  # search_connection.key
        )

        
        if search_query is None:
            messages = []
        if context is None:
            context = {}

        # get the top number of search results to return
        overrides = context.get("overrides", {})
        top = overrides.get("top", 5)

        # logger.debug(f"🧠 Intent mapping: {search_query}")

        # generate a vector representation of the search query
       
        ##print(f"search vector embedding: \n {search_vector}")
        
        # search the index for data or documents matching the search query
        vector_query = VectorizedQuery(vector=embedding_query, k_nearest_neighbors=top, fields="text_vector")

        search_results = search_client.search(
            search_text=search_query, vector_queries=[vector_query], select=["id", "firstName", "lastName", "chunk", "resumeContent", "filepath", "title", "url"]
        )

        documents = [
            {
                "id": result["id"],
                "firstName": result["firstName"],
                "lastName": result["lastName"],
                "content": result["chunk"],
                "filepath": result["filepath"],
                "title": result["title"],
                "url": result["url"],
            }
            for result in search_results
            #if result.get("firstName") is not None and result.get("title") is not None
        ]
        
        # Print the contents of the documents variable
        # print("Retrieved documents:")
        # for doc in documents:
        #     print(pprint(doc, indent=2))

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

    plugin = AISearchPlugin()
    result = plugin.search_resumes(messages=[{"role": "user", "content": query}])
    print(json.dumps(result, indent=4, ensure_ascii=False))