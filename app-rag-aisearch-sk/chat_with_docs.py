import os
import json
from datetime import datetime
from semantic_kernel import Kernel, __version__
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import kernel_function, KernelArguments
# Memory, AI Search
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.memory.volatile_memory_store import VolatileMemoryStore
from semantic_kernel.functions import KernelFunction
from semantic_kernel.prompt_template import PromptTemplateConfig
# Vector Store
from collections.abc import Awaitable, Callable
from typing import Any
from semantic_kernel.agents import AgentThread, ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.azure_ai_search import AzureAISearchCollection
from semantic_kernel.filters import FilterTypes, FunctionInvocationContext
from semantic_kernel.functions import KernelParameterMetadata, KernelPlugin
from semantic_kernel.kernel_types import OptionalOneOrList
from azure.ai.inference.prompts import PromptTemplate
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents import ChatHistory

from resume_datamodel import ResumeModel
from search_docs import AISearchPlugin
import asyncio
import pprint
import logging
    
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
for handler in [logging.FileHandler("chat_with_docs.log", "w"), logging.StreamHandler()]:
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
__version__

#load .env file
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  

# Load Azure OpenAI configuration from environment
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
EMBEDDINGS_DEPLOYMENT = os.environ.get("EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002")
AIPROJECT_CONNECTION_STRING = os.environ.get("AIPROJECT_CONNECTION_STRING")
AISEARCH_INDEX_RESUME = os.environ.get("AISEARCH_INDEX_RESUME")
AZURE_SEARCH_ADMIN_KEY = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_SERVICE_ENDPOINT = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")

# Initialize Semantic Kernel
kernel = Kernel()

kernel.remove_all_services()

service_id = None

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

service_id = "default"
kernel.add_service(
    AzureChatCompletion(
        service_id=service_id,
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
    ),
)

# Register Azure OpenAI text completion service
chat_service = AzureChatCompletion(
    service_id="azure-chat",
    endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    deployment_name=AZURE_OPENAI_DEPLOYMENT
)

kernel.add_service(chat_service)

kernel.add_service(
    AzureTextEmbedding(
        service_id="embedding",
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        deployment_name=EMBEDDINGS_DEPLOYMENT
    )
)

# # Add embedding generation service
# kernel.add_text_embedding_generation_service(
#     service_id="embedding",
#     service= OpenAITextEmbedding(
#         model_id="text-embedding-ada-002",
#         api_key="YOUR_OPENAI_KEY"
#     )
# )

plugins_directory = "./plugins/"
resumeSearchPlugin = kernel.add_plugin(parent_directory=plugins_directory, plugin_name="ResumeSearchPlugin")

aintentMapping_PromptTemplate = resumeSearchPlugin["IntentMapping_PromptTemplate"]

aisearch_query_docs_plugin = kernel.add_plugin(AISearchPlugin(), "AISearchPlugin")

memory = SemanticTextMemory(storage=VolatileMemoryStore(), embeddings_generator=kernel.get_service("embedding"))

# deprecation warning/error
# kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPlugin")

from semantic_kernel.connectors.azure_ai_search import AzureAISearchStore
from azure.search.documents.indexes import SearchIndexClient
from semantic_kernel.connectors.azure_ai_search import AzureAISearchStore
from azure.core.credentials import AzureKeyCredential

vector_store = AzureAISearchStore(
    search_endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    api_key=AZURE_SEARCH_ADMIN_KEY,
    embedding_generator=AzureTextEmbedding(
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        deployment_name=EMBEDDINGS_DEPLOYMENT
    )
)
# Todo: how to query with vector_store

search_index_client = SearchIndexClient(endpoint=AZURE_SEARCH_SERVICE_ENDPOINT, 
                                        credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY))


# validation error
# collection = AzureAISearchCollection[str, ResumeModel](
#     record_type=ResumeModel,
#     search_index_client=search_index_client,
#     search_endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
#     api_key=AZURE_SEARCH_ADMIN_KEY,
#     index_name="resume-index"
# )

# search_plugin = KernelPlugin(
#     name="azure_ai_search",
#     description="A plugin that allows you to search for resume docs in Azure AI Search.",
#     functions=[
#         collection.create_search_function(
#             # this create search method uses the `search` method of the text search object.
#             # remember that the text_search object for this sample is based on
#             # the text_search method of the Azure AI Search.
#             # but it can also be used with the other vector search methods.
#             # This method's description, name and parameters are what will be serialized as part of the tool
#             # call functionality of the LLM.
#             # And crafting these should be part of the prompt design process.
#             # The default parameters are `query`, `top`, and `skip`, but you specify your own.
#             # The default parameters match the parameters of the VectorSearchOptions class.
#             description="A resume search engine, allows searching for hotels in specific cities, "
#             "you do not have to specify that you are searching for hotels, for all, use `*`.",
#             search_type="keyword_hybrid",
#             # Next to the dynamic filters based on parameters, I can specify options that are always used.
#             # this can include the `top` and `skip` parameters, but also filters that are always applied.
#             # In this case, I am filtering by country, so only hotels in the USA are returned.
#             filter=None,
#             parameters=[
#                 KernelParameterMetadata(
#                     name="query",
#                     description="What to search for.",
#                     type="str",
#                     is_required=True,
#                     type_object=str,
#                 ),
#                 KernelParameterMetadata(
#                     name="content",
#                     description=f"The skills and experience that you want to search for a resume in",
#                     type="str",
#                     type_object=str,
#                 ),
#                 KernelParameterMetadata(
#                     name="top",
#                     description="Number of results to return.",
#                     type="int",
#                     default_value=5,
#                     type_object=int,
#                 ),
#             ],
#             # and here the above created function is passed in.
#             # filter_update_function=filter_update,
#             # finally, we specify the `string_mapper` function that is used to convert the record to a string.
#             # This is used to make sure the relevant information from the record is passed to the LLM.
#             string_mapper=lambda x: f"(id :{x.record.id}) {x.record.firstName} {x.record.LastName} - {x.record.title} {x.record.content}. ",  # noqa: E501
#         )
#     ],
# )

#kernel.add_plugin(search_plugin, "ResumeSearchPlugin")

# travel_agent = ChatCompletionAgent(
#     name="TravelAgent",
#     description="A travel agent that helps you find a hotel.",
#     service=AzureChatCompletion(),
#     instructions="What are the resumes with cloud devops experience?",
#     function_choice_behavior=FunctionChoiceBehavior.Auto(),
#     plugins=[search_plugin],
# )

# maybe deprecated
# from semantic_kernel.connectors.memory.azure_cognitive_search import AzureCognitiveSearchMemoryStore

# acs_memory_store = AzureCognitiveSearchMemoryStore(vector_size=1536, 
#                                                     search_endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
#                                                     admin_key=AZURE_SEARCH_ADMIN_KEY 
#                                                     )

# memory = SemanticTextMemory(storage=acs_memory_store, embeddings_generator=embedding_gen)
# kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPluginACS")

async def intent_mapping_search_query(messages: list):
    chat_history = ChatHistory()
    intent_prompty = PromptTemplate.from_prompty("assets/intent_mapping.prompty")
    intent_messages = intent_prompty.create_messages(conversation=messages)
    # Add intent_prompty messages to the chat history
    for message in intent_messages:
        if message.get("role") == "system":
            chat_history.add_system_message(message.get("content", ""))
        elif message.get("role") == "user":
            chat_history.add_user_message(message.get("content", ""))
        elif message.get("role") == "assistant":
            chat_history.add_assistant_message(message.get("content", ""))
    
    # Get the chat completion service
    chat_completion_service = kernel.get_service("default")
    execution_settings = kernel.get_prompt_execution_settings_from_service_id("default")
    # Get the response from the AI model
    response = await chat_completion_service.get_chat_message_content(chat_history,
                                                                      settings=execution_settings)
    
    return response  # Return the content of the response
    # Output the response


# do a grounded chat call using the search results
async def setup_chat_with_memory(
        kernel: Kernel,
        service_id: str,
    ) -> KernelFunction:
    """Setup a chat function with memory and semantic search capabilities."""
    prompt = """
    ChatBot can have a conversation with you about any topic.
    It can give explicit instructions or say 'I don't know' if
    it does not have an answer.

    Information about me, from previous conversations:
    - {{recall 'budget by year'}} What is my budget for 2024?
    - {{recall 'savings from previous year'}} What are my savings from 2023?
    - {{recall 'investments'}} What are my investments?

    {{$request}}
    """.strip()

    prompt_template_config = PromptTemplateConfig(
        template=prompt,
        execution_settings={
            service_id: kernel.get_service(service_id).get_prompt_execution_settings_class()(service_id=service_id)
        },
    )

    return kernel.add_function(
            function_name="chat_with_memory",
            plugin_name="chat",
            prompt_template_config=prompt_template_config,
        )


# Parameter messages: e.g. [{"role": "user", "content": "I need to hire a new data scientist, what would you recommend?"}
# @tracer.start_as_current_span(name="chat_with_docs")
async def chat_with_docs(messages: list, context: dict = None) -> dict:
    logger.info("Starting chat_with_docs function...")
    chat_history = ChatHistory()
    
    # Extract the user query from the messages
    search_query = None
    for message in messages:
        if message.get("role") == "user":
            search_query = message.get("content")
            break
    
    if not search_query:
        logger.error("No user query found in messages")
        return {"error": "No user query found in messages"}
    
    logger.info(f"Processing search query: {search_query}")
    
    # todo: use chat_history to pass messages
    result = await intent_mapping_search_query(messages)
    logger.info("Intent Mapping Response (formatted):\n%s", pprint.pformat(result.__dict__, indent=2, width=120))
    search_query = result.content
    logger.info("AI intent search query: %s", search_query)

    #################
    embedding_service = kernel.get_service(service_id="embedding")
    from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatPromptExecutionSettings
    execution_settings = AzureAIInferenceChatPromptExecutionSettings()
    
    from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceTextEmbedding

    embedding_model = AzureAIInferenceTextEmbedding(
        api_key=AZURE_OPENAI_KEY,
        endpoint=AZURE_OPENAI_ENDPOINT,
        ai_model_id=EMBEDDINGS_DEPLOYMENT,
        
    )

    #embedding = await embedding_model.generate_embeddings([search_query_intent])
    #embedding_vector = embedding[0].embedding
    #print(f"Generated embedding vector with {len(embedding_vector)} dimensions")
    # azure.core.exceptions.ResourceNotFoundError: (404) Resource not found
    #print(f"Embedding for query '{search_query_intent}': {embedding}")
    ###############
    
    ###############
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key = AZURE_OPENAI_KEY,  
        api_version = "2023-05-15",
        azure_endpoint = AZURE_OPENAI_ENDPOINT
    )

    response = client.embeddings.create(
        input = search_query,
        model= EMBEDDINGS_DEPLOYMENT
    )
    embedding_query = response.data[0].embedding
    
    #################
    
    # search docs in AI Search index with AI Search Plugin
    aisearch_query_docs_function = aisearch_query_docs_plugin["SearchResumes"]
    document_results = await aisearch_query_docs_function(kernel,
        search_query=search_query, 
        embedding_query=embedding_query,
        context={"overrides": {"top": 5}}
    )
    
    # print("Search Results:")
    logger.info(f"Search Results: {len(document_results.value)} documents found")
    for doc in document_results.value:
        logger.info("Document:\n%s", pprint.pformat(doc.__dict__ if hasattr(doc, "__dict__") else doc, indent=2, width=120))
        #logger.info("Search Result Document:\n%s", pprint.pformat(doc, indent=2))

    # do a grounded chat call using the search results
    logger.info("Creating grounded chat prompt...")
    from azure.ai.inference.prompts import PromptTemplate
    context = {}
    if context is None:
        context = {}
    grounded_chat_prompt = PromptTemplate.from_prompty("plugins/ResumeSearchPlugin/Grounded_Chat/grounded_chat.prompty")
    system_message = grounded_chat_prompt.create_messages(documents=document_results.value, context=context)
    messages = [{"role": "user", "content": search_query}]
    logger.info("System Message (Grounded Chat Prompt):\n%s", json.dumps(system_message, indent=2, ensure_ascii=False))
    #logger.info(f" system_message: \n  {system_message}")
    logger.info(f" messages: \n {json.dumps(messages, indent=2, ensure_ascii=False)}")
    
    for message in list(messages) + list(system_message):
        if message.get("role") == "system":
            chat_history.add_system_message(message.get("content", ""))
        elif message.get("role") == "user":
            chat_history.add_user_message(message.get("content", ""))
        elif message.get("role") == "assistant":
            chat_history.add_assistant_message(message.get("content", ""))

    

    logger.info(f"Parsed search_query: {search_query}")
    chat_completion_service = kernel.get_service("default")
    execution_settings = kernel.get_prompt_execution_settings_from_service_id("default")
    # Get the response from the AI model
    response = await chat_completion_service.get_chat_message_content(chat_history,
                                                                      settings=execution_settings)
    logger.info("Recommended Resumes:\n%s", response)
    
    # Return the structured response
    return {
        "content": response.content,
        "search_query": search_query,
        "documents_found": len(document_results.value),
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    }
    
    # Initialize the Semantic Memory
    # disable
    # collection_id = "generic"
    # async def populate_memory(memory: SemanticTextMemory) -> None:
    #     await memory.save_information(collection=collection_id, id="info1", text="Your budget for 2024 is $100,000")
    #     await memory.save_information(collection=collection_id, id="info2", text="Your savings from 2023 are $50,000")
    #     await memory.save_information(collection=collection_id, id="info3", text="Your investments are $80,000")
    #await populate_memory(memory)
    # chat_func = await setup_chat_with_memory(kernel, service_id)


if __name__ == "__main__":
    import asyncio

    asyncio.run(chat_with_docs(messages=[{"role": "user", "content": "Resumes with cloud devops engineering and azure"}]))