import streamlit as st
import asyncio
import os
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.memory.azure_cognitive_search import AzureCognitiveSearchMemoryStore
from semantic_kernel.memory import SemanticTextMemory
from azure.ai.inference.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
#from prompty import Prompt
# pip install azure-ai-projects~=1.0.0b7
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.services.azure_text_embedding import AzureTextEmbedding
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai.services.open_ai_text_embedding import OpenAITextEmbedding
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin
from semantic_kernel.kernel import Kernel
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.memory.volatile_memory_store import VolatileMemoryStore
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings,
)

# load environment variables
load_dotenv()
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]  
azure_openai_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]  
search_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]  
search_key = os.environ["AZURE_SEARCH_ADMIN_KEY"]  
azure_openai_key = os.environ["AZURE_OPENAI_KEY"] if len(os.environ["AZURE_OPENAI_KEY"]) > 0 else None
search_index = os.environ["AZURE_SEARCH_INDEX"]
search_credential = AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]) if len(os.environ["AZURE_SEARCH_ADMIN_KEY"]) > 0 else DefaultAzureCredential()


# === Azure Agent AI Setup ===
@st.cache_resource
def setup_agent():
    project_client = AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(),
        conn_str="eastus.api.azureml.ms;f1a72634-3a90-4cdb-a6d6-069cf5115068;openai;project-rkopenai"
    )
    agent = project_client.agents.get_agent("asst_I0EgCJ2MP4OiDoJbtTm8mvwZ")
    thread = project_client.agents.get_thread("thread_aVF7IGNC1jBx7tDJ54El6HUM")
    return project_client, agent, thread

project_client, agent, thread = setup_agent()

# Load Prompty prompt templates
#intent_template = Prompt.from_file("prompts/intent_mapping.prompty")
intent_prompty = PromptTemplate.from_prompty("prompts/intent_mapping.prompty")
#search_template = Prompt.from_file("prompts/search_prompt.prompty")
search_template = PromptTemplate.from_prompty("prompts/search_prompt.prompty")
#response_template = Prompt.from_file("prompts/response_template.prompty")
response_template = PromptTemplate.from_prompty("prompts/response_prompt.prompty")

# Cache Kernel initialization
kernel = Kernel()

chat_completion = AzureChatCompletion(
            service_id="chat_completion",
            deployment_name="gpt-4o",
            endpoint=azure_openai_endpoint,
            api_key=azure_openai_key
        )

kernel.add_service(chat_completion)

embedding_gen = AzureTextEmbedding(
    service_id="embedding"
)

kernel.add_service(embedding_gen)

memory_store = AzureCognitiveSearchMemoryStore(
    search_endpoint=search_endpoint,
    admin_key=search_key,
    
    vector_size=1536
)

text_memory = SemanticTextMemory(storage=memory_store, embeddings_generator=embedding_gen)
kernel.add_plugin(TextMemoryPlugin(text_memory), "TextMemoryPlugin")

    

 # Enable planning
execution_settings = AzureChatPromptExecutionSettings()
execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
# Create a history of the conversation
history = ChatHistory()
settings: OpenAIChatPromptExecutionSettings = kernel.get_prompt_execution_settings_from_service_id(
        service_id="chat_completion"
    )
#settings.function_choice_behavior = FunctionChoiceBehavior.Auto(filters={"included_plugins": ["weather", "time"]})


# Set Streamlit page config
#st.set_page_config(page_title="RAG Chatbot with Prompty", layout="wide")


# Setup chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Handle input
if user_input := st.chat_input("Ask me a question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    async def run_rag():
        # Step 1: Intent Mapping
        intent_prompt = intent_prompty.create_messages(user_input=user_input)
        #intent_result = await kernel.get_service(service_id="chat_completion").complete(intent_prompt)
        # intent_result = await chat_completion.get_chat_message_content(
        #     settings=execution_settings,
        #     kernel=kernel,
        # )
        intent_result = kernel.invoke_prompt_stream(
            function_name="prompt_test",
            plugin_name="weather_test",
            prompt=intent_prompt,
            settings=settings,
        )
        intent = intent_result.strip().lower()

        if "ask_document" in intent:
            # Step 2: Run search prompt
            search_prompt = search_template.create_messages(user_input=user_input)
            search_results = await kernel.memory.search_async(
                collection="docs",
                query=user_input,
                limit=3
            )
            doc_chunks = "\n".join([r.text for r in search_results])

            # Step 3: Response grounding prompt
            response_prompt = response_template.create_messages(
                user_input=user_input,
                document_chunks=doc_chunks
            )
            response = await kernel.get_service("chat_completion").complete(response_prompt)
            return response

        elif "chitchat" in intent:
            # Just forward to model
            return await kernel.get_service("chat_completion").complete(user_input)

        else:
            return "I'm not sure how to handle that. Try rephrasing."

    async def run_rag_with_agent():
        # Step 1: Classify intent
        intent_prompt = intent_prompty.create_messages(user_input=user_input)
        #intent_result = await kernel.get_service("chat_completion").complete(intent_prompt)
        # intent_result = await chat_completion.get_chat_message_content(
        #     settings=execution_settings,
        #     kernel=kernel,
        # )
        intent_result = kernel.invoke_prompt_stream(
            function_name="prompt_test",
            prompt=intent_prompt,
            settings=settings,
        )
        intent = str(intent_result)
        print(f"Intent: {dir(intent_result)}")
        
        return intent
        # Step 2: Search documents
        # search_results = await kernel.memory.search_async("docs", query=user_input, limit=3)
        # search_results = await text_memory.search_async(
        #                         collection="docs",
        #                         query=user_input,
        #                         limit=3
        #                     )
        # doc_chunks = "\n".join([r.text for r in search_results])

        # # Step 3: Grounding prompt
        # response_prompt = response_template.create_messages(user_input=user_input, document_chunks=doc_chunks)
        # grounded_response = await kernel.get_service("chat_completion").complete(response_prompt)

        # # Step 4: Pass grounded response to Azure Agent AI
        # project_client.threads.add_message(
        #     thread_id=thread.id,
        #     role="user",
        #     content=f"User question: {user_input}\n\nSearch-based answer: {grounded_response}"
        # )

        # run = project_client.threads.run(thread.id, agent.id)
        # final_response = project_client.threads.get_message(run.last_message_id).content
        #return final_response
        
    response = asyncio.run(run_rag_with_agent())
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)
