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
from semantic_kernel.kernel import Kernel
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin

from semantic_kernel.agents import AzureAIAgent
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings

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
    thread = project_client.agents.get_thread("thread_gTUyDYv0hZ4LZUuSvjFZcsWt")
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


    async def run_rag_with_agent():
        message = project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=user_input
        )

        run = project_client.agents.create_and_process_run(
            thread_id=thread.id,
            agent_id=agent.id)
        messages = project_client.agents.list_messages(thread_id=thread.id)

        # for text_message in messages.text_messages:
        #     print(text_message.as_dict())
            
        # assistant_message = ""
        # for message in messages.data:
        #     if message["role"] == "assistant":
        #         assistant_message = message["content"][0]["text"]["value"]
        #         print(f"assistant message: {assistant_message}")
        
        return messages.as_dict()["data"][0]["content"][0]["text"]["value"]
        
    response = asyncio.run(run_rag_with_agent())
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)
