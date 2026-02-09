import os
import base64
from openai import AzureOpenAI
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
import sys
import pathlib
from azure.search.documents import SearchClient
import requests
import json
path_env = find_dotenv()

load_dotenv()
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]  
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]  
search_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]  
search_key = os.environ["AZURE_SEARCH_ADMIN_KEY"]  
azure_openai_key = os.environ["AZURE_OPENAI_KEY"] if len(os.environ["AZURE_OPENAI_KEY"]) > 0 else None
search_index = os.environ["AZURE_SEARCH_INDEX"]
search_credential = AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]) if len(os.environ["AZURE_SEARCH_ADMIN_KEY"]) > 0 else DefaultAzureCredential()

st.title("ChatGPT-like clone")


SYSTEM_MESSAGE = {"role": "assistant", "content": "How can I help you today?"}
if "messages" not in st.session_state:
    st.session_state['messages'] = [SYSTEM_MESSAGE]

# Function to clear chat history
def clear_chat_history():
    st.session_state['messages'] = [SYSTEM_MESSAGE]
    
# Initialize Azure OpenAI Service client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=azure_openai_key,  
    api_version="2025-01-01-preview",
)

# Initialize Azure Cognitive Search client
search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=search_index,
    credential=AzureKeyCredential(search_key),
)

def search_query(query_text):
    results = search_client.search(query_text)
    return [result for result in results]



### SIDE BAR
with st.sidebar:
  st.title('Settings')
  
  model_endpoint = st.text_input('Endpoint', endpoint, help='The endpoint of the model to use for the chatbot.')
  model_name = st.text_input('Model name', deployment, help='Enter the name of the model to use for the chatbot.')
  chat_endpoint = st.text_input('Chat Endpoint', 'http://localhost:5000/api/chat', help='The endpoint of the model to use for the chatbot.')
  enable_test_search = st.checkbox('Enable test search', value=False, help='Check to enable testing search upon app start.')
  #response_temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=1.0, value=0.7, step=0.01, help='The temperature to use when generating text. The higher the temperature, the more creative the response.')
  #response_top_k = st.sidebar.slider('Top K', min_value=0, max_value=1, value=-1, step=1, help='The number of top tokens to consider. Set to -1 to consider all tokens.')
  #response_top_p = st.sidebar.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01, help='The cumulative probability threshold for nucleus sampling.')
  #response_repetition_penalty = st.sidebar.slider('Repetition Penalty', min_value=0.0, max_value=2.0, value=1.0, step=0.01, help='The repetition penalty to use when generating text.')
  #response_max_tokens = st.sidebar.slider('Max Tokens', min_value=200, max_value=1000, value=200, step=100, help='The maximum number of tokens to generate in the response.')

  # Button to start a new chat
  st.button('New chat', on_click=clear_chat_history)

# Example usage
if enable_test_search:
    query_text = "search for cloud engineer"
    search_results = search_query(query_text)
    st.write("Search Results:", search_results)
    
### CHAT 
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = deployment

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# PROMPTING OPEN AI MODEL
# if prompt := st.chat_input("What is up?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         stream = client.chat.completions.create(
#             model=st.session_state["openai_model"],
#             messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ],
#             stream=True,
#         )
#         response = st.write_stream(stream)
#     st.session_state.messages.append({"role": "assistant", "content": response})
    
# PROMPTING CHAT_WTH_RESUMES_API API
if prompt := st.chat_input("What skills and experience you looking for in a candidate?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Prepare the payload for the custom LLM API
                payload = {
                    "query": prompt,
                    "messages": [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                }
                headers = {"Content-Type": "application/json"}

                # Call the custom LLM API
                try:
                    response = requests.post(
                        url=chat_endpoint, json=payload, headers=headers
                    )
                    response.raise_for_status()  # Raise an error for HTTP errors
                    response_data = response.json()  # Parse the JSON response
                    print(response) 
                    print(type(response)) 
                    assistant_response = response_data.get("content", "No response from model.")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    assistant_response = "Error: Invalid JSON response from the server."
                except requests.exceptions.RequestException as e:
                    assistant_response = f"Error: {e}"

                st.markdown(assistant_response)

            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    
    
