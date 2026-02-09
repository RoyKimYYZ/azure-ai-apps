# pip install azure-ai-projects~=1.0.0b7
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings

project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str="eastus.api.azureml.ms;f1a72634-3a90-4cdb-a6d6-069cf5115068;openai;project-rkopenai")

agent = project_client.agents.get_agent("asst_I0EgCJ2MP4OiDoJbtTm8mvwZ")

agent.instructions = "You are a helpful assistant. You can help the user find resumes of .NET developers. You can also answer questions about the resumes."

thread = project_client.agents.get_thread("thread_gTUyDYv0hZ4LZUuSvjFZcsWt")

# messages = project_client.agents.list_messages(thread_id=thread.id)
# print(f"debug: {messages}")
    
message = project_client.agents.create_message(
    thread_id=thread.id,
    role="user",
    content="find me a resume of .NET developers"
)


run = project_client.agents.create_and_process_run(
    thread_id=thread.id,
    agent_id=agent.id)
messages = project_client.agents.list_messages(thread_id=thread.id)

for text_message in messages.text_messages:
    print(text_message.as_dict())
    
# assistant_message = ""
# for message in messages.data:
#     if message["role"] == "assistant":
#         assistant_message = message["content"][0]["text"]["value"]
