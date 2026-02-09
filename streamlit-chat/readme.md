# Streamlit Chat — Web Frontend for Azure AI RAG Chatbots

A Streamlit-based ChatGPT-like chat UI that connects to Azure OpenAI and Azure AI Search for document-grounded Q&A. This project demonstrates how to build a conversational web interface for RAG pipelines, with two implementation variants: a **basic chatbot** and an **agentic chatbot** using Semantic Kernel memory + Azure AI Agents.

---

## What This Project Demonstrates

| Concept | Where to Look |
|---------|---------------|
| Building a chat UI with Streamlit (`st.chat_input`, `st.chat_message`, `st.session_state`) | `chatbot.py` |
| Connecting Streamlit to Azure OpenAI for chat completions | `chatbot.py` |
| Calling a separate Flask RAG API from a Streamlit frontend | `chatbot.py` — POST to `/api/chat` |
| Azure AI Search integration (keyword search from the UI) | `chatbot.py` — `SearchClient` |
| Semantic Kernel memory with `AzureCognitiveSearchMemoryStore` + `TextMemoryPlugin` | `chatbot-agentic.py` |
| Azure AI Foundry Agents API (`AIProjectClient`, create/process runs, threads) | `chatbot-agentic.py`, `agent.py` |
| Prompty templates for intent mapping, search, and response generation | `prompts/` folder |
| Sidebar settings (model endpoint, search toggle, configurable parameters) | `chatbot.py` |
| Containerizing a Streamlit app with Docker | `dockerfile` |

---

## Architecture

```
┌──────────────────────────┐
│   Streamlit Chat UI      │
│   (chatbot.py)           │
│                          │
│  ┌────────────────────┐  │         ┌─────────────────────┐
│  │  User Chat Input   │──┼────────▶│  Flask RAG API      │
│  └────────────────────┘  │  POST   │  /api/chat          │
│                          │         │  (separate service)  │
│  ┌────────────────────┐  │         └─────────────────────┘
│  │  Azure OpenAI      │◀─┼─── Chat Completions (GPT-4o)
│  └────────────────────┘  │
│                          │
│  ┌────────────────────┐  │
│  │  Azure AI Search   │◀─┼─── Keyword/Vector Search
│  └────────────────────┘  │
└──────────────────────────┘

┌──────────────────────────┐
│   Agentic Chat UI        │
│   (chatbot-agentic.py)   │
│                          │
│  ┌────────────────────┐  │
│  │  Semantic Kernel    │  │
│  │  · TextMemoryPlugin │  │
│  │  · AzureCogSearch   │  │
│  │    MemoryStore      │  │
│  └────────────────────┘  │
│                          │
│  ┌────────────────────┐  │
│  │  Azure AI Foundry   │  │
│  │  Agents API         │  │
│  │  · Threads + Runs   │  │
│  └────────────────────┘  │
└──────────────────────────┘
```

---

## Project Structure

```
streamlit-chat/
├── chatbot.py                # Basic chatbot — Azure OpenAI + Search + Flask API calls
├── chatbot-agentic.py        # Agentic chatbot — SK memory + Azure AI Agents
├── agent.py                  # Standalone Azure AI Foundry agent script (no UI)
├── config.py                 # Shared config — env vars, logging, telemetry
├── prompts/
│   ├── intent_mapping.prompty    # Classify user intent (ask_document / chitchat / other)
│   ├── search_prompt.prompty     # Generate search queries from user input
│   └── response_prompt.prompty   # Generate grounded responses from search results
├── .env-sample               # Environment variable template
├── .streamlit/config.toml    # Streamlit theme/config
├── dockerfile                # Docker container for deployment
└── requirements.txt          # Python dependencies
```

---

## Key Technologies

| Technology | How It's Used |
|-----------|---------------|
| **Streamlit** | Chat UI framework — `st.chat_input`, `st.chat_message`, `st.session_state` for conversation state |
| **Azure OpenAI** | `AzureOpenAI` client for GPT-4o chat completions |
| **Azure AI Search** | `SearchClient` for querying the resume index |
| **Semantic Kernel** | `AzureChatCompletion`, `AzureTextEmbedding`, `TextMemoryPlugin`, `AzureCognitiveSearchMemoryStore` |
| **Azure AI Foundry Agents** | `AIProjectClient.agents` — create messages, run agent threads, retrieve responses |
| **Prompty** | `.prompty` template files for intent mapping and response generation |
| **Docker** | `dockerfile` for containerized deployment |

---

## Getting Started

### Prerequisites

- Python 3.9+
- Azure OpenAI resource with a `gpt-4o` deployment
- Azure AI Search resource with a resume index
- (For agentic variant) Azure AI Foundry project with a configured agent

### Setup

```bash
cd streamlit-chat

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env-sample .env
# Edit .env with your Azure resource credentials
```

### Run

**Basic chatbot:**
```bash
streamlit run chatbot.py --server.port 8502
```

**Agentic chatbot (SK + Azure AI Agents):**
```bash
streamlit run chatbot-agentic.py --server.port 8502
```

**Standalone agent script (no UI):**
```bash
python agent.py
```

**Docker:**
```bash
docker build -t streamlit-chat .
docker run -p 8501:8501 streamlit-chat
```

---

## Environment Variables

Copy `.env-sample` to `.env` and fill in your values:

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Chat model deployment name (e.g., `gpt-4o`) |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key (leave empty for RBAC/`DefaultAzureCredential`) |
| `AZURE_SEARCH_SERVICE_ENDPOINT` | Azure AI Search endpoint |
| `AZURE_SEARCH_INDEX` | Search index name (e.g., `resume-index`) |
| `AZURE_SEARCH_ADMIN_KEY` | Search admin key (leave empty for RBAC) |
| `AIPROJECT_CONNECTION_STRING` | Azure AI Foundry project connection string (agentic variant) |

---

## Debugging in VS Code

To attach the VS Code debugger to a running Streamlit app, add this to your `.vscode/launch.json`:

```json
{
    "name": "Python: Streamlit",
    "type": "debugpy",
    "request": "launch",
    "module": "streamlit",
    "args": [
        "run",
        "${file}",
        "--server.port",
        "8502"
    ],
    "cwd": "${workspaceFolder}/streamlit-chat"
}
```

Make sure VS Code is using the correct Python interpreter from your virtual environment (`Python: Select Interpreter` in the Command Palette).

---

## References

- [Streamlit — Build Conversational Apps](https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps)
- [Azure OpenAI Python SDK](https://learn.microsoft.com/azure/ai-services/openai/quickstart)
- [Semantic Kernel Python](https://learn.microsoft.com/semantic-kernel/overview/)
- [Azure AI Foundry Agents](https://learn.microsoft.com/azure/ai-services/agents/overview)
- [Prompty](https://prompty.ai/)