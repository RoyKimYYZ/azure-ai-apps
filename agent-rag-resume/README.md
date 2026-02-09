# Agent RAG Resume — Azure AI Search + Semantic Kernel

A RAG (Retrieval-Augmented Generation) chatbot that searches indexed resume documents using Azure AI Search and answers questions grounded in the retrieved content. Built with [Semantic Kernel](https://github.com/microsoft/semantic-kernel) for LLM orchestration, [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/) for chat and embeddings, and [Streamlit](https://streamlit.io/) for the web UI.

The application embeds a user's question, performs a vector search against an Azure AI Search index of resume PDFs, and passes the retrieved document chunks as grounding context to the LLM — which then generates a cited, factual answer.

## What This Project Demonstrates

- **RAG pattern with Azure AI Search** — Vector search (VectorizedQuery) against a pre-built index of resume documents with embeddings.
- **Semantic Kernel orchestration** — `Kernel`, `AzureChatCompletion`, `AzureTextEmbedding`, `ChatHistory`, prompt plugins, and function invocation.
- **Prompt plugin system** — A Semantic Kernel prompt plugin (`UserPromptPlugin/USER_PROMPT_TEMPLATE`) with a dedicated `skprompt.txt` and `config.json` that defines the grounded chat prompt as a reusable SK function.
- **Two application variants** — A Semantic Kernel version (`ai_foundry_agent_sk.py`) and a direct Azure OpenAI SDK version (`azure_rag_chat_ai_foundry_agent_style_python_single_file_app.py`) for comparison.
- **Grounded answers with citations** — The system prompt enforces strict grounding; answers include `[title]` citations from retrieved passages.
- **Environment-driven configuration** — All Azure endpoints, keys, model names, and search parameters are set via `.env` / environment variables.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python ≥ 3.12 |
| Package Manager | [uv](https://docs.astral.sh/uv/) (Astral) |
| LLM Orchestration | [Semantic Kernel](https://github.com/microsoft/semantic-kernel) ≥ 1.0.3 |
| Chat Model | Azure OpenAI (gpt-4o / gpt-4o-mini) |
| Embeddings | Azure OpenAI (text-embedding-ada-002 / text-embedding-3-large) |
| Vector Search | [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/) (azure-search-documents ≥ 11.6.0b1) |
| Web UI | [Streamlit](https://streamlit.io/) ≥ 1.35 |
| Azure Identity | [azure-identity](https://pypi.org/project/azure-identity/) ≥ 1.15 |
| AI Inference | [azure-ai-inference](https://pypi.org/project/azure-ai-inference/) ≥ 1.0.0b1 |
| AI Projects | [azure-ai-projects](https://pypi.org/project/azure-ai-projects/) ≥ 1.0.0b1 |
| Build System | [Hatchling](https://hatch.pypa.io/) |
| Linting | Ruff, Black |

## How It Works

```
User Question
     │
     ▼
┌─────────────────────┐
│  Embed question      │  Azure OpenAI Embeddings
│  (text-embedding)    │
└────────┬────────────┘
         │ vector
         ▼
┌─────────────────────┐
│  Vector search       │  Azure AI Search index
│  (VectorizedQuery)   │  Fields: id, chunk, title, text_vector
└────────┬────────────┘
         │ top-k passages
         ▼
┌─────────────────────┐
│  Build grounding     │  Concatenate retrieved chunks with
│  context             │  titles and citations
└────────┬────────────┘
         │ context string
         ▼
┌─────────────────────┐
│  SK Kernel.invoke()  │  UserPromptPlugin → AzureChatCompletion
│  (grounded chat)     │  System: "answer from context only"
└────────┬────────────┘
         │ answer + citations
         ▼
┌─────────────────────┐
│  Streamlit UI        │  Chat messages, retrieved passages
│                      │  expander, prompt template sidebar
└─────────────────────┘
```

## Setup and Deployment

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- An Azure OpenAI resource with a chat deployment (e.g., `gpt-4o-mini`) and an embedding deployment (e.g., `text-embedding-ada-002`)
- An Azure AI Search resource with a pre-built index containing resume documents with a vector field (`text_vector`)

### 1. Clone and Install

```bash
git clone https://github.com/RoyKimYYZ/azureai-chatapp.git
cd azureai-chatapp/agent-rag-resume

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### 2. Configure Environment Variables

```bash
cp .env-sample .env
```

Edit `.env` with your values:

| Variable | Description |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | Chat model deployment name (e.g., `gpt-4o`) |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding deployment name (e.g., `text-embedding-ada-002`) |
| `AZURE_SEARCH_ENDPOINT` | Azure AI Search service endpoint |
| `AZURE_SEARCH_API_KEY` | Azure AI Search admin key |
| `AZURE_SEARCH_INDEX_NAME` | Name of the search index (e.g., `resume-index`) |
| `AIPROJECT_CONNECTION_STRING` | *(Optional)* Azure AI Foundry project connection string |

### 3. Run the Application

```bash
# Semantic Kernel version (recommended)
uv run streamlit run agent_rag_resume/ai_foundry_agent_sk.py

# Or use the runner script
./run_ai_foundry_agent_sk.sh

# Direct Azure OpenAI SDK version (no Semantic Kernel)
uv run streamlit run agent_rag_resume/azure_rag_chat_ai_foundry_agent_style_python_single_file_app.py
```

### 4. Run as a Python Module

```bash
uv run python -m agent_rag_resume
```

### 5. Run Tests

```bash
./run_ai_foundry_agent_sk.sh test

# Or directly
uv run pytest
```

### 6. Build the Package

```bash
uv build
# Output: dist/agent_rag_resume-0.1.0-py3-none-any.whl
```

## Semantic Kernel Prompt Plugin

The `UserPromptPlugin/USER_PROMPT_TEMPLATE` plugin defines the grounded chat prompt as a reusable Semantic Kernel function:

- **`skprompt.txt`** — The prompt template with `{{$question}}`, `{{$context}}`, and `{{$documents}}` variables. Instructs the LLM to select the top 3 relevant resumes, list skills, summarize experience, draft screening questions, and compose a candidate outreach email.
- **`config.json`** — Declares the input variables and execution settings (`temperature: 0.1`, `max_tokens: 800`).

The plugin is loaded at startup via `kernel.add_plugin(parent_directory=plugins_dir, plugin_name="UserPromptPlugin")` and invoked with `kernel.invoke(USER_PROMPT_FUNC, ...)`.

---

## References and Documentation

### Semantic Kernel

- [microsoft/semantic-kernel](https://github.com/microsoft/semantic-kernel) — official GitHub repo (Python SDK under `python/`)
- [Semantic Kernel documentation](https://learn.microsoft.com/en-us/semantic-kernel/overview/) — Microsoft Learn overview and guides
- [Semantic Kernel Python API reference](https://learn.microsoft.com/en-us/python/api/semantic-kernel/) — class and module reference
- [Semantic Kernel prompt template syntax](https://learn.microsoft.com/en-us/semantic-kernel/prompts/prompt-template-syntax) — `{{$variable}}` and function call syntax

### Azure AI Search

- [Azure AI Search documentation](https://learn.microsoft.com/en-us/azure/search/) — official service docs
- [Azure AI Search vector search](https://learn.microsoft.com/en-us/azure/search/vector-search-overview) — vector indexing and querying
- [azure-search-documents Python SDK](https://learn.microsoft.com/en-us/python/api/azure-search-documents/) — `SearchClient`, `VectorizedQuery` reference
- [Azure/azure-sdk-for-python (search)](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/search/azure-search-documents) — SDK source code and samples

### Azure OpenAI

- [Azure OpenAI Service documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/) — model deployments and API reference
- [Azure OpenAI embeddings guide](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/embeddings) — embedding model usage
- [openai/openai-python](https://github.com/openai/openai-python) — Python client used for Azure OpenAI calls

### Streamlit

- [Streamlit documentation](https://docs.streamlit.io/) — official docs
- [streamlit/streamlit](https://github.com/streamlit/streamlit) — GitHub repo
- [Streamlit Chat elements](https://docs.streamlit.io/develop/api-reference/chat) — `st.chat_message`, `st.chat_input`

### Python Tooling

- [astral-sh/uv](https://github.com/astral-sh/uv) — fast Python package manager
- [Hatchling build backend](https://hatch.pypa.io/latest/) — build system used in `pyproject.toml`
- [astral-sh/ruff](https://github.com/astral-sh/ruff) — linter used in this project

---

---

## Deploy to AKS

The `aks/` subfolder contains a complete set of Kubernetes manifests managed by Kustomize.

### Prerequisites

- Docker (or Podman)
- Azure CLI (`az`) with an active subscription
- `kubectl` configured for your AKS cluster
- [Ingress-NGINX](https://kubernetes.github.io/ingress-nginx/) controller installed in the cluster
- Azure Container Registry accessible from the AKS cluster

### 1. Build and Push the Docker Image

```bash
# Build + push + deploy in one step
cd aks
./runbook_deploy.sh build

# Or manually:
az acr login --name rkimacr
docker build -t rkimacr.azurecr.io/agent-rag-resume:v1 -f Dockerfile .
docker push rkimacr.azurecr.io/agent-rag-resume:v1
```

### 2. Configure Secrets

```bash
cd aks
cp secret.yaml-sample secret.yaml
# Edit secret.yaml — fill in your API keys:
#   AZURE_OPENAI_API_KEY
#   AZURE_SEARCH_API_KEY
#   AIPROJECT_CONNECTION_STRING (optional)
```

> **Note:** `secret.yaml` is git-ignored. Never commit real secrets.

### 3. Update ConfigMap

Edit `aks/configmap.yaml` with your Azure resource endpoints and deployment names.

### 4. Deploy

```bash
cd aks
./runbook_deploy.sh
```

The script will:
1. Attach ACR to the AKS cluster (if not already attached)
2. Detect the Ingress-NGINX controller's external LoadBalancer IP
3. Verify `secret.yaml` exists
4. Apply all manifests via `kubectl apply -k`
5. Wait for the rollout to complete and print the app URL

### 5. Access the Application

```
http://<INGRESS_EXTERNAL_IP>/agent-rag-resume
```

### Kubernetes Resources Created

| Resource | Name | Details |
|---|---|---|
| Deployment | `agent-rag-resume` | 1–3 replicas, CPU 100m–1, Mem 512Mi–1Gi |
| Service | `agent-rag-resume` | ClusterIP, port 80 → targetPort 8501 |
| Ingress | `agent-rag-resume` | NGINX class, path `/agent-rag-resume` with rewrite |
| HPA | `agent-rag-resume` | Scale at 70% CPU average utilization |
| ConfigMap | `agent-rag-resume-config` | Non-sensitive environment variables |
| Secret | `agent-rag-resume-secrets` | API keys (git-ignored) |

---

This project is for educational and demonstration purposes.