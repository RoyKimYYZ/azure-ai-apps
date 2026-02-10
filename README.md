# Azure AI Apps

A collection of Python demo applications for engineers learning how to build **Retrieval-Augmented Generation (RAG)** pipelines, **AI agent orchestration**, and **chatbot UIs** on Azure AI services. One key focus is to show how to leverage AI hosting capabilities on Azure Kubernetes such as KAITO inference AI models and hosting the applications in AKS. 

These are demo and learning projects for myself that I [blog](https://www.roykim.ca/) about, use as demos for online and in-person tech presentations and show case in my youtube channel https://youtube.com/roykimyyz

Each sub-project is a standalone, runnable demo that highlights a different architecture pattern or Azure service integration — from simple RAG chat to multi-agent frameworks deployed on Azure Kubernetes.

---

## Table of Contents

| # | Project | What You'll Learn | Key Technologies |
|---|---------|-------------------|------------------|
| 1 | [agent-rag-resume](#1-agent-rag-resume) | Building a RAG agent with Semantic Kernel plugin system | Semantic Kernel, Azure AI Search, Streamlit, Azure Kubernetes Service |
| 2 | [agentframework](#2-agentframework) | Streamlit chatbot with Multi-agent orchestration with Microsoft Agent Framework SDK on AKS | Agent Framework SDK, KAITO, AKS |
| 3 | [app-rag-aisearch-sk](#3-app-rag-aisearch-sk) | SK vector store connectors, memory plugins, and search skillset pipelines | Semantic Kernel (plugins + memory), Azure AI Search, Flask |
| 4 | [rag-app-resumes](#4-rag-app-resumes) | End-to-end RAG indexer pipeline with skillsets and Prompty templates | Azure AI Projects SDK, Prompty, OpenTelemetry |
| 5 | [rag-chatapp-retail](#5-rag-chatapp-retail) | RAG with AI evaluation (groundedness metrics) | Azure AI Evaluation SDK, Prompty |
| 6 | [streamlit-chat](#6-streamlit-chat) | Streamlit chat UI with basic and agentic variants. Basic implementation. | Streamlit, Semantic Kernel, Azure AI Agents |
| 7 | [python-uv-project-template](#7-python-uv-project-template) | Starter template for uv-based Python projects | uv, Ruff, mypy |

---

## Core Azure Services Used

| Service | Role in These Demos |
|---------|---------------------|
| **Azure OpenAI** | Chat completions (GPT-4o / GPT-4o-mini) and embeddings (text-embedding-ada-002 / text-embedding-3-large) |
| **Azure AI Search** | Document indexing, vector/hybrid/semantic search with skillsets and indexers |
| **Azure AI Foundry** | Project management, connection handling, and AI inference via `AIProjectClient` |
| **Azure Blob Storage** | Source document storage (resume PDFs, product data) |
| **Azure Kubernetes Service (AKS)** | Container hosting with KAITO GPU inference and Ingress |
| **Azure Container Registry (ACR)** | Docker image storage and deployment |
| **Azure Monitor / Application Insights** | OpenTelemetry tracing and telemetry |

---

## Sub-Projects

### 1. agent-rag-resume

📁 [`agent-rag-resume/`](agent-rag-resume/)

**What it demonstrates:** A RAG chatbot that searches indexed resume documents using Azure AI Search and generates grounded, cited answers via Semantic Kernel's plugin system.

**How it works:** Embeds the user's question → performs vector search against a pre-built resume index → passes retrieved document chunks as grounding context to the LLM → returns cited answers with `[title]` references.

| | |
|---|---|
| **Technologies** | Semantic Kernel (≥ 1.0.3), Azure OpenAI, Azure AI Search, Streamlit, azure-ai-projects |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) |
| **Entry Points** | `agent_rag_resume/ai_foundry_agent_sk.py` — SK version<br>`agent_rag_resume/azure_rag_chat_ai_foundry_agent_style_python_single_file_app.py` — direct SDK version |
| **Key Concepts** | SK prompt plugins (`UserPromptPlugin`), grounded answers with citations, two implementation variants for comparison |
| **Deployment** | AKS manifests included (Kustomize) |

```bash
cd agent-rag-resume
uv sync && uv run streamlit run agent_rag_resume/ai_foundry_agent_sk.py
```

---

### 2. agentframework

📁 [`agentframework/`](agentframework/)

**What it demonstrates:** Multi-agent AI chatbot built with Microsoft's [Agent Framework SDK](https://pypi.org/project/agent-framework/), showing how to orchestrate multiple LLM backends through a single config-driven architecture.

**How it works:** Uses the SDK's `ChatAgent` orchestration loop and `BaseChatClient` contract to route conversations to different backends (Azure AI Foundry, KAITO GPU inference, KAITO RAGEngine) based on YAML configuration.

| | |
|---|---|
| **Technologies** | Agent Framework SDK (v1.0.0b), Streamlit, Jinja2, PyYAML, Pydantic, Click |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) · Python 3.13 |
| **Entry Points** | `chatbot/chatbot.py` — Streamlit UI<br>`cli.py` — CLI agents<br>`ai_chat_client.py` — BaseChatClient implementation |
| **Key Concepts** | 5 agents across 3 backends, Strategy pattern for backend swapping, custom `KaitoChatClient`, config-driven agent selection |
| **Deployment** | Production K8s (Kustomize, Ingress-NGINX, HPA) |

```bash
cd agentframework
uv sync && uv run streamlit run chatbot/chatbot.py
```

---

### 3. app-rag-aisearch-sk

📁 [`app-rag-aisearch-sk/`](app-rag-aisearch-sk/)

**What it demonstrates:** RAG application powered by Semantic Kernel's full plugin ecosystem — vector store connectors, `ChatCompletionAgent`, `TextMemoryPlugin`, and Prompty prompt templates for document and resume chat.

**How it works:** Uses SK's `AzureAISearchCollection` vector store connector to query indexed documents, maintains conversation history via memory plugins, and generates answers through SK's `ChatCompletionAgent`.

| | |
|---|---|
| **Technologies** | Semantic Kernel (with Azure extensions + memory), Azure AI Search, Azure OpenAI, Flask, Prompty |
| **Package Manager** | pip (`requirements.txt`) |
| **Entry Points** | `chat_with_docs.py` — SK document chat<br>`chat_with_resumes.py` — resume RAG chat<br>`chat_with_docs_api.py` / `chat_with_resumes_api.py` — Flask APIs<br>`create_skillset_run_indexerv2.py` — indexer pipeline setup |
| **Key Concepts** | SK vector store connectors, SK prompt plugins, memory plugins, index/skillset/indexer pipeline creation, Prompty templates |

```bash
cd app-rag-aisearch-sk
pip install -r requirements.txt
python chat_with_docs.py
# or Flask API:
python chat_with_resumes_api.py  # http://localhost:5000
```

---

### 4. rag-app-resumes

📁 [`rag-app-resumes/`](rag-app-resumes/)

**What it demonstrates:** End-to-end RAG pipeline for tech recruiters — from PDF upload to Azure Blob Storage, through indexing with Azure AI Search skillsets, to natural-language Q&A with grounded, cited answers.

**How it works:** Upload resume PDFs → index with Azure AI Search (SplitSkill → EmbeddingSkill → EntityRecognitionSkill) → hybrid vector + keyword search → generate grounded answers using Azure AI Projects SDK + Prompty templates.

| | |
|---|---|
| **Technologies** | Azure AI Projects SDK, Azure AI Inference SDK, azure-search-documents, Prompty, Flask, OpenTelemetry, pandas |
| **Package Manager** | pip (`requirements.txt`) |
| **Entry Points** | `chat_with_resumes.py` — CLI<br>`chat_with_resumes_api.py` — Flask API (`/api/chat`)<br>`create_skillset_run_indexerv2.py` — indexer pipeline<br>`upload-data-blobstorage.py` — PDF upload |
| **Key Concepts** | Full indexer pipeline (SplitSkill → EmbeddingSkill → EntityRecognitionSkill), Prompty templates for intent mapping and grounded chat, hybrid search, OpenTelemetry tracing |

```bash
cd rag-app-resumes
pip install -r requirements.txt
python chat_with_resumes.py --query "Find candidates with Python and Azure experience"
```

---

### 5. rag-chatapp-retail

📁 [`rag-chatapp-retail/`](rag-chatapp-retail/)

**What it demonstrates:** RAG for retail product search with an **AI evaluation pipeline** — measures answer groundedness against source documents using Azure AI Evaluation SDK.

**How it works:** Maps user intent → generates embeddings → performs vector search against a product index → generates grounded answers → evaluates response quality with `GroundednessEvaluator`.

| | |
|---|---|
| **Technologies** | Azure AI Projects SDK, Azure AI Evaluation SDK, Prompty, Flask, OpenTelemetry, pandas |
| **Package Manager** | pip (`requirements.txt`) |
| **Entry Points** | `chat_with_products.py` — product RAG chat<br>`get_product_documents.py` — intent → embed → vector search<br>`evaluate.py` — groundedness evaluation<br>`create_search_index.py` — index creation |
| **Key Concepts** | Intent mapping with Prompty, `VectorizedQuery` product search, `GroundednessEvaluator` evaluation pipeline, chat protocol compliant responses |

```bash
cd rag-chatapp-retail
pip install -r requirements.txt
python chat_with_products.py --query "I need a new tent for 4 people"
python evaluate.py  # Run groundedness evaluation
```

---

### 6. streamlit-chat

📁 [`streamlit-chat/`](streamlit-chat/)

**What it demonstrates:** Streamlit-based ChatGPT-like web UI for querying indexed documents, with both a basic and an agentic variant using Semantic Kernel memory and Azure AI Agents.

**How it works:** Provides a chat interface that connects to Azure OpenAI and Azure AI Search. The agentic version adds SK `TextMemoryPlugin` with `AzureCognitiveSearchMemoryStore` and Azure AI Agent integration.

| | |
|---|---|
| **Technologies** | Streamlit, OpenAI Python SDK, Semantic Kernel (with Azure memory), azure-search-documents, Prompty |
| **Package Manager** | pip (`requirements.txt`) |
| **Entry Points** | `chatbot.py` — basic chatbot UI<br>`chatbot-agentic.py` — SK + Azure AI Agent version<br>`agent.py` — standalone agent script |
| **Key Concepts** | Chat UI patterns, sidebar settings (model/endpoint/search toggle), SK memory store integration, Azure AI Agent integration |

```bash
cd streamlit-chat
pip install -r requirements.txt
streamlit run chatbot.py --server.port 8502
```

---

### 7. python-uv-project-template

📁 [`python-uv-project-template/`](python-uv-project-template/)

**What it demonstrates:** Minimal starter template for new Python projects using the [uv](https://docs.astral.sh/uv/) package manager with Ruff linting and mypy type checking.

| | |
|---|---|
| **Technologies** | Python 3.13, uv, Ruff, mypy |
| **Entry Points** | `main.py` |
| **Key Concepts** | `pyproject.toml` configuration, uv workflow, Ruff + mypy setup |

```bash
cd python-uv-project-template
uv sync && uv run main.py
```

---

## Getting Started

### Prerequisites

- **Python 3.12+** (3.13 for agentframework)
- [**uv**](https://docs.astral.sh/uv/getting-started/installation/) package manager (for uv-based projects)
- **Azure CLI** (`az`) with an active subscription
- An **Azure OpenAI** resource with chat and embedding model deployments
- An **Azure AI Search** resource

### Setup Pattern

Each sub-project follows a similar workflow:

1. **Navigate** to the sub-project directory
2. **Install dependencies** — `uv sync` (uv projects) or `pip install -r requirements.txt` (pip projects)
3. **Configure environment** — copy `.env-sample` to `.env` and fill in your Azure resource credentials
4. **Run** — see the commands and entry points listed above

> 💡 **Tip:** Each sub-project has its own README with more detailed setup instructions.

---

## License

This project is for educational and demonstration purposes.
