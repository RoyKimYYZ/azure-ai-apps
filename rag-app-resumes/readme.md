# RAG App Resumes — Azure AI Search + Azure AI Foundry

A Retrieval-Augmented Generation (RAG) application for tech recruiters. Upload resume PDFs to Azure Blob Storage, index them with Azure AI Search (vector + keyword hybrid search), and ask natural-language questions like *"Find me candidates with Terraform and Azure experience"* — the app retrieves the most relevant resumes and generates grounded, cited answers using Azure OpenAI.

Built with the [Azure AI Projects SDK](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-projects-readme), [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/), and [Prompty](https://prompty.ai/) prompt templates.

## What This Project Demonstrates

- **End-to-end RAG pipeline** — Upload PDFs → crack & chunk documents → generate embeddings → index → vector search → grounded LLM answer with citations.
- **Azure AI Search indexer pipeline** — Blob data source, SplitSkill (text chunking), AzureOpenAIEmbeddingSkill (vector embeddings), EntityRecognitionSkill (location extraction), index projections, and field mappings — all configured programmatically.
- **Azure AI Foundry integration** — `AIProjectClient` for connection management, embedding generation, and chat completions via the Azure AI Inference SDK.
- **Prompty templates** — Intent mapping, grounded chat, and search prompts defined as `.prompty` files in the `assets/` folder.
- **Two interfaces** — Flask REST API (`/api/chat`) and CLI with argparse for direct testing.
- **Hybrid retrieval** — Combines vector search (`VectorizedQuery`) with keyword search, with optional semantic ranking.
- **Telemetry** — OpenTelemetry tracing with Azure Monitor / Application Insights integration.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12+ |
| AI Project SDK | [azure-ai-projects](https://pypi.org/project/azure-ai-projects/) ≥ 1.0.0b1 |
| AI Inference SDK | [azure-ai-inference](https://pypi.org/project/azure-ai-inference/) ≥ 1.0.0b1 |
| Search SDK | [azure-search-documents](https://pypi.org/project/azure-search-documents/) ≥ 11.6.0b1 |
| Chat Model | Azure OpenAI (gpt-4o-mini) |
| Embeddings | Azure OpenAI (text-embedding-ada-002) |
| Vector Search | Azure AI Search (HNSW + optional semantic ranking) |
| Blob Storage | Azure Blob Storage (resume PDFs) |
| Prompt Templates | [Prompty](https://prompty.ai/) (`.prompty` files) |
| REST API | Flask |
| Authentication | [azure-identity](https://pypi.org/project/azure-identity/) (DefaultAzureCredential) |
| Telemetry | OpenTelemetry + Azure Monitor |
| Linting | Ruff, Black |

## Project Structure

```
rag-app-resumes/
├── config.py                               # Central configuration (env vars, credentials, logging)
├── requirements.txt                        # pip dependencies
├── .env-sample                             # Environment variable template
├── chat_with_resumes.py                    # RAG chat — search + grounded LLM answer (CLI)
├── chat_with_resumes_api.py                # Flask REST API wrapping chat_with_resumes
├── search_resumes.py                       # Intent mapping → embedding → vector search
├── search_resumes_api.py                   # Flask REST API wrapping search_resumes
├── create_search_index_resume.py           # Create index from CSV + upload embeddings
├── create_index_run_indexer.py             # Create index, data source, skillset, indexer (lib)
├── create_skillset_run_indexerv2.py        # Create skillset + indexer v2 (split, embed, entity)
├── create_index_skill_indexer_datasource_blob.py  # Standalone index + skillset + indexer setup
├── upload-data-blobstorage.py              # Upload resume PDFs to Azure Blob Storage
├── test-cmds.sh                            # Setup and test commands reference
├── prompttest.http                         # HTTP test requests (REST Client)
├── search-tests.http                       # AI Search REST test queries
├── resume-index.json                       # Exported index schema (reference)
├── assets/
│   ├── grounded_chat.prompty               # Grounded chat prompt template
│   ├── intent_mapping.prompty              # Intent/query extraction prompt
│   ├── search_prompt.prompty               # Search prompt template
│   ├── response_prompt.prompty             # Response formatting prompt
│   ├── basic.prompty                       # Basic prompt template
│   ├── resumes.csv                         # Sample resume data (CSV)
│   ├── products.csv                        # Sample product data
│   ├── chat_eval_data.jsonl                # Evaluation dataset
│   └── notes.txt / notes.json              # Dev notes
├── lib/
│   ├── common.py                           # Shared index/datasource/skillset/indexer builders
│   └── azure_blob_uploader.py              # Blob upload utilities
├── resume-pdfs/                            # Resume PDF files (uploaded to Blob Storage)
├── job-descriptions-txt/                   # Job description text files
└── nasa-ebooks/                            # Additional test documents
```

## How It Works

```
Resume PDFs (Blob Storage)
     │
     ▼
┌──────────────────────────┐
│  Indexer Pipeline          │
│  SplitSkill → chunk text   │
│  EmbeddingSkill → vectors  │
│  EntitySkill → locations   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Azure AI Search Index     │  resume-index
│  Fields: chunk, text_vector, title, firstName, lastName, ...
└──────────────────────────┘
           ▲
           │ vector search
┌──────────┴───────────────┐
│  User Question             │
│  → Intent Mapping (LLM)   │  intent_mapping.prompty
│  → Embed query             │  text-embedding-ada-002
│  → VectorizedQuery         │
└──────────┬───────────────┘
           │ top-k documents
           ▼
┌──────────────────────────┐
│  Grounded Chat (LLM)      │  grounded_chat.prompty
│  "Answer from context only"│  gpt-4o-mini
│  → Cited answer            │
└──────────────────────────┘
```

## Setup

### Prerequisites

- Python 3.12+
- Azure CLI (`az`) with an active subscription
- An Azure OpenAI resource with `gpt-4o-mini` and `text-embedding-ada-002` deployments
- An Azure AI Search resource
- An Azure Blob Storage account
- An Azure AI Foundry project (for `AIProjectClient` connection string)

### 1. Clone and Install

```bash
git clone https://github.com/RoyKimYYZ/azureai-chatapp.git
cd azureai-chatapp/rag-app-resumes

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env-sample .env
```

Edit `.env` with your values:

| Variable | Description |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key |
| `AZURE_SEARCH_SERVICE_ENDPOINT` | Azure AI Search service endpoint |
| `AZURE_SEARCH_ADMIN_KEY` | Azure AI Search admin key |
| `AZURE_SEARCH_INDEX` | Search index name (e.g., `resume-index`) |
| `AZURE_BLOB_CONNECTION_STRING` | Azure Blob Storage connection string |
| `AZURE_BLOB_CONTAINER` | Blob container name (e.g., `resume-sample`) |
| `AIPROJECT_CONNECTION_STRING` | Azure AI Foundry project connection string |
| `EMBEDDINGS_MODEL` | Embedding deployment name (e.g., `text-embedding-ada-002`) |
| `CHAT_MODEL` | Chat deployment name (e.g., `gpt-4o-mini`) |

### 3. Upload Resume PDFs

```bash
python upload-data-blobstorage.py
```

### 4. Create the Search Index and Run the Indexer

**Option A — From CSV (direct upload with embeddings):**

```bash
python create_search_index_resume.py
```

**Option B — From Blob Storage (indexer pipeline with skillsets):**

```bash
python create_index_run_indexer.py
# or
python create_skillset_run_indexerv2.py
```

### 5. Run the Application

**CLI:**

```bash
python chat_with_resumes.py --query "Find candidates with Python and Azure experience"
```

**Flask API:**

```bash
python chat_with_resumes_api.py
# POST http://localhost:5000/api/chat
# Body: {"query": "Find me a data scientist"}
```

**Search API:**

```bash
python search_resumes_api.py
# GET http://localhost:5001/search_resumes?query=terraform
```

## Azure AI Search Index Pipeline

The indexer pipeline processes resume PDFs through three stages:

| Stage | Skill | Description |
|---|---|---|
| 1. Chunk | `SplitSkill` | Split document text into 2000-character pages with 500-character overlap |
| 2. Embed | `AzureOpenAIEmbeddingSkill` | Generate 1536-dim vectors using text-embedding-ada-002 |
| 3. Extract | `EntityRecognitionSkill` | Extract location entities from each chunk |

Results are projected into the search index via `SearchIndexerIndexProjection`, mapping chunks, vectors, locations, titles, and metadata into the final index fields.

---

## References and Documentation

### Azure AI Search

- [Azure AI Search documentation](https://learn.microsoft.com/en-us/azure/search/) — official service docs
- [Vector search overview](https://learn.microsoft.com/en-us/azure/search/vector-search-overview) — vector indexing and querying
- [Indexer overview](https://learn.microsoft.com/en-us/azure/search/search-indexer-overview) — data source, skillset, and indexer concepts
- [Built-in skills reference](https://learn.microsoft.com/en-us/azure/search/cognitive-search-predefined-skills) — SplitSkill, EmbeddingSkill, EntityRecognitionSkill
- [Blob storage indexing](https://learn.microsoft.com/en-us/azure/search/search-howto-indexing-azure-blob-storage) — connecting Blob Storage to AI Search
- [azure-search-documents SDK](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/search/azure-search-documents) — Python SDK source and samples

### Azure AI Projects & Inference

- [azure-ai-projects SDK](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-projects-readme) — AIProjectClient reference
- [azure-ai-inference SDK](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-inference-readme) — chat completions and embeddings client
- [Azure AI Foundry documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/) — project management, model catalog

### Azure OpenAI

- [Azure OpenAI Service documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/) — model deployments and API reference
- [Embeddings guide](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/embeddings) — text-embedding-ada-002 usage
- [openai/openai-python](https://github.com/openai/openai-python) — Python client

### Prompty

- [Prompty documentation](https://prompty.ai/) — prompt template format and tooling
- [Prompty specification](https://github.com/microsoft/prompty) — GitHub repo and spec

### Additional

- [Flask documentation](https://flask.palletsprojects.com/) — REST API framework
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/) — tracing and instrumentation
- [Azure Monitor OpenTelemetry](https://learn.microsoft.com/en-us/azure/azure-monitor/app/opentelemetry-enable) — Application Insights integration

---

This project is for educational and demonstration purposes.
