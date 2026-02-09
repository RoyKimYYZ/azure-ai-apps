"""
Azure RAG Chat — Single-file Streamlit app using Azure OpenAI + Azure AI Search (vector + hybrid)

Features
- Prompt templates (system + user)
- Hybrid retrieval (vector + keyword) with optional semantic ranking
- Citations
- Simple chat history memory (client-side)
- Environment-driven configuration

Run
  pip install -r requirements.txt  # see requirements_str below
  streamlit run app.py

Env Vars (set these before running)
  AZURE_OPENAI_ENDPOINT=...
  AZURE_OPENAI_API_KEY=...
  AZURE_OPENAI_API_VERSION=2024-08-01-preview  # or newer supported
  AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini  # chat deployment name
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large  # embedding deployment name

  AZURE_SEARCH_ENDPOINT=https://<your-search-service>.search.windows.net
  AZURE_SEARCH_API_KEY=...
  AZURE_SEARCH_INDEX_NAME=<index-name>
  AZURE_SEARCH_SEMANTIC_CONFIG=<optional-semantic-config-name>

Optional
  TOP_K=5               # retrieved chunks
  MAX_TOKENS=800        # for the LLM answer
  TEMPERATURE=0.1
  CONTEXT_CHAR_LIMIT=12000   # to keep prompt size bounded

Index assumptions
  - Your Azure AI Search index has fields:
      id (Edm.String, key=true)
      content (Edm.String)
      content_vector (Collection(Edm.Single))  # same dim as embedding model
      title (Edm.String)
      source (Edm.String)  # optional URL or path
      chunk_id (Edm.String) # optional
"""

import os
import textwrap
from typing import List, Dict, Any

import streamlit as st

# Azure OpenAI SDK
from openai import AzureOpenAI
try:
    from openai import AzureOpenAI
except Exception:
    #AzureOpenAI = None
    raise RuntimeError(
        "openai package not available. Please `pip install openai>=1.30.0`. "
        "If using a virtual environment, make sure to activate it first."
    )

# Azure Cognitive Search SDK
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType, VectorQuery, VectorizedQuery
# ------------------------------
# Configuration
# ------------------------------
from dotenv import load_dotenv
load_dotenv()

def get_env(name: str, default: str = "") -> str:
    val = os.getenv(name, default)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val

# Defaults with safe fallbacks (some optional)
AZURE_OPENAI_ENDPOINT = get_env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = get_env("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_DEPLOYMENT = get_env("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = get_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

AZURE_SEARCH_ENDPOINT = get_env("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = get_env("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = get_env("AZURE_SEARCH_INDEX_NAME")
AZURE_SEARCH_SEMANTIC_CONFIG = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "")

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
CONTEXT_CHAR_LIMIT = int(os.getenv("CONTEXT_CHAR_LIMIT", "12000"))

# ------------------------------
# Prompt Templates
# ------------------------------
SYSTEM_TEMPLATE = (
    """
You are a helpful enterprise assistant. Ground all answers strictly in the provided context.
Rules:
- If the answer is not in the context, say "I don't know based on the indexed documents." Do not fabricate.
- Prefer concise, direct answers.
- Always provide citations as [title](source) at the end of relevant sentences.
- If listing steps, present them as short bullet points.
    """
    .strip()
)

USER_TEMPLATE = (
    """
User question:
{question}

Context (from retrieved documents):
{context}

Answer the user's question using ONLY the context above. If insufficient, say you don't know.
Provide citations like [{{title}}]({{source}}) for statements derived from specific passages.
    """
    .strip()
)

# ------------------------------
# Azure Clients
# ------------------------------

if AzureOpenAI is None:
    print("Azure OpenAI client not available. Please install the openai package.")
    #raise RuntimeError("openai package not available. Please `pip install openai>=1.30.0`. ")

aoai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
)

# ------------------------------
# Embedding & Retrieval
# ------------------------------

def embed_text(text: str) -> List[float]:
    resp = aoai_client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=[text],
    )
    return resp.data[0].embedding  # type: ignore[attr-defined]

def search_index_vectorized(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Search index using VectorizedQuery (Python SDK >=11.6.0b1)."""
    
    # fields are the vector fields in the search index
    vector_query = VectorizedQuery(vector=embed_text(query), k_nearest_neighbors=3, fields="text_vector")

    results = search_client.search(
        vector_queries=[vector_query],
        # Ensure these fields exist in the search index
        select=["id", "chunk", "title", "chunk_id"],
    )

    
    # embedding = embed_text(query)
    # vectorized_query = VectorizedQuery(
    #     vector=embedding,
    #     fields="text_vector",
    #     k_nearest_neighbors=top_k,
    # )
    # kwargs = {
    #     "top": top_k,
    #     "vectorized_queries": [vectorized_query],
    #     "select": ["id", "content", "title", "source", "chunk_id"],
    #     "search_text": query,
    # }
    # results = search_client.search(**kwargs)
    hits: List[Dict[str, Any]] = []
    for doc in results:
        d = dict(doc)
        hits.append(
            {
                "id": d.get("id"),
                "chunk": d.get("chunk", ""),
                "title": d.get("title") or "Untitled",
                #"source": d.get("source") or "",
                "chunk_id": d.get("chunk_id") or "",
            }
        )
    return hits
    
    
def search_index(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Hybrid: keyword + vector; optional semantic ranking."""
    embedding = embed_text(query)

    # vector -> vector_queries with VectorQuery
    vector_query = VectorQuery(
        vector=embedding,
        kind="vector",
        k_nearest_neighbors=top_k,
        fields="content_vector",
    )
    vector_query.kind = "vector"
    kwargs = {
        "top": top_k,
        "vector_queries": [vector_query],
        "select": ["id", "chunk", "title", "chunk_id"],
    }

    # Add semantic if configured; still include search_text for hybrid
    if AZURE_SEARCH_SEMANTIC_CONFIG:
        kwargs.update(
            {
                "query_type": QueryType.SEMANTIC,
                "semantic_configuration_name": AZURE_SEARCH_SEMANTIC_CONFIG,
                "search_text": query,
                "kind": "semantic"
            }
        )
    else:
        kwargs.update({"search_text": query})

    results = search_client.search(**kwargs)

    hits: List[Dict[str, Any]] = []
    for doc in results:
        d = dict(doc)
        hits.append(
            {
                "id": d.get("id"),
                "chunk": d.get("chunk", ""),
                "title": d.get("title") or "Untitled",
                "chunk_id": d.get("chunk_id") or "",
            }
        )

    return hits


# ------------------------------
# Prompt Assembly & Chat
# ------------------------------

def build_context(passages: List[Dict[str, Any]], limit_chars: int = CONTEXT_CHAR_LIMIT) -> str:
    parts = []
    total = 0
    for p in passages:
        title = p.get("title") or "Untitled"
        src = p.get("source") or ""
        chunk = p.get("content") or ""
        header = f"[title: {title}] [source: {src}]\n"
        block = header + chunk.strip()
        if total + len(block) > limit_chars:
            remaining = max(0, limit_chars - total)
            block = block[:remaining]
        parts.append(block)
        total += len(block)
        if total >= limit_chars:
            break
    return "\n\n---\n\n" + "\n\n---\n\n".join(parts)


def format_user_prompt(question: str, context: str) -> str:
    return USER_TEMPLATE.format(question=question, context=context)


def chat_completion(messages: List[Dict[str, str]]) -> str:
    resp = aoai_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content or ""


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Azure RAG Chat", page_icon="🧠", layout="wide")

st.title("🧠 Azure RAG Chat — AI Search + Azure OpenAI")
with st.expander("Configuration (read-only from env)"):
    st.write({
        "search_index": AZURE_SEARCH_INDEX_NAME,
        "semantic_config": AZURE_SEARCH_SEMANTIC_CONFIG or "(none)",
        "chat_model": AZURE_OPENAI_DEPLOYMENT,
        "embedding_model": AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        "top_k": TOP_K,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {role, content}

# Sidebar: Prompt templates preview
with st.sidebar:
    st.subheader("Prompt Templates")
    st.code(textwrap.dedent(SYSTEM_TEMPLATE), language="markdown")
    st.code(textwrap.dedent(USER_TEMPLATE), language="markdown")

# Chat display
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask a question about your indexed documents…")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Retrieving relevant passages…"):
        passages = search_index_vectorized(question, top_k=TOP_K)

    context = build_context(passages)

    # Show retrieved snippets + citations panel
    with st.expander("Retrieved passages"):
        for i, p in enumerate(passages, start=1):
            st.markdown(f"**{i}. {p['title']}** — {p.get('source','')}")
            st.write(textwrap.shorten(p["chunk"], width=500, placeholder=" …"))
            st.divider()

    # Build messages
    messages: List[Dict[str, str]] = []
    messages.append({"role": "system", "content": SYSTEM_TEMPLATE})

    # Include limited chat memory (last 4 user/assistant turns)
    history_tail = [m for m in st.session_state.chat_history[-8:] if m["role"] in ("user", "assistant")]
    for h in history_tail:
        messages.append(h)

    user_prompt = format_user_prompt(question, context)
    messages.append({"role": "user", "content": user_prompt})

    with st.spinner("Thinking…"):
        answer = chat_completion(messages)

    # Simple post-processing: ensure at least one citation if context exists
    if context.strip() and "[" not in answer:
        # add a loose citation to first passage
        first = passages[0] if passages else None
        if first and first.get("title"):
            cite_title = first["title"]
            cite_src = first.get("source", "#") or "#"
            answer += f"\n\n[{cite_title}]({cite_src})"

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

# ------------------------------
# requirements.txt helper (printed in footer)
# ------------------------------
requirements_str = """
openai>=1.30.0
azure-search-documents>=11.6.0b1
azure-core>=1.30.0
streamlit>=1.35.0
""".strip()

# with st.expander("requirements.txt"):
#     st.code(requirements_str, language="text")
