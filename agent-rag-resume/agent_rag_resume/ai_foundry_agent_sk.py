"""
Azure RAG Chat (Semantic Kernel) — Single-file Streamlit app using Azure OpenAI + Azure AI Search

Env Vars
  AZURE_OPENAI_ENDPOINT=...
  AZURE_OPENAI_API_KEY=...
  AZURE_OPENAI_API_VERSION=2024-08-01-preview
  AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

  AZURE_SEARCH_ENDPOINT=https://<your-search>.search.windows.net
  AZURE_SEARCH_API_KEY=...
  AZURE_SEARCH_INDEX_NAME=<index-name>
  AZURE_SEARCH_VECTOR_FIELD=content_vector           # default if not set
  AZURE_SEARCH_SEMANTIC_CONFIG=<optional-config>

Optional
  TOP_K=5
  MAX_TOKENS=800
  TEMPERATURE=0.1
  CONTEXT_CHAR_LIMIT=12000
"""

import os
import textwrap
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Azure Cognitive Search
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType, VectorQuery, VectorizedQuery

# Semantic Kernel (chat orchestration)
from semantic_kernel import Kernel
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions import KernelFunction
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
from semantic_kernel.contents import ChatHistory

# Azure OpenAI SDK (embeddings call for reliability)
try:
    from openai import AzureOpenAI
except ImportError as ex:
    raise RuntimeError(
        "openai package not available. Install it in the same environment:\n"
        "  python -m pip install 'openai>=1.30.0'"
    ) from ex


def get_env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


# Config
AZURE_OPENAI_ENDPOINT = get_env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = get_env("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_DEPLOYMENT = get_env("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = get_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

AZURE_SEARCH_ENDPOINT = get_env("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = get_env("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = get_env("AZURE_SEARCH_INDEX_NAME")
AZURE_SEARCH_VECTOR_FIELD = os.getenv("AZURE_SEARCH_VECTOR_FIELD", "text_vector")
AZURE_SEARCH_SEMANTIC_CONFIG = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "")

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
CONTEXT_CHAR_LIMIT = int(os.getenv("CONTEXT_CHAR_LIMIT", "12000"))

SYSTEM_TEMPLATE = """
You are a helpful enterprise assistant. Ground all answers strictly in the provided context.
Rules:
- If the answer is not in the context, say "I don't know based on the indexed documents." Do not fabricate.
- Prefer concise, direct answers.
- If listing steps, present them as short bullet points.
""".strip()

USER_TEMPLATE = """
User question:
{question}

Context (from retrieved documents):
{context}

Answer the user's question using ONLY the context above. If insufficient, say you don't know.
Provide citations like [{{title}}] for statements derived from specific passages.
""".strip()

# Azure AI Search client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
)

aoai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# Semantic Kernel setup
kernel = Kernel()
kernel.add_service(
    AzureChatCompletion(
        service_id="default",
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )
)
kernel.add_service(
    AzureTextEmbedding(
        service_id="embedding",
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    )
)

# After kernel services are added
plugins_dir = os.path.join(os.path.dirname(__file__), "plugins")
USER_PROMPT_PLUGIN = kernel.add_plugin(parent_directory=plugins_dir, plugin_name="UserPromptPlugin")
USER_PROMPT_FUNC = USER_PROMPT_PLUGIN["USER_PROMPT_TEMPLATE"]


def register_grounded_chat_plugin(k: Kernel) -> KernelFunction:
    """Register an SK plugin/function that uses SYSTEM_TEMPLATE and USER_TEMPLATE with parameters.

    Exposes plugin "grounded_chat" with function "answer_with_context(question, context)".
    """
    # Convert USER_TEMPLATE placeholders to SK variables
    sk_user_template = (
        "User question:\n{{$question}}\n\n"
        "Context (from retrieved documents):\n{{$context}}\n\n"
        "Answer the user's question using ONLY the context above. If insufficient, say you don't know.\n"
        "Provide citations like [{{title}}] for statements derived from specific passages."
    )

    # Compose a single prompt that includes system and user content
    template = f"""
                System:
                {SYSTEM_TEMPLATE}

                User:
                {sk_user_template}
                """.strip()

    svc = k.get_service("default")
    Settings = svc.get_prompt_execution_settings_class()
    exec_settings = Settings(
        service_id="default", max_tokens=MAX_TOKENS, temperature=TEMPERATURE
    )

    cfg = PromptTemplateConfig(
        template=template,
        input_variables=[
            {"name": "question", "description": "User question", "default": ""},
            {"name": "context", "description": "Retrieved grounding context", "default": ""},
        ],
        execution_settings={"default": exec_settings},
    )

    return k.add_function(
        plugin_name="grounded_chat",
        function_name="answer_with_context",
        prompt_template_config=cfg,
    )


# Register the grounded chat plugin once
_GROUND_FUNC = register_grounded_chat_plugin(kernel)


def embed_text(text: str) -> List[float]:
    resp = aoai_client.embeddings.create(model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT, input=text)
    if not resp or not getattr(resp, "data", None) or not resp.data:
        raise RuntimeError("Embedding response is empty.")
    vec = getattr(resp.data[0], "embedding", None)
    if not vec:
        raise RuntimeError("Embedding vector is None. Check embedding deployment name.")
    return vec


def search_index(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    embedding = embed_text(query)
    
    # fields are the vector fields in the search index
    vector_query = VectorizedQuery(vector=embed_text(query), k_nearest_neighbors=3, fields="text_vector")

    results = search_client.search(
        vector_queries=[vector_query],
        # Ensure these fields exist in the search index
        select=["id", "chunk", "title", "chunk_id"],
    )


    # vq = VectorQuery(vector=embedding, fields=AZURE_SEARCH_VECTOR_FIELD, 
    #                  k_nearest_neighbors=top_k
    #                  )
    # #vq.kind = "vector"
    # vq.vector = embedding
    # kwargs = {
    #     "top": top_k,
    #     "vector_queries": [vq],
    #     "select": ["id", "chunk", "title", "chunk_id"],
    #     "search_text": query,  # hybrid: vector + keyword
    #     #"kind": "vector"
    # }
    # if AZURE_SEARCH_SEMANTIC_CONFIG:
    #     kwargs.update({
    #         "query_type": QueryType.SEMANTIC,
    #         "semantic_configuration_name": AZURE_SEARCH_SEMANTIC_CONFIG,
    #     })
    # else:
    #     kwargs.update({"query_type": QueryType.SEMANTIC})

    # results = search_client.search(**kwargs)
    hits: List[Dict[str, Any]] = []
    for doc in results:
        d = dict(doc)
        hits.append({
            "id": d.get("id"),
            "chunk": d.get("chunk", d.get("content", "")),
            "title": d.get("title") or "Untitled",
            "chunk_id": d.get("chunk_id", ""),
        })
    return hits


def build_context(passages: List[Dict[str, Any]], limit_chars: int = CONTEXT_CHAR_LIMIT) -> str:
    parts, total = [], 0
    for p in passages:
        title = p.get("title") or "Untitled"
        chunk = p.get("chunk") or ""
        block = f"[title: {title}] \n{chunk.strip()}"
        if total + len(block) > limit_chars:
            block = block[: max(0, limit_chars - total)]
        parts.append(block)
        total += len(block)
        if total >= limit_chars:
            break
    return "\n\n---\n\n".join(parts)


def build_documents_block(passages: List[Dict[str, Any]], limit_chars: int = CONTEXT_CHAR_LIMIT) -> str:
    """Return an enumerated documents block suitable for the USER_PROMPT_TEMPLATE."""
    parts, total = [], 0
    for i, p in enumerate(passages, start=1):
        title = p.get("title") or "Untitled"
        cid = p.get("chunk_id", "")
        pid = p.get("id", "")
        chunk = (p.get("chunk") or "").strip()
        block = f"{i}. Title: {title}\n   Id: {pid}  ChunkId: {cid}\n   Content: {chunk}"
        if total + len(block) > limit_chars:
            block = block[: max(0, limit_chars - total)]
        parts.append(block)
        total += len(block)
        if total >= limit_chars:
            break
    return "\n\n".join(parts)


def sk_chat(question: str, context: str, documents: str) -> str:
    try:
        result = kernel.invoke(USER_PROMPT_FUNC, question=question, context=context, documents=documents)
        if hasattr(result, "__await__"):
            import asyncio
            result = asyncio.run(result)
        return getattr(result, "value", None) or getattr(result, "content", "") or ""
    except Exception:
        # Fallback to direct chat if plugin invocation fails
        chat_history = ChatHistory()
        chat_history.add_system_message(SYSTEM_TEMPLATE)
        chat_history.add_user_message(
            USER_TEMPLATE.format(question=question, context=context)
            + "\n\nDocuments (enumerated):\n"
            + documents
        )
        chat_svc = kernel.get_service("default")
        settings = kernel.get_prompt_execution_settings_from_service_id("default")
        settings.max_tokens = MAX_TOKENS
        settings.temperature = TEMPERATURE
        result = chat_svc.get_chat_message_content(chat_history, settings=settings)
        if hasattr(result, "__await__"):
            import asyncio
            result = asyncio.run(result)
        return getattr(result, "content", "") or ""


# Streamlit UI
st.set_page_config(page_title="Azure RAG Chat (Semantic Kernel)", page_icon="🧠", layout="wide")
st.title("🧠 Azure RAG Chat — Semantic Kernel + AI Search")

with st.expander("Configuration (env)"):
    st.write({
        "search_index": AZURE_SEARCH_INDEX_NAME,
        "vector_field": AZURE_SEARCH_VECTOR_FIELD,
        "semantic_config": AZURE_SEARCH_SEMANTIC_CONFIG or "(none)",
        "chat_model": AZURE_OPENAI_DEPLOYMENT,
        "embedding_model": AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        "top_k": TOP_K,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.subheader("Prompt Templates")
    st.code(textwrap.dedent(SYSTEM_TEMPLATE), language="markdown")
    st.code(textwrap.dedent(USER_TEMPLATE), language="markdown")

for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask a question about your indexed documents…")
if "prepopulated" not in st.session_state:
    st.session_state.prepopulated = True
    question = "Find me resume documents with terraform skills with azure cloud"
    
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Retrieving relevant passages…"):
        passages = search_index(question, top_k=TOP_K)

    context = build_context(passages)
    documents_block = build_documents_block(passages)

    with st.expander("Retrieved document chunks"):
        for i, p in enumerate(passages, start=1):
            st.markdown(f"**{i}. {p['title']}** ")
            st.write(textwrap.shorten(p["chunk"], width=500, placeholder=" …"))
            st.divider()

    with st.spinner("Thinking…"):
        answer = sk_chat(question, context, documents_block)

    if context.strip() and "[" not in answer and passages:
        cite_title = passages[0].get("title") or "Untitled"
        answer += f"\n\n[{cite_title}]"

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer[0])