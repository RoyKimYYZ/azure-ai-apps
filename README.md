# Repo Overview

A collection of demos of RAG scenarios and techniques. Purpose is to showcase to business leaders and project managers capabilities of chatting and asking about corporate documents and knowledge. 

Azure AI Search is used for indexing, storing and keyword and vector querying the corporate documents.

Azure Open AI models such as embeddings models to vectorize content for indexing and query. This enabled searching documents based on semantic meaning rather than relying on keyword based searching.

Azure Open AI models such as gpt-4o is used process search results of documents and formulize a natural language response to the original human query.

### \streamlit-chat


* Web front end ui with chatbot style interaction
* Key configuration
    * LLM model endpoint such as Azure Open AI gpt-4o
    * search_resumes_api.py endpoint (localhost:5000)

### \rag-app-resumes

Back end apis to query against AI Search
* [create_skillset_run_indexrv2.py](./rag-app-resumes/create_skillset_run_indexrv2.py) Script to create data source, chunking skillset, create indexer and run indexer
 * [create_search_index_resume.py](rag-app-resumes/create_search_index_resume.py) - creates search index called resume-index
