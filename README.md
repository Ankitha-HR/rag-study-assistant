# RAG Study Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions from PDF documents
and cites the source file and page number. If the answer is not in the documents,
it says so instead of guessing.

## How it works
PDFs -> Chunking -> Embeddings -> Vector DB (ChromaDB)
Question -> Embedding -> Top-k similar chunks -> Prompt -> Gemini LLM -> Answer + sources

## Tech stack
Python, LangChain, ChromaDB, Sentence-Transformers (all-MiniLM-L6-v2), Google Gemini API, Google Colab

## Key features
- Loads PDFs and splits them into overlapping chunks (800 characters, 150 overlap)
- Semantic search over 176 chunks from 34 pages (2 sample research papers)
- Answers grounded in retrieved context, with source and page citations
- Refuses questions that are not in the documents (reduces hallucination)
- Retry with backoff, request timeout and model fallback for API errors (503/429)

## Results
Small sanity-check evaluation on 7 questions (5 answerable, 2 out-of-scope):
7/7 handled correctly, and the correct source document was retrieved for all 5 answerable questions.
(7 questions is a small test set. A larger evaluation is future work.)

## What I learned
- Debugged a failed answer by inspecting retrieved chunks, then increased top-k (4 to 6)
  and spelled out an acronym in the query
- Handled overloaded API servers with retries, timeouts and fallback models
- Kept API keys out of the code using Colab Secrets

## How to run
1. Open the notebook in Google Colab
2. Add your Gemini API key as a Colab Secret named GEMINI_API_KEY (turn on Notebook access)
3. Runtime -> Run all

## Future improvements
- Streamlit web interface
- Let users upload their own PDFs
- Hybrid search (keyword + vector) and re-ranking
- Larger evaluation set
