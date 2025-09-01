
from __future__ import annotations
from typing import List, Optional, Tuple
import os
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain

# LLMs
# from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOllama


# ----------------------
# Model helpers
# ----------------------
def get_llm(provider: str = "openai",
            openai_model: str = "gpt-4o-mini",
            ollama_model: str = "llama3.1",
            temperature: float = 0.2):
    """
    Return a LangChain chat model based on provider settings.
    """
    provider = (provider or "openai").lower()
    if provider == "ollama":
        # No API key required; ensure Ollama is running locally.
        return ChatOllama(model=ollama_model, temperature=temperature)
    else:
        # Default to OpenAI. Requires OPENAI_API_KEY in env.
        return None


def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)


# ----------------------
# KB / Vectorstore
# ----------------------
def build_documents_from_df(df: pd.DataFrame) -> List[Document]:
    docs = []
    for _, row in df.iterrows():
        meta = {
            "title": str(row.get("title", "")),
            "year": int(row.get("year", 0)) if pd.notna(row.get("year")) else None,
            "genres": str(row.get("genres", ""))
        }
        page_content = f"Title: {meta['title']}\nYear: {meta['year']}\nGenres: {meta['genres']}\nPlot: {str(row.get('plot',''))}".strip()
        docs.append(Document(page_content=page_content, metadata=meta))
    return docs


def make_vectorstore(docs: List[Document], embeddings_model: str):
    emb = get_embeddings(embeddings_model)
    # Use a text splitter to keep chunks small, but since plots are short,
    # we can index as-is to keep metadata intact.
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, emb)


# ----------------------
# QA & Summarize Chains
# ----------------------
def build_retrieval_qa(llm, vectorstore: FAISS, k: int = 4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )
    return qa


def summarize_reviews(llm, reviews: List[str]) -> str:
    docs = [Document(page_content=rev.strip()) for rev in reviews if rev and rev.strip()]
    if not docs:
        return "No reviews provided."
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)
    summary = chain.run(docs)
    return summary


# ----------------------
# Recommendations
# ----------------------
def recommend_similar(
    vectorstore: FAISS,
    query_text: str,
    top_k: int = 5
):
    """
    Return list of (title, score, metadata) for nearest neighbors to query_text.
    Score is vector distance; lower is better.
    """
    docs_scores = vectorstore.similarity_search_with_score(query_text, k=top_k)
    results = []
    for doc, score in docs_scores:
        meta = doc.metadata or {}
        title = meta.get("title", "Unknown")
        results.append((title, float(score), meta))
    results.sort(key=lambda x: x[1])
    return results

