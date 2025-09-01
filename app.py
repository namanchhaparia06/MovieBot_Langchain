
import os
import json
from pathlib import Path
from typing import List

import streamlit as st
import pandas as pd

from chains import (
    get_llm,
    get_embeddings,
    build_documents_from_df,
    make_vectorstore,
    build_retrieval_qa,
    summarize_reviews,
    recommend_similar,
)

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DEFAULT_MOVIE_CSV = DATA_DIR / "sample_movies.csv"
DEFAULT_REVIEWS_JSONL = DATA_DIR / "reviews.jsonl"


# ----------------------
# Page config & styles
# ----------------------
st.set_page_config(
    page_title="🎬 Movie Recommender Bot",
    page_icon="🎬",
    layout="wide",
)
st.markdown(
    """
    <style>
    .big-title { font-size: 2rem; font-weight: 800; margin-bottom: 0.25rem; }
    .subtle { color: #6b7280; }
    .card { background: #ffffff; border-radius: 16px; padding: 1rem 1.25rem; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
    .pill { padding: 2px 8px; background: #f3f4f6; border-radius: 999px; margin-right: 6px; display: inline-block; font-size: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------
# Sidebar: settings
# ----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("LLM Provider", ["OpenAI", "Ollama"])
    if provider == "OpenAI":
        openai_model = st.text_input("OpenAI model", value="gpt-4o-mini")
        ollama_model = None
    else:
        openai_model = None
        ollama_model = st.text_input("Ollama model", value="llama3.1")
        st.caption("Make sure Ollama is running locally and the model is pulled (e.g., `ollama run llama3.1`).")
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.2, 0.1)

    st.divider()
    st.subheader("Embeddings")
    emb_model = st.text_input("SentenceTransformer", value="sentence-transformers/all-MiniLM-L6-v2")
    top_k = st.slider("Top-K (similarity search)", 2, 10, 5, 1)

    st.divider()
    st.subheader("Extend KB")
    user_csv = st.file_uploader("Upload extra movies CSV", type=["csv"])
    if user_csv is not None:
        st.session_state["user_csv"] = user_csv

    st.divider()
    st.caption("Tip: Set OPENAI_API_KEY in your environment if using OpenAI.")


# ----------------------
# Data loading & vectorstore (cached)
# ----------------------
@st.cache_resource(show_spinner=True)
def load_base_df() -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_MOVIE_CSV)
    return df

@st.cache_resource(show_spinner=True)
def load_reviews() -> List[str]:
    reviews = []
    if DEFAULT_REVIEWS_JSONL.exists():
        for line in open(DEFAULT_REVIEWS_JSONL, "r", encoding="utf-8"):
            try:
                obj = json.loads(line.strip())
                if obj.get("review"):
                    reviews.append(obj["review"])
            except Exception:
                continue
    return reviews

@st.cache_resource(show_spinner=True)
def build_store(df: pd.DataFrame, emb_model_name: str):
    docs = build_documents_from_df(df)
    vs = make_vectorstore(docs, emb_model_name)
    return vs

def get_df_with_user_upload() -> pd.DataFrame:
    df = load_base_df()
    if "user_csv" in st.session_state and st.session_state["user_csv"] is not None:
        try:
            user_df = pd.read_csv(st.session_state["user_csv"])
            # minimal sanity: enforce required columns if present
            needed = {"title","year","genres","plot"}
            missing = needed - set(c.lower() for c in user_df.columns)
            # try to normalize column names
            if missing:
                rename_map = {c: c.lower() for c in user_df.columns}
                user_df.rename(columns=rename_map, inplace=True)
                missing = needed - set(user_df.columns)
            if not missing:
                user_df = user_df[list(["title","year","genres","plot"])]
                df = pd.concat([df, user_df], ignore_index=True)
            else:
                st.warning(f"Uploaded CSV is missing columns: {missing}. Expected: {needed}")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
    return df


# ----------------------
# Header
# ----------------------
st.markdown('<div class="big-title">🎬 Movie Recommender Bot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Answer movie questions, summarize reviews, and get similar-title recommendations.</div>', unsafe_allow_html=True)
st.write("")

# Build LLM
llm = get_llm(
    provider=provider,
    openai_model=openai_model or "gpt-4o-mini",
    ollama_model=ollama_model or "llama3.1",
    temperature=temperature
)

# Build / update vectorstore
base_plus_user_df = get_df_with_user_upload()
vectorstore = build_store(base_plus_user_df, emb_model)


# ----------------------
# Tabs: Ask | Summarize Reviews | Recommend
# ----------------------
tab1, tab2, tab3 = st.tabs(["💬 Ask about movies", "📝 Summarize reviews", "🔍 Recommend similar"])

with tab1:
    st.markdown("#### Ask anything")
    query = st.text_input("Your question", placeholder="e.g., Movies like Inception but lighter; or What is Parasite about?")
    if st.button("Answer", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            qa = build_retrieval_qa(llm, vectorstore, k=top_k)
            with st.spinner("Thinking..."):
                result = qa.invoke({"query": query})
            st.markdown("##### Answer")
            st.write(result.get("result","(No answer)"))

            # Sources
            src_docs = result.get("source_documents", []) or []
            if src_docs:
                with st.expander("Sources"):
                    for d in src_docs:
                        meta = d.metadata or {}
                        title = meta.get("title","Unknown")
                        year = meta.get("year", "")
                        genres = meta.get("genres","")
                        st.markdown(f"- **{title}** ({year}) — _{genres}_")

with tab2:
    st.markdown("#### Summarize reviews")
    st.caption("Paste multiple reviews (one per line) or upload a JSONL like `data/reviews.jsonl`.")
    colA, colB = st.columns(2)
    with colA:
        default_reviews = load_reviews()
        default_text = "\n".join(default_reviews[:6])
        reviews_text = st.text_area("Reviews (one per line)", value=default_text, height=220)
    with colB:
        upload = st.file_uploader("Upload JSONL of reviews", type=["jsonl"], key="rev_upload")
        reviews_from_file: List[str] = []
        if upload is not None:
            try:
                for line in upload.getvalue().decode("utf-8").splitlines():
                    obj = json.loads(line.strip())
                    if obj.get("review"):
                        reviews_from_file.append(obj["review"])
            except Exception as e:
                st.error(f"Couldn't parse JSONL: {e}")

    reviews_list = [r.strip() for r in (reviews_text.splitlines() if reviews_text else []) if r.strip()]
    if reviews_from_file:
        reviews_list.extend(reviews_from_file)

    if st.button("Summarize", type="primary", use_container_width=True):
        if not reviews_list:
            st.warning("Provide at least one review via text area or upload.")
        else:
            with st.spinner("Summarizing..."):
                summary = summarize_reviews(llm, reviews_list)
            st.markdown("##### Summary")
            st.write(summary)

with tab3:
    st.markdown("#### Recommend similar titles")
    mode = st.radio("Recommendation mode", ["By title", "By description"], horizontal=True)
    query_text = ""
    if mode == "By title":
        # Offer a select box using known titles
        options = sorted(base_plus_user_df["title"].astype(str).unique().tolist())
        seed_title = st.selectbox("Pick a title", options)
        # Fetch that title's plot for the query
        row = base_plus_user_df[base_plus_user_df["title"] == seed_title].iloc[0]
        query_text = f"{row['title']} ({row['year']}), genres: {row['genres']}. Plot: {row['plot']}"
    else:
        query_text = st.text_area("Describe what you're in the mood for", placeholder="e.g., A heartwarming sports drama with father-daughter theme.")
    top_k_rec = st.slider("How many recommendations?", 3, 10, 5, 1)

    if st.button("Recommend", type="primary", use_container_width=True):
        if not query_text.strip():
            st.warning("Please provide a title or description.")
        else:
            with st.spinner("Finding similar titles..."):
                recs = recommend_similar(vectorstore, query_text, top_k=top_k_rec + 1)
            if mode == "By title" and recs:
                # Filter out the seed title itself if it appears
                seed = seed_title.lower().strip()
                recs = [r for r in recs if r[0].lower().strip() != seed]
                recs = recs[:top_k_rec]

            if not recs:
                st.info("No recommendations found.")
            else:
                st.markdown("##### Recommendations")
                for (title, dist, meta) in recs:
                    year = meta.get("year","")
                    genres = meta.get("genres","")
                    with st.container():
                        st.markdown(f"**{title}** ({year})")
                        st.caption(f"{genres}")
                        st.progress(max(0.0, min(1.0, 1.0/(1.0+dist))), text="Similarity (higher is better)")

                # Ask the LLM for a justification once, given the list
                titles_list = ", ".join([t for (t, _, _) in recs])
                prompt = f"User wanted: {query_text}\nRecommended: {titles_list}\nExplain briefly why these fit, in 3-5 bullet points."
                try:
                    resp = llm.invoke(prompt)
                    if hasattr(resp, "content"):
                        st.markdown("###### Why these?")
                        st.write(resp.content)
                except Exception:
                    pass

st.write("")
st.caption("Built with LangChain, FAISS, SentenceTransformers, and Streamlit.")
