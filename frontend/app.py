"""DocWhisperer Streamlit frontend.

Provides a chat interface for querying the RAG pipeline, document ingestion
controls, and RAGAS evaluation metrics display.
"""
import httpx
import streamlit as st

st.set_page_config(page_title="DocWhisperer", layout="wide", page_icon="📚")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    server_url = st.text_input("API Endpoint", value="http://127.0.0.1:8000")

    st.divider()
    st.header("📂 Document Ingestion")
    ingest_path = st.text_input("Directory path", placeholder="/path/to/docs")
    if st.button("🔄 Ingest Documents", use_container_width=True):
        if ingest_path.strip():
            with st.spinner("Ingesting documents..."):
                try:
                    resp = httpx.post(
                        f"{server_url}/ingest",
                        json={"path": ingest_path},
                        timeout=300.0,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.success(
                        f"✅ Processed {data['files_processed']} files → "
                        f"{data['chunks_created']} chunks"
                    )
                    if data.get("errors"):
                        for err in data["errors"]:
                            st.warning(err)
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")
        else:
            st.warning("Enter a directory path first")

    st.divider()
    st.header("📊 Evaluation")
    if st.button("Run RAGAS Eval", use_container_width=True):
        with st.spinner("Running evaluation..."):
            try:
                resp = httpx.get(f"{server_url}/eval", timeout=120.0)
                resp.raise_for_status()
                data = resp.json()
                if "message" in data:
                    st.info(data["message"])
                else:
                    st.session_state["eval_metrics"] = data
                    st.success("Evaluation complete!")
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

# --- Main Area ---
st.title("📚 DocWhisperer")
st.caption("Agentic RAG Pipeline for Technical Documentation")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src.get('id', 'unknown')}** (score: {src.get('score', 0):.3f})")
                    if src.get("text"):
                        st.text(src["text"][:300])

# Chat input
if prompt := st.chat_input("Ask a question about your documentation..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = httpx.post(
                    f"{server_url}/query",
                    json={"query": prompt},
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()

                answer = data.get("answer", "No answer received")
                sources = data.get("sources", [])

                st.markdown(answer)
                if sources:
                    with st.expander("📄 Sources"):
                        for src in sources:
                            st.markdown(f"- **{src.get('id', 'unknown')}** (score: {src.get('score', 0):.3f})")
                            if src.get("text"):
                                st.text(src["text"][:300])

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception as exc:
                error_msg = f"Error: {exc}"
                st.error(error_msg)
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": error_msg,
                })

# --- Metrics Panel ---
if st.session_state.get("eval_metrics"):
    st.divider()
    st.subheader("📊 RAGAS Evaluation Metrics")
    metrics = st.session_state["eval_metrics"]
    cols = st.columns(4)
    cols[0].metric("Faithfulness", f"{metrics.get('faithfulness', 0):.3f}")
    cols[1].metric("Answer Relevancy", f"{metrics.get('answer_relevancy', 0):.3f}")
    cols[2].metric("Context Precision", f"{metrics.get('context_precision', 0):.3f}")
    cols[3].metric("Queries Evaluated", metrics.get("evaluated_queries", 0))
