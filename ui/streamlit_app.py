import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Assistant", page_icon="🔎")
st.title("🔎 RAG Assistant")
st.caption("Ask a question. Answers are grounded in your ingested documents, generated locally via Ollama.")

# Health check on load so the user knows if the backend is down
try:
    health = requests.get(f"{API_URL}/health", timeout=3)
    if health.status_code != 200:
        st.error("Backend is reachable but unhealthy.")
except requests.exceptions.ConnectionError:
    st.error("Can't reach the API. Make sure `uvicorn app.main:app --reload` is running.")
    st.stop()

question = st.text_input("Your question:", placeholder="e.g. What is model drift?")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving context and generating answer..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question, "top_k": 3},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Sources")
            for source in result["sources"]:
                st.markdown(f"- `{source}`")

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")