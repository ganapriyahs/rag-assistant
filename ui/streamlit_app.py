import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Assistant", page_icon="🔎")
st.title("🔎 RAG Assistant")
st.caption("Ask a question. Answers are grounded in your ingested documents, generated locally via Ollama.")

# Health check on load so the user knows if the backend is down or busy
try:
    health = requests.get(f"{API_URL}/health", timeout=3)
    if health.status_code != 200:
        st.error("Backend is reachable but unhealthy.")
except requests.exceptions.RequestException:
    st.error("Can't reach the API, or it's busy. Make sure `uvicorn app.main:app --reload` is running, then refresh.")
    st.stop()

question = st.text_input("Your question:", placeholder="e.g. What is distillation?")

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

            if result["sources"]:
                st.subheader("Sources")
                for source in result["sources"]:
                    st.markdown(f"- `{source}`")
            else:
                st.caption("No sources: the documents didn't give a grounded answer.")

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")