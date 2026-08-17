import streamlit as st
import requests

st.set_page_config(page_title="Adaptive Token-Efficient AI", layout="wide")
st.title("🧠 Adaptive Token-Efficient AI Platform")
st.caption("A production-grade RAG engine prioritizing maximum answer quality with minimum tokens.")

API_URL = "http://localhost:8000"

with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Process & Index"):
        if uploaded_files:
            with st.spinner("Extracting structure, parsing tables, and embedding..."):
                files_data = [("files", (file.name, file.getvalue(), "application/pdf")) for file in uploaded_files]
                try:
                    response = requests.post(f"{API_URL}/upload", files=files_data)
                    if response.status_code == 200:
                        st.success("Indexed successfully!")
                        st.json(response.json())
                    else:
                        st.error("Upload failed.")
                except Exception as e:
                    st.error("Backend API is not running. Please start the FastAPI server.")
        else:
            st.warning("Please upload a file first.")

st.header("💬 Reasoning Interface")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a complex multi-document question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Planning, Retrieving, Reranking, & Compressing Context..."):
            try:
                res = requests.post(f"{API_URL}/ask", json={"query": prompt, "mock_mode": False}).json()
                answer = res.get("answer", "Error generating answer.")
                st.markdown(answer)
                
                with st.expander("📊 Token Efficiency & Observability Telemetry"):
                    st.json(res.get("telemetry", {}))
                    
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("Cannot connect to Backend API. Is FastAPI running on port 8000?")
