import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/search"

st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Semantic Search Engine")

st.markdown(
    "Search internal documents using **MiniLM embeddings + FAISS**"
)

query = st.text_input(
    "Search query",
    placeholder="How to deploy Kubernetes models?"
)

top_k = st.slider(
    "Number of results",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("Search", type="primary"):

    if query.strip() == "":
        st.warning("Please enter a query.")
        st.stop()

    response = requests.get(
        API_URL,
        params={
            "q": query,
            "top_k": top_k
        }
    )

    if response.status_code != 200:
        st.error("API connection failed.")
        st.stop()

    data = response.json()

    st.success(f"{data['total_results']} documents found")

    for result in data["results"]:

        with st.container(border=True):

            col1, col2 = st.columns([4, 1])

            with col1:
                st.subheader(result["title"])
                st.write(result["text"])

            with col2:
                st.metric(
                    "Score",
                    f"{result['score']:.3f}"
                )