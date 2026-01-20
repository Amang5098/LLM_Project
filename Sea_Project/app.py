import streamlit as st
import os
import sys
import pickle
import chromadb
from chromadb.utils import embedding_functions

# Add parent directory to path to allow importing utils
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.llmclass import RAGGenerator
from utils.search import HybridSearch

# Page Configuration
st.set_page_config(page_title="Maritime RAG Assistant", page_icon="⚓", layout="centered")

# Title and Caption
st.title("⚓ Maritime AI Assistant")
st.caption("Ask questions about maritime regulations and procedures.")

# Initialize Resources (Cached)
@st.cache_resource
def load_resources():
    # Paths based on current file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "seamanuals")
    bm25_path = os.path.join(db_path, "bm25_retriever.pkl")
    
    client = chromadb.PersistentClient(path=db_path)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name="Sea-Database", embedding_function=embedding_func)
    
    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)
        
    search_engine = HybridSearch.get(collection, bm25_retriever)
    rag_generator = RAGGenerator(model_name='llama-3.2-1b:free')
    
    return search_engine, rag_generator

try:
    search, rag = load_resources()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display context if available (for assistant messages)
        if "context_docs" in message:
            with st.expander("View Retrieved Context Sources"):
                for doc in message["context_docs"]:
                    st.markdown(f"**Rank {doc['rank']} - Source: {doc.get('source', 'N/A')}**")
                    st.text(doc.get('content', ''))
                    st.divider()

# User Input
if prompt := st.chat_input("How do I battle fires aboard other boats?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Searching manuals..."):
            # 1. Retrieve Context
            docs = search(prompt)
            context_list = ["-> ".join([str(doc.get('source', 'Unknown')), str(doc.get('content', ''))]) for doc in docs]
            
            # 2. Generate Answer
            answer = rag.generate_answer(prompt, context_list)
            
            # Display Answer
            st.markdown(answer)
            
            # Display Context Dropdown
            with st.expander("View Retrieved Context Sources"):
                for doc in docs:
                    st.markdown(f"**Rank {doc['rank']} - Source: {doc.get('source', 'N/A')}**")
                    st.text(doc.get('content', ''))
                    st.divider()
            
            # Add assistant response to chat history (including context for persistence)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "context_docs": docs
            })
