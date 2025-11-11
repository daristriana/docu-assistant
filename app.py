import streamlit as st
import google.generativeai as genai
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings  # <-- Uses the free model
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

# --- CONFIGURATION ---

DOCS_URLS = [
    "https://developer.fiskaly.com/",
    "https://developer.fiskaly.com/api/",
    "https://developer.fiskaly.com/products/kassensichv-de",
    "https://developer.fiskaly.com/products/dsfinv-k-de"
]
SUPPORT_EMAIL = "support@mycompany.com"

# --- HELPER FUNCTIONS ---

@st.cache_resource(show_spinner="Loading and indexing documentation (this may take a moment)...")
def load_and_index_docs(urls):  # <-- No API key needed here
    """
    Loads documentation from web URLs, splits it into chunks,
    creates embeddings using a FREE open-source model, and
    stores them in a FAISS vector store.
    
    Returns:
        A LangChain retriever object.
    """
    try:
        # 1. Load Documents
        loader = WebBaseLoader(urls)
        docs = loader.load()
        
        # 2. Split Documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(docs)
        
        # 3. Create Embeddings (Using the free model)
        # This model runs on the Streamlit server's CPU.
        # [Image of a RAG pipeline highlighting the embedding model as a separate, local component]
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Ensure it runs on CPU
        )
        
        # 4. Create Vector Store (FAISS)
        vector_store = FAISS.from_documents(split_docs, embeddings)
        
        # 5. Create Retriever
        return vector_store.as_retriever(search_kwargs={"k": 4})
    
    except Exception as e:
        st.error(f"Error loading or indexing documents: {e}")
        st.stop()
