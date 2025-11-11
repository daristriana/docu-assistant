import streamlit as st
import google.generativeai as genai

# Updated, cleaner imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
# ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain # <-- We will use this
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

@st.cache_resource(show_spinner="Loading and indexing documentation...")
def load_and_index_docs(api_key):
    """
    Loads docs, splits them, creates embeddings using the GOOGLE API,
    and stores them in FAISS.
    
    Returns:
        A LangChain retriever object.
    """
    try:
        # 1. Load Documents
        loader = WebBaseLoader(DOCS_URLS)
        docs = loader.load()
        
        # 2. Split Documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=20
