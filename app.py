import streamlit as st
import google.generativeai as genai

# Updated, cleaner imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
# ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

# --- CONFIGURATION ---

# --- LISTA DE URLS ACTUALIZADA CON LAS 23 PÁGINAS ---
DOCS_URLS = [
    # URLs originales
    "https://developer.fiskaly.com/",
    "https://developer.fiskaly.com/api/",
    "https://developer.fiskaly.com/products/kassensichv-de",
    "https://developer.fiskaly.com/products/dsfinv-k-de",
    # Nuevas URLs de 'sign-es'
    "https://developer.fiskaly.com/sign-es/introduction",
    "https://developer.fiskaly.com/sign-es/glossary",
    "https://developer.fiskaly.com/sign-es/integration_process",
    "https://developer.fiskaly.com/sign-es/guide_new_customers",
    "https://developer.fiskaly.com/sign-es/guide_signde_customers",
    "https://developer.fiskaly.com/sign-es/electronic_certificate",
    "https://developer.fiskaly.com/sign-es/responsibility_declaration",
    "https://developer.fiskaly.com/sign-es/sendingfiles_verifactu",
    "https://developer.fiskaly.com/sign-es/invoicecompliance_verifactu",
    "https://developer.fiskaly.com/sign-es/storage_verifactu",
    "https://developer.fiskaly.com/sign-es/connectionloss_verifactu",
    "https://developer.fiskaly.com/sign-es/registration",
    "https://developer.fiskaly.com/sign-es/device_certificate",
    "https://developer.fiskaly.com/sign-es/optional_device_certificate",
    "https://developer.fiskaly.com/sign-es/sendingfiles",
    "https://developer.fiskaly.com/sign-es/invoicecompliance",
    "https://developer.fiskaly.com/sign-es/storage",
    "https://developer.fiskaly.com/sign-es/connectionloss",
    "https://developer.fiskaly.com/sign-es/digital_receipt"
]
SUPPORT_EMAIL = "support@mycompany.com"

# --- HELPER FUNCTIONS ---

@st.cache_resource(show_spinner="Loading and indexing documentation (this may take a moment)...")
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
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(docs)
        
        # 3. Create Google Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # 4. Create Vector Store (FAISS)
        vector_store = FAISS.from_documents(split_docs, embeddings)
        
        # 5. Create Retriever
        return vector_store.as_retriever(search_kwargs={"k": 5}) # Aumentado a 5 'chunks'
    
    except Exception as e:
        st.error(f"Error loading or indexing documents: {e}")
        st.stop()

def get_contextual_retriever_chain(retriever, llm):
    """
    Creates a chain that takes chat history and the latest user question,
    rephrases the question to be standalone, and retrieves relevant documents.
    """
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    return create_history_aware_retriever(llm, retriever, retriever_prompt)

def get_stuff_chain(llm):
    """
    Creates the main document stuffing chain that answers user questions 
    based on retrieved context and chat history, following specific rules.
    """
    system_prompt = f"""
    You are an expert assistant for the Fiskaly developer documentation.
    Your task is to answer user questions based *only* on the provided context.
    The user may ask in Spanish or English; answer in the language of the user's question.
    Follow these rules strictly:
    1.  **Base all answers on the context:** Do not use any outside knowledge.
    2.  **Be concise:** Provide a clear and direct answer.
    3.  **If the answer is not in the context:** You MUST state, "I could not find an answer in the documentation. You can reach out to {SUPPORT_EMAIL} for more help." (If the user asked in Spanish, say: "No pude encontrar una respuesta en la documentación. Puede contactar a {SUPPORT_EMAIL} para más ayuda.")
    4.  **Do not make up answers:** If the context is empty or irrelevant, follow rule 3.
    
    Here is the context:
    <context>
    {{context}}
    </context>
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    return create_stuff_documents_chain(llm, qa_prompt)

def get_escalation_summary(chat_history_messages, user_question, failed_solution, llm):
    """
    Generates a summary for the support team when a user indicates
    a solution failed.
    """
    history_str = "\n".join(
        [f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
         for msg in chat_history_messages]
    )
    summary_prompt = f"""
    A user is having trouble with a solution from our documentation.
    Please generate a concise summary of the conversation for a support ticket.
    Include the user's original question and the solution that failed.
    
    Conversation History:
    {history_str}
    
    User's original question (approximate): {user_question}
    Solution that failed (approximate): {failed_solution}
    
    Generate a summary for the support team:
    """
    try:
        response = llm.invoke([HumanMessage(
