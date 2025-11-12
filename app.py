import streamlit as st
import google.generativeai as genai

# --- IMPORTS CORREGIDOS ---
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.vectorstores import FAISS
# ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- CONFIGURATION ---

# --- LISTA DE URLS "PURA" DE ESPAÑA ---
DOCS_URLS = [
    "https://developer.fiskaly.com/",
    "https://developer.fiskaly.com/api/",
    "https://developer.fiskaly.com/products/kassensichv-de",
    "https://developer.fiskaly.com/products/dsfinv-k-de",
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
# --- ARCHIVO DE TEXTO LOCAL PARA LA API ---
API_TEXT_FILE = "api_content.txt"
# --- CORREO DE SOPORTE ACTUALIZADO ---
SUPPORT_EMAIL = "dev-support@fiskaly.com"

# --- HELPER FUNCTIONS ---

@st.cache_resource(show_spinner="Loading and indexing documentation (this may take a moment)...")
def load_and_index_docs(api_key):
    """
    Loads docs from BOTH web URLs and the local text file,
    splits them, creates embeddings, and stores them in FAISS.
    
    Returns:
        A LangChain retriever object.
    """
    try:
        # 1. Load Documents from Web (las 22 guías)
        web_loader = WebBaseLoader(DOCS_URLS)
        docs_from_web = web_loader.load()
        
        # 2. Load Documents from Local Text File (el contenido de la API)
        text_loader = TextLoader(API_TEXT_FILE, encoding="utf-8")
        docs_from_text = text_loader.load()
        
        # 3. Combine all docs
        all_docs = docs_from_web + docs_from_text
        
        # 4. Split Documents
        # --- AJUSTE DE MÁXIMA PRECISIÓN ---
        # Trozos más pequeños (500) con superposición (100)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(all_docs)
        
        if not split_docs:
            st.error("Failed to load any documents. Check URLs and network access.")
            st.stop()
            
        # 5. Create Google Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # 6. Create Vector Store (FAISS)
        vector_store = FAISS.from_documents(split_docs, embeddings)
        
        # 7. Create Retriever
        # Búsqueda estándar (precisa) con k=6
        return vector_store.as_retriever(search_kwargs={"k": 6})
    
    except FileNotFoundError:
        st.error(f"Error: El archivo '{API_TEXT_FILE}' no se encontró.")
        st.write(f"Por favor, crea el archivo '{API_TEXT_FILE}' en tu repositorio de GitHub y pega el contenido de la documentación de la API de SIGN ES en él.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading or indexing documents: {e}")
        st.stop()


def get_stuff_chain(llm):
    """
    Creates the main document stuffing chain that answers user questions 
    based on retrieved context and chat history, following specific rules.
    """
    # --- PROMPT DEL SISTEMA ACTUALIZADO ---
    system_prompt = f"""
    Eres una máquina. Eres un bot de búsqueda y respuesta de documentación.
    Tu *única* función es encontrar fragmentos relevantes del <context> y presentárselos al usuario.
    Se te prohíbe usar cualquier conocimiento externo. No puedes responder ninguna pregunta usando información que no esté *explícitamente* escrita en el <context>.
    El usuario puede preguntar en español o inglés; responde en el idioma de la pregunta del usuario.

    Sigue estas reglas con absoluta precisión:
    1.  **Busca en el <context>:** Lee el <context> proporcionado.
    2.  **Encuentra la Respuesta:** Encuentra los fragmentos exactos que responden a la pregunta del usuario.
    3.  **Formula la Respuesta:** Formula una respuesta directa usando *solo* las palabras y hechos de esos fragmentos.
    4.  **Maneja la Información Faltante:** Si la respuesta no está en el <context>, o si el contexto está vacío, DEBES responder con el siguiente texto:
        "I'm sorry, I couldn't find an answer to your question in the documentation. For further assistance, please email our team at {SUPPORT_EMAIL} with as much detail as possible about your issue, and we'll get back to you as soon as possible."
        (Si el usuario preguntó en español, responde con: "Lo siento, no pude encontrar una respuesta a tu pregunta en la documentación. Para más ayuda, por favor envía un correo a nuestro equipo a {SUPPORT_EMAIL} con todos los detalles posibles sobre tu problema, y te responderemos lo antes posible.")
    5.  **NO ALUCINES:** No inventes hechos. Si el contexto menciona "JWT" y el usuario pregunta por "OAuth", DEBES decir que el contexto solo menciona "JWT" o que no puedes encontrar información sobre "OAuth". No inventes una respuesta bajo ninguna circunstancia.

    Aquí está el contexto:
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
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        return response.content
    except Exception as e:
        return f"Error generating summary: {e}"

# --- STREAMLIT APP ---

st.set_page_config(page_title="Doc Assistant", page_icon="🤖")
st.title("🤖 AI Documentation Assistant")
st.write("Ask questions about the Fiskaly developer documentation.")

# 1. Get API Key in Sidebar
with st.sidebar:
    st.header("Configuration")
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("API key loaded from secrets! 🤫")
    else:
        google_api_key = st.text_input(
            "Enter your Google Gemini API Key", 
            type="password"
        )
    if not google_api_key:
        st.info("Please enter your API key to start (or add it to your Streamlit secrets as GOOGLE_API_KEY).")
        st.stop()

# 2. Configure Gemini (globally for the chat model)
try:
    genai.configure(api_key=google_api_key)
except Exception as e:
    st.error(f"Failed to configure Google AI: {e}")
    st.stop()

# 3. Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.m.flash-preview-09-2025",
    google_api_key=google_api_key,
    temperature=0.0 # <-- PUESTO A 0.0 PARA CERO ALUCINACIONES
)

# 4. Load Retriever (cached)
retriever = load_and_index_docs(google_api_key)

#
