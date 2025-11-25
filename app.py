import streamlit as st
import google.generativeai as genai

# --- IMPORTS ---
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- CONFIGURATION ---

# --- LISTA DE URLS ACTUALIZADA ---
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
    "https://developer.fiskaly.com/sign-es/digital_receipt",
    # URL NUEVA AGREGADA (Nota: El loader estándar leerá el texto visible de esta página, 
    # pero no hará clic automáticamente en los artículos dentro de ella)
    "https://support.fiskaly.com/hc/es/categories/7263083281692-SIGN-ES"
]

# --- ARCHIVO YAML LOCAL ---
# Asegúrate de que este archivo exista en la misma carpeta o repo
API_YAML_FILE = "oas.yaml"

# --- CORREO DE SOPORTE ---
SUPPORT_EMAIL = "dev-support@fiskaly.com"

# --- HELPER FUNCTIONS ---

@st.cache_resource(show_spinner="Loading and indexing documentation (this may take a moment)...")
def load_and_index_docs(api_key):
    """
    Loads docs from BOTH web URLs and the local YAML file.
    """
    try:
        # 1. Load Documents from Web
        web_loader = WebBaseLoader(DOCS_URLS)
        docs_from_web = web_loader.load()
        
        # 2. Load Documents from Local YAML File
        # Usamos TextLoader porque para RAG el YAML es simplemente texto estructurado.
        # Si el archivo es muy complejo, la IA igual entenderá la estructura de indentación.
        text_loader = TextLoader(API_YAML_FILE, encoding="utf-8")
        docs_from_yaml = text_loader.load()
        
        # 3. Combine all docs
        all_docs = docs_from_web + docs_from_yaml
        
        # 4. Split Documents
        # Mantenemos chunks de 300/50 para precisión granular
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, 
            chunk_overlap=50
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
        return vector_store.as_retriever(search_kwargs={"k": 5})
    
    except FileNotFoundError:
        st.error(f"Error: El archivo '{API_YAML_FILE}' no se encontró.")
        st.write(f"Por favor, sube el archivo '{API_YAML_FILE}' a tu directorio de trabajo.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading or indexing documents: {e}")
        st.stop()


def get_stuff_chain(llm):
    """
    Creates the chain aimed at reasoning over the context without hallucinating.
    """
    # --- PROMPT MEJORADO PARA RAZONAMIENTO ---
    system_prompt = f"""
    Actúa como un Ingeniero de Soporte Técnico Senior experto en la integración de Fiskaly SIGN ES.
    Tu objetivo es proporcionar respuestas precisas, lógicas y útiles basadas ESTRICTAMENTE en el contexto proporcionado.

    Instrucciones de Razonamiento:
    1.  **Analiza:** Lee detenidamente la pregunta del usuario y escanea el <context> proporcionado.
    2.  **Conecta:** Si la respuesta requiere información de dos partes diferentes del contexto (por ejemplo, un endpoint en el archivo YAML y una explicación conceptual en una guía web), debes sintetizar esa información para dar una respuesta completa.
    3.  **Razona:** Si el usuario pregunta "Cómo hago X", y el contexto explica los pasos A, B y C, explica el proceso lógico. No te limites a copiar y pegar fragmentos si puedes explicar el "flujo".
    4.  **Verifica:** Antes de responder, pregúntate: "¿Esta información está respaldada por el contexto?". Si no lo está, no la digas.

    Reglas de Seguridad (Anti-Alucinación):
    - NO inventes parámetros, endpoints o reglas que no aparezcan en el texto.
    - Si el contexto está vacío o no contiene la respuesta, DEBES responder EXCLUSIVAMENTE con el siguiente mensaje de error estándar:
      "Lo siento, no pude encontrar una respuesta precisa a tu pregunta en la documentación actual. Para investigar esto a fondo, por favor envía un correo a nuestro equipo a {SUPPORT_EMAIL} con los detalles de tu caso."
    - Si el usuario pregunta en inglés, traduce el mensaje de error anterior al inglés.

    Idioma:
    - Responde siempre en el mismo idioma en el que el usuario formuló la pregunta.

    Contexto disponible:
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
    history_str = "\n".join(
        [f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
         for msg in chat_history_messages]
    )
    summary_prompt = f"""
    A user is reporting that a solution failed. Generate a technical summary for the support team.
    
    History: {history_str}
    User Question: {user_question}
    Failed Solution: {failed_solution}
    
    Output strictly a concise summary of the technical issue and what was tried.
    """
    try:
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        return response.content
    except Exception as e:
        return f"Error generating summary: {e}"

# --- STREAMLIT APP ---

st.set_page_config(page_title="Doc Assistant", page_icon="🤖")
st.title("🤖 AI Documentation Assistant (Gemini 3.0)")
st.write("Ask questions about the Fiskaly developer documentation.")

# 1. Get API Key
with st.sidebar:
    st.header("Configuration")
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("API key loaded! 🚀")
    else:
        google_api_key = st.text_input("Enter Google Gemini API Key", type="password")
    
    if not google_api_key:
        st.info("Please enter your API key.")
        st.stop()

# 2. Configure Gemini
try:
    genai.configure(api_key=google_api_key)
except Exception as e:
    st.error(f"Failed to configure Google AI: {e}")
    st.stop()

# 3. Initialize LLM - CAMBIO A GEMINI 3 FLASH
# Ajustamos la temperatura a 0.1 para permitir un mínimo de creatividad en la 
# redacción (razonamiento) pero manteniéndonos pegados a los hechos.
llm = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview", 
    google_api_key=google_api_key,
    temperature=0.1
)

# 4. Load Retriever
retriever = load_and_index_docs(google_api_key)

# 5. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! I'm updated with the latest documentation (OAS & Support). How can I help you integrate?")
    ]

# 6. Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg.type):
        st.write(msg.content)

# 7. User Input
user_prompt = st.chat_input("Ask your question here...")

if user_prompt:
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    with st.chat_message("user"):
        st.write(user_prompt)
        
    # Escalation Logic
    failed_keywords = ["didn't work", "not working", "failed", "no funcionó", "error"]
    is_failure_report = any(keyword in user_prompt.lower() for keyword in failed_keywords)
    
    if is_failure_report and len(st.session_state.messages) > 2:
        last_bot_answer = st.session_state.messages[-2].content
        last_user_question = st.session_state.messages[-3].content
        
        with st.chat_message("ai"):
            with st.spinner("Generating support summary..."):
                summary = get_escalation_summary(
                    st.session_state.messages[:-1], 
                    last_user_question, last_bot_answer, llm
                )
                escalation_response = f"""
                I'm sorry that didn't work. Here is a summary for a support ticket to **{SUPPORT_EMAIL}**:
                
                **Ticket Summary:**
                {summary}
                """
                st.write(escalation_response)
                st.session_state.messages.append(AIMessage(content=escalation_response))
    else:
        # RAG Logic
        with st.chat_message("ai"):
            with st.spinner("Analyzing documentation..."):
                try:
                    stuff_chain = get_stuff_chain(llm)
                    conversational_rag_chain = create_retrieval_chain(retriever, stuff_chain)
                    
                    response = conversational_rag_chain.invoke({
                        "chat_history": st.session_state.messages[:-1],
                        "input": user_prompt
                    })
                    
                    answer = response['answer']
                    st.write(answer)
                    st.session_state.messages.append(AIMessage(content=answer))
                    
                except Exception as e:
                    st.error(f"Error: {e}")
