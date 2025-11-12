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
from langchain.chains.history_aware_retriever import create_history_aware_retriever
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
        # Trozos más pequeños (300) con superposición (50)
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
        # --- AJUSTE DE PRECISIÓN ---
        # k=4 para un contexto más pequeño y enfocado
        return vector_store.as_retriever(search_kwargs={"k": 4})
    
    except FileNotFoundError:
        st.error(f"Error: El archivo '{API_TEXT_FILE}' no se encontró.")
        st.write(f"Por favor, crea el archivo '{API_TEXT_FILE}' en tu repositorio de GitHub y pega el contenido de la documentación de la API de SIGN ES en él.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading or indexing documents: {e}")
        st.stop()

# --- SE RE-INTRODUCE ESTA FUNCIÓN ---
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
    model="gemini-2.5-flash-preview-09-2025",
    google_api_key=google_api_key,
    temperature=0.0 # <-- PUESTO A 0.0 PARA CERO ALUCINACIONES
)

# 4. Load Retriever (cached)
retriever = load_and_index_docs(google_api_key)

# 5. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! How can I help you with the Fiskaly documentation? Are you running into a specific error or bug?")
    ]

# 6. Display prior chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg.type):
        st.write(msg.content)

# 7. Get user input
user_prompt = st.chat_input("Ask your question here...")

if user_prompt:
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    with st.chat_message("user"):
        st.write(user_prompt)
        
    # --- CORRECCIÓN DE ESCALADA ---
    # Se eliminó "error" de la lista para evitar falsos positivos
    failed_keywords = ["didn't work", "not working", "did not help", "failed", "no funcionó", "no me ayudó"]
    is_failure_report = any(keyword in user_prompt.lower() for keyword in failed_keywords)
    
    if is_failure_report and len(st.session_state.messages) > 2:
        last_bot_answer = st.session_state.messages[-2].content
        last_user_question = st.session_state.messages[-3].content
        
        with st.chat_message("ai"):
            with st.spinner("I'm sorry to hear that. Generating a summary for support..."):
                summary = get_escalation_summary(
                    st.session_state.messages[:-1], 
                    last_user_question, # <-- Error de variable corregido
                    last_bot_answer,  # <-- Error de variable corregido
                    llm
                )
                escalation_response = f"""
                I'm sorry the previous solution didn't work.
                
                I can help you create a support ticket. Here is a summary of our conversation:
                
                ---
                **Support Ticket Summary:**
                {summary}
                ---
                
                You can forward this summary to **{SUPPORT_EMAIL}** to create a ticket.
                """
                st.write(escalation_response)
                st.session_state.messages.append(AIMessage(content=escalation_response))

    else:
        # --- Normal RAG Chat Logic ---
        with st.chat_message("ai"):
            with st.spinner("Searching the documentation..."):
                try:
                    # --- LÓGICA COMPLETA RE-INTRODUCIDA ---
                    
                    # 1. Crear el 'stuff chain' (este sí usa el historial para el chat)
                    stuff_chain = get_stuff_chain(llm)
                    
                    # 2. Crear el 'history aware retriever' (para entender el seguimiento)
                    retriever_chain = get_contextual_retriever_chain(retriever, llm)
                    
                    # 3. Crear la cadena de recuperación final.
                    conversational_rag_chain = create_retrieval_chain(
                        retriever_chain,
                        stuff_chain
                    )
                    
                    # 4. Invocar la cadena final
                    response = conversational_rag_chain.invoke({
                        "chat_history": st.session_state.messages[:-1],
                        "input": user_prompt
                    })
                    
                    # --- LÓGICA DE CITACIÓN DE FUENTES ---
                    
                    # 5. Extraer la respuesta y el contexto
                    answer = response['answer']
                    context_docs = response.get('context', [])
                    
                    is_failure_message = (SUPPORT_EMAIL in answer)
                    full_answer = answer # Empezar con la respuesta del bot
                    
                    # 6. Añadir fuentes si la respuesta NO es un mensaje de fallo
                    if context_docs and not is_failure_message:
                        sources = set()
                        for doc in context_docs:
                            if 'source' in doc.metadata:
                                source_url = doc.metadata['source']
                                # Mapear el archivo local de vuelta a su URL web
                                if source_url == API_TEXT_FILE:
                                    source_url = "https://developer.fiskaly.com/api/sign-es/v1"
                                
                                # Limpiar los enlaces de anclaje (#) para URLs más limpias
                                sources.add(source_url.split('#')[0])
                        
                        if sources:
                            source_list = "\n".join(f"- {s}" for s in sources)
                            # Añadir la citación (puedes cambiar "Source(s)" a "Fuentes:")
                            full_answer += f"\n\n**Source(s):**\n{source_list}"

                    # 7. Mostrar y guardar la respuesta completa
                    st.write(full_answer)
                    st.session_state.messages.append(AIMessage(content=full_answer))
                    
                except Exception as e:
                    error_msg = f"An error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append(AIMessage(content=error_msg))
