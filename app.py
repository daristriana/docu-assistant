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
# --- SE ELIMINÓ EL IMPORT DE 'create_history_aware_retriever' ---
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
        return vector_store.as_retriever(search_kwargs={"k": 8, "search_type": "mmr"})
    
    except Exception as e:
        st.error(f"Error loading or indexing documents: {e}")
        st.stop()

# --- SE ELIMINÓ LA FUNCIÓN 'get_contextual_retriever_chain' ---

def get_stuff_chain(llm):
    """
    Creates the main document stuffing chain that answers user questions 
    based on retrieved context and chat history, following specific rules.
    """
    # --- PROMPT DEL SISTEMA MÁS ESTRICTO PARA EVITAR ALUCINACIONES ---
    system_prompt = f"""
    Eres una máquina. Eres un bot de búsqueda y respuesta de documentación.
    Tu *única* función es encontrar fragmentos relevantes del <context> y presentárselos al usuario.
    Se te prohíbe usar cualquier conocimiento externo. No puedes responder ninguna pregunta usando información que no esté *explícitamente* escrita en el <context>.
    El usuario puede preguntar en español o inglés; responde en el idioma de la pregunta del usuario.

    Sigue estas reglas con absoluta precisión:
    1.  **Busca en el <context>:** Lee el <context> proporcionado.
    2.  **Encuentra la Respuesta:** Encuentra los fragmentos exactos que responden a la pregunta del usuario.
    3.  **Formula la Respuesta:** Formula una respuesta directa usando *solo* las palabras y hechos de esos fragmentos.
    4.  **Maneja la Información Faltante:** Si la respuesta no está en el <context>, o si el contexto está vacío, DEBES decir: "No pude encontrar una respuesta en la documentación. Puede contactar a {SUPPORT_EMAIL} para más ayuda." (Usa la versión en español si el usuario preguntó en español).
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
    temperature=0.0 # <-- Puesto a 0.0 para reducir la creatividad/alucinación
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
        
    failed_keywords = ["didn't work", "not working", "did not help", "failed", "error", "no funcionó", "no me ayudó"]
    is_failure_report = any(keyword in user_prompt.lower() for keyword in failed_keywords)
    
    if is_failure_report and len(st.session_state.messages) > 2:
        last_bot_answer = st.session_state.messages[-2].content
        last_user_question = st.session_state.messages[-3].content
        
        with st.chat_message("ai"):
            with st.spinner("I'm sorry to hear that. Generating a summary for support..."):
                summary = get_escalation_summary(
                    st.session_state.messages[:-1], 
                    last_user_question,
                    last_bot_answer,
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
                    # --- LÓGICA SIMPLIFICADA Y CORREGIDA ---
                    
                    # 1. Crear el 'stuff chain' (este sí usa el historial para el chat)
                    stuff_chain = get_stuff_chain(llm)
                    
                    # 2. Crear la cadena de recuperación final.
                    # Pasamos el 'retriever' simple, no el que depende del historial.
                    conversational_rag_chain = create_retrieval_chain(
                        retriever, # <-- Esta es la simplificación clave
                        stuff_chain
                    )
                    
                    # 3. Invocar la cadena final
                    response = conversational_rag_chain.invoke({
                        "chat_history": st.session_state.messages[:-1],
                        "input": user_prompt
                    })
                    
                    # 4. Mostrar y guardar la respuesta
                    answer = response['answer']
                    st.write(answer)
                    st.session_state.messages.append(AIMessage(content=answer))
                    
                except Exception as e:
                    error_msg = f"An error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append(AIMessage(content=error_msg))
