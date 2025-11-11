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
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(docs)
        
        # 3. Create Google Embeddings
        # 
        # This is lightweight because it's just an API call.
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # 4. Create Vector Store (FAISS)
        vector_store = FAISS.from_documents(split_docs, embeddings)
        
        # 5. Create Retriever
        return vector_store.as_retriever(search_kwargs={"k": 4})
    
    except Exception as e:
        # This will now clearly show the "quota" error if billing is not enabled
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

def get_conversational_rag_chain(retriever_chain, llm):
    """
    Creates the main RAG chain that answers user questions based on
    retrieved context and chat history, following specific rules.
    """
    system_prompt = f"""
    You are an expert assistant for the Fiskaly developer documentation.
    Your task is to answer user questions based *only* on the provided context.
    Follow these rules strictly:
    1.  **Base all answers on the context:** Do not use any outside knowledge.
    2.  **Be concise:** Provide a clear and direct answer.
    3.  **If the answer is not in the context:** You MUST state, "I could not find an answer in the documentation. You can reach out to {SUPPORT_EMAIL} for more help."
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
    temperature=0.1 
)

# 4. Load Retriever (cached)
# This function now requires the API key to create the embeddings
retriever = load_and_index_docs(google_api_key)

# 5. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! How can I help you? Are you running into a specific error or bug?")
    ]

# 6. Display prior chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg.type):
        st.write(msg.content)

# 7. Get user input
user_prompt = st.chat_input("Ask your question here...")

if user_prompt:
    # Add user message to history and display it
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    with st.chat_message("user"):
        st.write(user_prompt)
        
    # --- Core Escalation Logic ---
    failed_keywords = ["didn't work", "not working", "did not help", "failed", "error"]
    is_failure_report = any(keyword in user_prompt.lower() for keyword in failed_keywords)
    
    if is_failure_report and len(st.session_state.messages) > 2:
        # Find the user's last *real* question and the bot's last answer
        last_bot_answer = st.session_state.messages[-2].content
        last_user_question = st.session_state.messages[-3].content
        
        with st.chat_message("ai"):
            with st.spinner("I'm sorry to hear that. Generating a summary for support..."):
                summary = get_escalation_summary(
                    st.session_state.messages[:-1], # History before the "it failed" message
                    last_user_question,
                    last_bot_answer, # Corrected this
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
                    # 1. Create chains
                    retriever_chain = get_contextual_retriever_chain(retriever, llm)
                    conversational_rag_chain = get_conversational_rag_chain(retriever_chain, llm)
                    
                    # 2. Invoke chain
                    response = conversational_rag_chain.invoke({
                        "chat_history": st.session_state.messages[:-1], # History *before* new prompt
                        "input": user_prompt
                    })
                    
                    # 3. Display and save response
                    answer = response['answer']
                    st.write(answer)
                    st.session_state.messages.append(AIMessage(content=answer))
                    
                except Exception as e:
                    error_msg = f"An error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append(AIMessage(content=error_msg))
