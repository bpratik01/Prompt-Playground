# playground.py
import streamlit as st
import os
from typing import List, Dict
import openai
import logging
import time
from pinecone import Pinecone, PineconeException
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory # Conceptually useful
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Core LangChain/RAG Functions (Adapted from main.py) ---

# Default Prompt Template
DEFAULT_PROMPT_TEMPLATE = """You are Mia, Mobio Solutions' friendly AI assistant. CRITICAL: KEEP ALL RESPONSES BETWEEN 100-150 WORDS MAXIMUM. Be concise.

IMPORTANT: The user has already seen a greeting message from the UI. DO NOT GREET the user. Start responding directly to their first message.

CONVERSATION FLOW:
1. Respond directly to the user's query based on the provided Context.
2. If the user asks about services: Provide a VERY brief overview (2-3 sentences max) focusing on key areas like Microsoft Solutions, AI/ML, Web/Mobile Dev, Cloud.
3. Answer specific questions concisely based *only* on the provided Context.
4. If info isn't in the Context, say so politely.

RESPONSE LENGTH - CRITICAL:
- STRICTLY adhere to the 100-150 word limit. Shorter is better.
- Be direct. Remove redundant information and filler phrases.

BEHAVIOR INSTRUCTIONS:
- Friendly, conversational tone.
- Mention Microsoft/AI expertise ONLY if directly relevant to the user's query or when briefly listing services.
- Only answer based on Context.

CONTACT INFORMATION TO SHARE ONLY WHEN ASKED:
Email: contact@mobiosolutions.com
Phone: +1 (512) 861-5446

Previous conversation:
{chat_history}

Context:
{context}

Question: {question}

Answer (MAX 150 words, DO NOT GREET, follow conversation flow):"""

def format_docs(docs):
    """Format documents with their page content and metadata."""
    formatted_docs = []
    for doc in docs:
        content = doc.page_content
        if hasattr(doc, 'metadata') and doc.metadata:
            if 'title' in doc.metadata:
                content = f"Title: {doc.metadata['title']}\n{content}"
            if 'source' in doc.metadata:
                content = f"{content}\nSource: {doc.metadata['source']}"
        formatted_docs.append(content)
    return "\n\n---\n\n".join(formatted_docs)

def format_chat_history(chat_history_messages: List[Dict]) -> str:
    """Format chat history from Streamlit session state for the prompt."""
    formatted_history = ""
    for message in chat_history_messages:
        if message["role"] == "user":
            formatted_history += f"Human: {message['content']}\n"
        elif message["role"] == "assistant":
            formatted_history += f"AI: {message['content']}\n"
    return formatted_history

def create_rag_chain(llm, retriever, chat_prompt_template):
    """Creates the RAG chain using initialized components."""
    prompt_from_template = ChatPromptTemplate.from_template(chat_prompt_template)

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
            chat_history=lambda x: format_chat_history(x["chat_history_messages"])
        )
        | prompt_from_template # Use the template string directly
        | llm
        | StrOutputParser()
    )
    return rag_chain


# --- Streamlit UI ---

st.set_page_config(layout="wide")
st.title(" RAG Chatbot Prompt Playground")

# --- Define Columns ---
col1, col2 = st.columns([0.3, 0.7], gap="large") # 30% for col1, 70% for col2

# --- Column 1: Configuration and Prompt Editing ---
with col1:
    st.header("Configuration")

    openai_api_key = st.text_input("OpenAI API Key", type="password", key="openai_key", value=st.session_state.get("openai_api_key", ""))
    pinecone_api_key = st.text_input("Pinecone API Key", type="password", key="pinecone_key", value=st.session_state.get("pinecone_api_key", ""))
    pinecone_index_name = st.text_input("Pinecone Index Name", key="pinecone_index", value=st.session_state.get("pinecone_index_name", "ms-knowledge-base"))

    st.header("System Prompt")
    current_prompt = st.text_area(
        "Edit the chatbot's system prompt:",
        height=350,  # Adjust height as needed for the column
        key="prompt_editor",
        value=st.session_state.get("current_prompt", DEFAULT_PROMPT_TEMPLATE)
    )

    if st.button("Apply Changes & Restart Chat"):
        # Validate inputs
        if not openai_api_key:
            st.warning("Please enter your OpenAI API Key.")
        elif not pinecone_api_key:
            st.warning("Please enter your Pinecone API Key.")
        elif not pinecone_index_name:
            st.warning("Please enter the Pinecone Index Name.")
        elif not current_prompt:
            st.warning("Please enter a system prompt.")
        else:
            # Store credentials and prompt in session state
            st.session_state.openai_api_key = openai_api_key
            st.session_state.pinecone_api_key = pinecone_api_key
            st.session_state.pinecone_index_name = pinecone_index_name
            st.session_state.current_prompt = current_prompt

            # Clear previous chat and initialized objects
            st.session_state.messages = []
            st.session_state.rag_chain = None
            st.session_state.initialized = False # Reset initialization flag

            # Attempt to initialize components
            try:
                with st.spinner("Initializing..."):
                    # Set environment variables implicitly
                    os.environ["OPENAI_API_KEY"] = openai_api_key
                    os.environ["PINECONE_API_KEY"] = pinecone_api_key

                    # Initialize Pinecone
                    st.session_state.pc = Pinecone(api_key=pinecone_api_key)
                    # Get list of indexes
                    index_list = st.session_state.pc.list_indexes()
                    # Check if the specified index exists
                    if pinecone_index_name not in [index.name for index in index_list]:
                       st.error(f"Pinecone index '{pinecone_index_name}' not found.")
                       st.stop()

                    # Initialize embedding model
                    st.session_state.embedding_model = OpenAIEmbeddings(
                        model="text-embedding-3-small",
                        openai_api_key=openai_api_key
                    )

                    # Initialize vector store and retriever
                    st.session_state.vectorstore = PineconeVectorStore(
                        index_name=pinecone_index_name,
                        embedding=st.session_state.embedding_model,
                        pinecone_api_key=pinecone_api_key
                    )
                    st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 8})

                    # Initialize LLM
                    st.session_state.llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        temperature=0,
                        request_timeout=60,
                        max_retries=3,
                        openai_api_key=openai_api_key
                    )

                    # Create the RAG chain
                    st.session_state.rag_chain = create_rag_chain(
                        st.session_state.llm,
                        st.session_state.retriever,
                        st.session_state.current_prompt
                    )

                    st.session_state.initialized = True
                    st.success("Chatbot initialized!")
                    logger.info("Chatbot initialized/re-initialized.")
                    st.rerun() # Rerun to update the chat column state

            except PineconeException as pe:
                st.error(f"Pinecone Error: {pe}. Check API key and index name.")
                logger.error(f"Pinecone initialization failed: {pe}", exc_info=True)
                st.session_state.initialized = False
            except openai.AuthenticationError:
                st.error("OpenAI Authentication Error: Invalid API Key.")
                logger.error("OpenAI Authentication Error.", exc_info=True)
                st.session_state.initialized = False
            except Exception as e:
                st.error(f"An error occurred during initialization: {e}")
                logger.error(f"Initialization failed: {e}", exc_info=True)
                st.session_state.initialized = False

# --- Column 2: Chat Area ---
with col2:
    st.header("Chat")

    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat Messages Container
    chat_container = st.container(height=500) # Set a fixed height for scrolling

    with chat_container:
        # Display past messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat Input Area (placed below the message container)
    if not st.session_state.get("initialized", False):
        st.info("Please configure the settings on the left and click 'Apply Changes & Restart Chat' to begin.")
    else:
        # Get user input using st.chat_input
        if prompt := st.chat_input("What is your question?"):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message immediately (inside the container logic)
            with chat_container: # Need to redraw inside container
                 with st.chat_message("user"):
                     st.markdown(prompt)

            # Generate and display AI response
            with chat_container: # Redraw inside container
                 with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    try:
                        with st.spinner("Thinking..."):
                             # Prepare input for the chain
                             chain_input = {
                                 "question": prompt,
                                 "chat_history_messages": st.session_state.messages[:-1]
                             }
                             response = st.session_state.rag_chain.invoke(chain_input)
                             full_response = response

                        message_placeholder.markdown(full_response)

                    except Exception as e:
                        logger.error(f"Error during RAG chain invocation: {e}", exc_info=True)
                        st.error(f"Sorry, an error occurred: {e}")
                        full_response = "Error generating response."

            # Add AI response to history (outside the "with" blocks, but still within the "if prompt")
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Rerun to ensure the chat container updates correctly after adding messages
            st.rerun()


    # Optional: Clear chat button (place it logically, maybe below input or header)
    if st.session_state.get("initialized", False) and len(st.session_state.messages) > 0:
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun() # Rerun to clear the chat display