# ============================================================
#   RAG CHATBOT — Day 2: Streamlit Web App
#   Run with: streamlit run app.py
#
#   What's new vs Day 1:
#   - Beautiful web UI instead of terminal
#   - Drag & drop PDF upload
#   - Chat interface with message history
#   - Confidence scores shown per answer
#   - Expandable source viewer
#   - PDF stats in sidebar
# ============================================================

import os
import PyPDF2
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


# ── PAGE CONFIG (must be first streamlit call) ────────────────
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide",                # Use full screen width
    initial_sidebar_state="expanded",
)


# ── HELPER: Read PDF ──────────────────────────────────────────
def read_pdf(uploaded_file):
    """Reads an uploaded PDF and returns text + page count."""
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text, len(reader.pages)


# ── HELPER: Split into chunks ─────────────────────────────────
def split_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_text(text)


# ── CACHED: Load embedding model (only loads once!) ───────────
# @st.cache_resource tells Streamlit: load this once, reuse forever
# Without this, model would reload every time user asks a question!
@st.cache_resource
def load_embedding_model():
    """Loads HuggingFace embedding model. Cached — loads only once."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


# ── CACHED: Load QA model (only loads once!) ──────────────────
@st.cache_resource
def load_qa_model():
    """Loads distilbert QA model. Cached — loads only once."""
    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


# ── HELPER: Create vector store ───────────────────────────────
def create_vector_store(chunks, embeddings):
    return FAISS.from_texts(chunks, embeddings)


# ── HELPER: Answer a question ─────────────────────────────────
def ask_question(vector_store, tokenizer, model, question):
    # Find top 3 relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(question)
    context = " ".join([doc.page_content for doc in relevant_docs])

    # Tokenize and run model
    inputs = tokenizer(
        question, context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract answer
    start_idx = torch.argmax(outputs.start_logits)
    end_idx   = torch.argmax(outputs.end_logits) + 1
    answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    # Confidence score
    start_score = torch.softmax(outputs.start_logits, dim=1).max().item()
    end_score   = torch.softmax(outputs.end_logits,   dim=1).max().item()
    confidence  = round((start_score + end_score) / 2 * 100, 1)

    return answer, confidence, relevant_docs


# ══════════════════════════════════════════════════════════════
#   STREAMLIT UI STARTS HERE
# ══════════════════════════════════════════════════════════════

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 RAG PDF Chatbot")
    st.markdown("Upload any PDF and ask questions about it!")
    st.divider()

    # File uploader widget
    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],               # Only allow PDF files
        help="Max size: 200MB",
    )

    # Process button — only show if file is uploaded
    if uploaded_file:
        if st.button("⚡ Process PDF", type="primary", use_container_width=True):

            # Show spinner while processing
            with st.spinner("Reading PDF..."):
                text, num_pages = read_pdf(uploaded_file)

            with st.spinner("Splitting into chunks..."):
                chunks = split_into_chunks(text)

            with st.spinner("Loading AI models..."):
                embeddings = load_embedding_model()
                tokenizer, model = load_qa_model()

            with st.spinner("Building vector database..."):
                vector_store = create_vector_store(chunks, embeddings)

            # Save to session state so it persists across interactions
            # session_state = Streamlit's way of storing data between reruns
            st.session_state.vector_store = vector_store
            st.session_state.tokenizer    = tokenizer
            st.session_state.model        = model
            st.session_state.chunks       = chunks
            st.session_state.num_pages    = num_pages
            st.session_state.filename     = uploaded_file.name
            st.session_state.messages     = []  # Reset chat history

            st.success("✅ PDF processed! Start asking questions.")

    # Show PDF stats if processed
    if "vector_store" in st.session_state:
        st.divider()
        st.markdown("**📊 PDF Stats**")
        st.markdown(f"📁 File: `{st.session_state.filename}`")
        st.markdown(f"📄 Pages: `{st.session_state.num_pages}`")
        st.markdown(f"🧩 Chunks: `{len(st.session_state.chunks)}`")
        st.markdown(f"✅ Status: Ready")

    st.divider()
    st.caption("Built with LangChain + HuggingFace + FAISS")


# ── MAIN AREA ─────────────────────────────────────────────────
st.title("💬 Ask Questions About Your PDF")

# Show welcome message if no PDF processed yet
if "vector_store" not in st.session_state:
    st.info("👈 Upload a PDF in the sidebar to get started!")

    # Show example questions
    st.markdown("### What can you ask?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - What is the main topic?
        - What are the key findings?
        - What is the author's name?
        """)
    with col2:
        st.markdown("""
        - What technologies are mentioned?
        - What is the CGPA / score?
        - What projects are listed?
        """)

else:
    # Initialize chat history if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show sources for bot messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View sources from PDF"):
                    for i, src in enumerate(message["sources"]):
                        st.caption(f"Source {i+1}:")
                        st.text(src[:300] + "...")

    # Chat input box at the bottom
    # st.chat_input = the text box you see at the bottom
    if question := st.chat_input("Ask a question about your PDF..."):

        # Add user message to history and display it
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        with st.chat_message("user"):
            st.markdown(question)

        # Get answer and display it
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                answer, confidence, source_docs = ask_question(
                    st.session_state.vector_store,
                    st.session_state.tokenizer,
                    st.session_state.model,
                    question,
                )

            # Show answer with confidence color coding
            # Green = high confidence, Orange = medium, Red = low
            if confidence >= 50:
                conf_color = "🟢"
            elif confidence >= 25:
                conf_color = "🟡"
            else:
                conf_color = "🔴"

            response = f"**{answer}**\n\n{conf_color} Confidence: {confidence}%"
            st.markdown(response)

            # Expandable sources section
            sources = [doc.page_content for doc in source_docs]
            with st.expander("📚 View sources from PDF"):
                for i, src in enumerate(sources):
                    st.caption(f"Source {i+1}:")
                    st.text(src[:300] + "...")

        # Save bot response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources,
        })
