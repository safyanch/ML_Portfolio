import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
# Page configuration
st.set_page_config(
    page_title="📚 ML Document Q&A",
    page_icon="🤖",
    layout="wide"
)

# Load API key from Streamlit secrets or .env
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Process uploaded PDF ─────────────────────────────
@st.cache_resource
def process_pdf(file_bytes, filename):
    """Load PDF → chunk → embed → store in ChromaDB"""

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    pages  = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)

    # Create embeddings — runs locally, free
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Store in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=f"chroma_{filename}"
    )

    os.unlink(tmp_path)  # clean up temp file
    return vectorstore, len(pages), len(chunks)


# ── Build RAG chain ──────────────────────────────────
def build_rag_chain(vectorstore):
    """Build retriever + prompt + LLM chain"""

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI tutor.
Answer questions using ONLY the context below.
If the answer is not in the context say:
'I cannot find this in the uploaded document.'
Be concise and clear.

Context:
{context}"""),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain, retriever
# ── App title ────────────────────────────────────────
st.title("📚 ML Document Q&A Assistant")
st.markdown("""
*Built by Dr. Muhammad Safyan — PhD ML Engineer & University Lecturer*
Upload any PDF and ask questions. Powered by RAG + LangChain + Llama3.
""")

# ── Session state initialisation ─────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ── Sidebar — PDF upload ──────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Upload any text-based PDF — lecture notes, papers, books"
    )

    if uploaded_file is not None:
        with st.spinner("Processing PDF... please wait"):
            vectorstore, n_pages, n_chunks = process_pdf(
                uploaded_file.read(),
                uploaded_file.name
            )
            st.session_state.vectorstore = vectorstore
            chain, retriever = build_rag_chain(vectorstore)
            st.session_state.rag_chain  = chain
            st.session_state.retriever  = retriever

        st.success(f"✅ Ready!")
        st.info(f"📄 Pages: {n_pages}")
        st.info(f"✂️ Chunks: {n_chunks}")

        # Clear chat when new PDF uploaded
        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.markdown("""
**How it works:**
1. Upload your PDF
2. Ask any question
3. AI answers from document only
4. No hallucination — grounded answers

**Built with:**
LangChain · ChromaDB · HuggingFace · Groq
    """)

# ── Main area — chat interface ────────────────────────
if st.session_state.vectorstore is None:
    st.info("👈 Upload a PDF from the sidebar to get started")
    st.markdown("""
### Example questions you can ask:
- What are the main topics covered in this document?
- Explain the concept of gradient descent
- What is the difference between overfitting and underfitting?
- Summarize chapter 3
    """)
else:
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)

    # Chat input
    question = st.chat_input("Ask a question about your document...")

    if question:
        # Show user message
        with st.chat_message("user"):
            st.write(question)

        # Get retrieved chunks for transparency
        retrieved_chunks = st.session_state.retriever.invoke(question)

        # Get RAG answer
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                answer = st.session_state.rag_chain.invoke(question)
            st.write(answer)

            # Show retrieved chunks in expander
            with st.expander("📄 View source chunks"):
                for i, chunk in enumerate(retrieved_chunks, 1):
                    st.markdown(f"**Chunk {i}** "
                                f"(page {chunk.metadata.get('page','?')+1})")
                    st.text(chunk.page_content[:300])
                    st.divider()

        # Save to session history
        st.session_state.chat_history.append(
            HumanMessage(content=question))
        st.session_state.chat_history.append(
            AIMessage(content=answer))