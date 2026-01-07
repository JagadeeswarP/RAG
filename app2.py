import streamlit as st
import faiss
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader

from fastembed import TextEmbedding

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_groq import ChatGroq

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "input_type" not in st.session_state:
    st.session_state.input_type = None

if "query" not in st.session_state:
    st.session_state.query = ""

if "answer" not in st.session_state:
    st.session_state.answer = ""

-
def clear_all():
    st.session_state.vectorstore = None
    st.session_state.query = ""
    st.session_state.answer = ""
    st.rerun()

st.sidebar.title(" Groq Configuration")

GROQ_API_KEY = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password",
    help="Required for LLM inference"
)

if not GROQ_API_KEY:
    st.sidebar.warning("Groq API key is required")
    st.stop()

CHAT_MODEL = st.sidebar.selectbox(
    "Groq Model",
    [
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768"
    ],
    index=0
)

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=CHAT_MODEL,
    temperature=0.2
)

@st.cache_resource
def load_embedding_model():
    return TextEmbedding(
        model_name="nomic-ai/nomic-embed-text-v1.5"
    )

embedding_model = load_embedding_model()

def generate_embedding(text: str):

    return list(embedding_model.embed([text]))[0]

def process_input(input_type, input_data):

    if input_type == "Link":
        loader = WebBaseLoader(input_data)
        docs = loader.load()
        text = "\n".join(d.page_content for d in docs)

    elif input_type == "PDF":
        pdf = PdfReader(BytesIO(input_data.read()))
        text = "\n".join(p.extract_text() or "" for p in pdf.pages)

    elif input_type == "DOCX":
        doc = Document(BytesIO(input_data.read()))
        text = "\n".join(p.text for p in doc.paragraphs)

    elif input_type == "TXT":
        text = input_data.read().decode("utf-8")

    elif input_type == "Text":
        text = input_data

    else:
        raise ValueError("Unsupported input type")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    # Create FAISS index
    dim = len(generate_embedding("test"))
    index = faiss.IndexFlatL2(dim)

    vectorstore = FAISS(
        embedding_function=generate_embedding,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vectorstore.add_texts(chunks)
    return vectorstore

def answer_question(vectorstore, query):

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are a RAG assistant.
Answer ONLY using the context below.
Be concise and factual.
If the answer is not present, say so clearly.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content.strip()

def main():
    st.set_page_config(page_title="RAG Assistant (FastEmbed)", layout="wide")
    st.title("Jagadeeswar's RAG Assistant")
    st.caption("FAISS • FastEmbed (Local) • Groq Inference")

    col1, _ = st.columns([1, 6])
    with col1:
        if st.button("🧹 Clear All"):
            clear_all()

    input_type = st.selectbox(
        "Input Type",
        ["Link", "PDF", "DOCX", "TXT", "Text"],
        key="input_type_select"
    )

    if st.session_state.input_type != input_type:
        st.session_state.input_type = input_type
        st.session_state.vectorstore = None
        st.session_state.query = ""
        st.session_state.answer = ""

    if input_type == "Link":
        input_data = st.text_input("Enter URL")
    elif input_type == "Text":
        input_data = st.text_area("Paste text")
    else:
        input_data = st.file_uploader("Upload file")

    if st.button("Process Input"):
        if not input_data:
            st.error("Input required")
        else:
            with st.spinner("Building vector store..."):
                st.session_state.vectorstore = process_input(input_type, input_data)
                st.success("Ready for questions!")

    if st.session_state.vectorstore:
        st.session_state.query = st.text_input(
            "Ask a question",
            value=st.session_state.query
        )

        if st.button("Get Answer") and st.session_state.query:
            with st.spinner("Groq is thinking..."):
                st.session_state.answer = answer_question(
                    st.session_state.vectorstore,
                    st.session_state.query
                )

        if st.session_state.answer:
            st.markdown("### Answer")
            st.write(st.session_state.answer)

if __name__ == "__main__":
    main()
