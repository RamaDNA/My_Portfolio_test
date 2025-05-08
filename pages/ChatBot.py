import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

# 🔑 Ambil token dari secrets & set ke environment variable
hf_token = st.secrets["HUGGINGFACE_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

st.title("🤖 Chatbot Profile (Langsung Tanya, Gratis)")

# --- Load PDF sekali di awal
@st.cache_resource
def load_knowledge_base():
    loader = PyPDFLoader("data/profile.pdf")  # Pastikan file profile.pdf ada di folder 'data'
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # 🔥 load model embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    return retriever

retriever = load_knowledge_base()

query = st.text_input("Tanyakan apapun tentang saya:")
if query:
    related_docs = retriever.get_relevant_documents(query)
    
    if related_docs:
        # Ambil dokumen paling relevan
        answer = related_docs[0].page_content
        st.write("💬", answer)
        
    else:
        st.warning("Tidak ditemukan jawaban yang relevan.")
else:
    st.info("Masukkan pertanyaan di atas.")
