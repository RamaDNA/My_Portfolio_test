import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import os
from huggingface_hub import InferenceClient

# 🔑 Ambil token dari secrets dan set ke environment variable
hf_token = st.secrets["HUGGINGFACE_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

st.title("🤖 Chatbot Profile (Langsung Tanya, Gratis)")

# --- Load PDF sekali di awal
@st.cache_resource
def load_knowledge_base():
    loader = PyPDFLoader("data/profile.pdf")  # Pastikan file profile.pdf ada di folder 'data'
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # 🔥 load model embeddings dari huggingface
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    return retriever

retriever = load_knowledge_base()

# --- Define LLM menggunakan HuggingFace API
client = InferenceClient(token=hf_token)

# Load model untuk chat response
def chat_response(query):
    response = client.text_generation(query, model="google/flan-t5-small")
    return response[0]['generated_text']

# --- Input pertanyaan user
query = st.text_input("Tanyakan apapun tentang saya:")
if query:
    # Mengambil dokumen terkait menggunakan retriever dari Langchain
    related_docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in related_docs])

    # Jawaban berdasarkan context
    chatbot_answer = chat_response(query + " " + context)
    st.write("💬", chatbot_answer)

    # Menampilkan dokumen yang relevan
    with st.expander("👀 Lihat dokumen terkait"):
        for i, doc in enumerate(related_docs):
            st.write(f"Doc {i+1}:")
            st.write(doc.page_content)
else:
    st.info("Masukkan pertanyaan di atas.")
