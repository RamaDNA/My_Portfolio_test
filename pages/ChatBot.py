import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import os

st.title("🤖 Chatbot Profile (Langsung Tanya, Gratis)")

# --- Load PDF sekali di awal
@st.cache_resource
def load_knowledge_base():
    loader = PyPDFLoader("data/profile.pdf")  # PDF kamu letakkan di folder 'data'
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    return retriever

retriever = load_knowledge_base()

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

query = st.text_input("Tanyakan apapun tentang saya:")
if query:
    related_docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in related_docs])

    result = qa_pipeline(question=query, context=context)
    st.write("💬", result['answer'])

    with st.expander("👀 Lihat dokumen terkait"):
        for i, doc in enumerate(related_docs):
            st.write(f"Doc {i+1}:")
            st.write(doc.page_content)
else:
    st.info("Masukkan pertanyaan di atas.")
