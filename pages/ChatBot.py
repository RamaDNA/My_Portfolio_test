import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import os

# 🔑 Token dari secrets
hf_token = st.secrets["HUGGINGFACE_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

st.title("🤖 Chatbot Profile (Chat Interaktif)")

# --- Load PDF dan build retriever
@st.cache_resource
def load_retriever():
    loader = PyPDFLoader("data/profile.pdf")
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    
    return db.as_retriever()

retriever = load_retriever()

# --- Load LLM dari huggingface hub
llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0, "max_length":512})

# --- ConversationalRetrievalChain
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

# --- Simpan chat history di session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- UI Input
query = st.text_input("💬 Kamu:", "")

if query:
    # Jalankan chain
    result = qa_chain({"question": query, "chat_history": st.session_state.chat_history})
    
    # Tambah ke history
    st.session_state.chat_history.append((query, result['answer']))

# --- Tampilkan semua history
for i, (user_q, bot_a) in enumerate(st.session_state.chat_history):
    st.markdown(f"**Kamu:** {user_q}")
    st.markdown(f"**Bot:** {bot_a}")
