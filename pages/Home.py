import streamlit as st
from streamlit_lottie import st_lottie
import requests

st.title("🏠 Home")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_3rwasyjy.json")

col1, col2 = st.columns(2)

with col1:
    st_lottie(lottie_coding, height=300)

with col2:
    st.image("https://i.imgur.com/9bqL5zC.png", width=200, caption="Foto Profil Kamu")
    st.markdown("""
    ### Hai, Saya [Nama Kamu] 👋
    - 🎓 Data Scientist & Data Analyst
    - 💻 Berpengalaman dengan Python, ML, Data Viz
    - 📊 Portfolio lengkap di halaman project
    """)
