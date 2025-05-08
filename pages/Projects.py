import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Project Data Science & Analysis")

st.markdown("""
Berikut beberapa project yang pernah saya kerjakan:
- Prediksi Penjualan E-commerce
- Analisis Sentimen Media Sosial
- Segmentasi Customer
- Visualisasi Interaktif Dashboard
""")

st.subheader("Contoh Visualisasi")

df = pd.DataFrame({
    "Kategori": ["A", "B", "C", "D"],
    "Nilai": [23, 45, 12, 67]
})

fig, ax = plt.subplots()
ax.bar(df["Kategori"], df["Nilai"], color="skyblue")
st.pyplot(fig)
