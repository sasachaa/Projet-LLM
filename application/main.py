# main.py

from pdf2text2 import main as pdf2txt_main
from splitChunk import main as chunk_main
from rag import main as rag_main
import streamlit as st

pdf_file = "DL-Exemple.pdf"

# 1) Conversion PDF -> .txt
txt_path = pdf2txt_main(pdf_file, 20)
st.write(f"Fichier texte généré: {txt_path}")

# 2) Découpe en chunks
output_chunk_file = chunk_main(
    txt_input_path=txt_path,
    chunk_size=400,
    chunk_overlap=0,
    separator=".",
    output_file="chunk.txt"
)
st.write("Fichier généré:", output_chunk_file)

# 3) Lancement du RAG
st.write("Input")
input_query = st.text_input()
stream = rag_main(dataset=output_chunk_file, input_query)
st.title("Chatbot response:")
for chunk_data in stream:
    st.write(chunk_data['message']['content'], end='', flush=True)
