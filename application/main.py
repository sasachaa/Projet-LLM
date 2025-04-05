import streamlit as st
from pdf2text2 import main as pdf2txt_main
from splitChunk import main as chunk_main
from rag import main as rag_main
import os

st.title("💬 Chatbot Médical (PDF)")

# 1) Upload du PDF par l'utilisateur
uploaded_file = st.file_uploader("Glissez un fichier PDF ici", type=["pdf"])

if uploaded_file:
    # Enregistrement temporaire du PDF
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF reçu avec succès")

    # 2) Conversion PDF -> TXT
    txt_path = pdf2txt_main("uploaded.pdf", 20)
    st.write(f"Fichier texte généré : {txt_path}")

    # 3) Découpage en chunks
    output_chunk_file = chunk_main(
        txt_input_path=txt_path,
        chunk_size=400,
        chunk_overlap=0,
        separator=".",
        output_file="chunk.txt"
    )
    st.write("Fichier chunk généré :", output_chunk_file)

    # 4) Zone de saisie utilisateur
    query = st.text_input("Posez une question sur le document :")

    # 5) Appel RAG + affichage dynamique
    if query:
        stream = rag_main(output_chunk_file, query)

        st.markdown("### 🤖 Réponse du chatbot")
        response = ""
        

        for chunk_data in stream:
            content = chunk_data['message']['content']
            response += content



