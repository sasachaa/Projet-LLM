import streamlit as st
from pathlib import Path

# Importer les fonctions principales de vos modules
from pdf2text2 import main as pdf2txt_main
from splitChunk import main as chunk_main
from rag import main as rag_main

def run_pipeline(pdf_file_path, max_pages=20, chunk_size=400, chunk_overlap=0, separator=".", output_file="chunk.txt"):
    """
    Exécute la chaîne complète :
      1. Conversion PDF -> .txt
      2. Découpage du texte en chunks
      3. Lancement du RAG
    Retourne le chemin du fichier texte et le fichier de chunks.
    """
    # 1) Conversion PDF -> .txt
    txt_path = pdf2txt_main(pdf_file_path, max_pages)
    st.write(f"Fichier texte généré: {txt_path}")
    
    # 2) Découpe en chunks
    output_chunk_file = chunk_main(
        txt_input_path=txt_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=separator,
        output_file=output_file
    )
    st.write(f"Fichier de chunks généré: {output_chunk_file}")
    
    # 3) Lancement du RAG
    # Ici, rag_main prend comme paramètre 'dataset' le fichier de chunks généré.
    rag_main(dataset=output_chunk_file)
    
    return txt_path, output_chunk_file

def main():
    st.title("Pipeline de Conversion PDF -> RAG")
    st.write("Uploader un fichier PDF pour lancer le pipeline complet.")
    
    # Uploader le PDF
    uploaded_file = st.file_uploader("Choisissez un fichier PDF", type=["pdf"])
    
    # Paramètres
    max_pages = st.number_input("Nombre maximum de pages à traiter", min_value=1, value=20)
    chunk_size = st.number_input("Taille max des chunks (en caractères)", min_value=100, value=400)
    chunk_overlap = st.number_input("Overlap entre chunks (en caractères)", min_value=0, value=0)
    
    if uploaded_file is not None:
        # Sauvegarder le PDF uploadé sur le serveur
        pdf_file_path = f"temp_{uploaded_file.name}"
        with open(pdf_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success(f"Fichier PDF sauvegardé: {pdf_file_path}")
        
        if st.button("Lancer le pipeline"):
            txt_path, output_chunk_file = run_pipeline(pdf_file_path, max_pages, chunk_size, chunk_overlap)
            st.success("Pipeline terminé !")
            st.write("Vous pouvez consulter les fichiers générés dans le répertoire de travail.")
            
if __name__ == "__main__":
    main()
