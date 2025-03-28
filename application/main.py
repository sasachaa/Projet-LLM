# main.py

from pdf2text2 import main as pdf2txt_main
from splitChunk import main as chunk_main
from rag import main as rag_main

pdf_file = "DL-Exemple.pdf"

# 1) Conversion PDF -> .txt
txt_path = pdf2txt_main(pdf_file, 20)
print(f"Fichier texte généré: {txt_path}")

# 2) Découpe en chunks
output_chunk_file = chunk_main(
    txt_input_path=txt_path,
    chunk_size=400,
    chunk_overlap=0,
    separator=".",
    output_file="chunk.txt"
)
print("Fichier généré:", output_chunk_file)

# 3) Lancement du RAG
rag_main(dataset=output_chunk_file)

# import streamlit as st
# import os
# from pathlib import Path

# # On alias les fonctions 'main' des trois scripts
# from pdf2text2 import main as pdf2txt_main
# from splitChunk import main as chunk_main
# from rag import main as rag_main

# def run_rag_pipeline():
#     """
#     Exécute la pipeline :
#     1) Uploader un PDF
#     2) PDF->TXT
#     3) TXT->Chunks
#     4) Lancement RAG
#     """
#     st.title("Pipeline RAG : PDF -> TXT -> Chunks -> Chatbot")

#     # 1) Upload PDF
#     uploaded_file = st.file_uploader("Choisissez un PDF", type=["pdf"])
#     if uploaded_file is not None:
#         # On va sauvegarder ce PDF en local
#         pdf_file_path = "temp_uploaded.pdf"
#         with open(pdf_file_path, "wb") as f:
#             f.write(uploaded_file.getvalue())
        
#         st.success(f"Fichier PDF reçu : {uploaded_file.name}")

#         # 2) PDF -> TXT
#         if st.button("Convertir PDF en .txt"):
#             txt_path = pdf2txt_main(pdf_file_path, 20)  # 20 pages max
#             st.write(f"Fichier texte généré: {txt_path}")

#             # 3) Découpage
#             if st.button("Découper le .txt en chunks"):
#                 output_chunk_file = chunk_main(
#                     txt_input_path=txt_path,
#                     chunk_size=400,
#                     chunk_overlap=0,
#                     separator=".",
#                     output_file="chunk.txt"
#                 )
#                 st.write(f"Fichier chunk généré : {output_chunk_file}")

#                 # 4) Lancement RAG
#                 st.write("#### Lancement du RAG (chatbot)")

#                 # On demande la question à l'utilisateur
#                 user_question = st.text_input("Posez votre question : ")
#                 if st.button("Poser la question au chatbot"):
#                     # rag_main habituellement lit un input() dans la console.
#                     # On va modifier rag_main pour qu'il reçoive la question en paramètre.
#                     #
#                     # Disons que dans rag_main, on ajoute un param "question=None".
#                     # Si question est fourni, on l'utilise, sinon on fait input().
#                     # => On suppose qu'on a déjà fait cette modif. 
                    
#                     # rag_main(chunk_file=output_chunk_file, question=user_question)
#                     # 
#                     # Pour voir la réponse, on pourrait imprimer en console 
#                     # ou renvoyer quelque chose depuis rag_main. 
#                     # 
#                     # Comme le code rag_main actuel l'affiche en direct, on ne peut pas 
#                     # facilement capter la réponse. On se contentera d'indiquer 
#                     # "Regardez la console..." OU on modifie rag_main pour renvoyer la 
#                     # réponse en plus.
                    
#                     rag_main(chunk_file=output_chunk_file, question=user_question)
#                     st.write("Réponse du chatbot : (voir console si stream)")

#     else:
#         st.info("Veuillez uploader un PDF.")

# def main():
#     run_rag_pipeline()

# if __name__ == "__main__":
#     main()
