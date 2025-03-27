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
rag_main(chunk_file=output_chunk_file)

