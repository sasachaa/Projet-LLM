from pdf2text2 import main
from splitChunk import main
from rag import main

pdf_file = "DL-Exemple.pdf"
txt_path = main(pdf_file, max_pages=20)
print(f"Fichier texte généré: {txt_path}")


output_chunk_file = main(
    txt_input_path=txt_path,
    chunk_size=400,
    chunk_overlap=0,
    separator=".",
    output_file="chunk.txt"
)
print("Fichier généré:", output_chunk_file)

main(chunk_file=output_chunk_file)
