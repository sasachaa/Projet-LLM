# text_chunker.py
"""
Script pour découper un fichier .txt en plusieurs chunks via CharacterTextSplitter.
Possède une fonction main(txt_input_path, chunk_size, chunk_overlap, separator)
qui crée 'chunk.txt' et renvoie le chemin du fichier créé.
"""

import os
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter

def main(
    txt_input_path: str,
    chunk_size: int = 400,
    chunk_overlap: int = 0,
    separator: str = ".",
    output_file: str = "chunk.txt"
) -> str:
    """
    Lit un fichier texte 'txt_input_path', le découpe en morceaux
    via CharacterTextSplitter, puis écrit le résultat dans 'output_file'.

    Args:
        txt_input_path (str): Chemin du fichier .txt à découper.
        chunk_size (int)    : Taille max en caractères de chaque chunk.
        chunk_overlap (int) : Overlap (nombre de caractères) entre chunks.
        separator (str)     : Séparateur préféré (ici on coupe sur un point).
        output_file (str)   : Nom du fichier de sortie (ex: 'chunk.txt').

    Returns:
        str: Chemin absolu du fichier chunks généré.
    """
    txt_input_path = "RESULT_DL-Exemple_OCR.txt"
    # Vérifie que c'est bien un fichier .txt
    input_path = Path(txt_input_path)
    if input_path.suffix.lower() != ".txt":
        raise ValueError(f"Le fichier {txt_input_path} n'est pas un .txt")

    # Lecture du contenu
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Instanciation du splitter
    text_splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Découpage
    chunks = text_splitter.split_text(content)

    # Écriture dans le fichier de sortie
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(chunks))

    print(f"Création de {output_path} terminée ({len(chunks)} chunks).")
    return str(output_path.resolve())


if __name__ == "__main__":
    # Exemple d'utilisation en direct
    # Supposons qu'on a un fichier "cat-facts.txt" dans le dossier courant
    input_file = "cat-facts.txt"
    output_file = "chunk.txt"

    # Lancement
    result_path = main(
        txt_input_path=input_file,
        chunk_size=400,
        chunk_overlap=0,
        separator=".",
        output_file=output_file
    )

    print(f"Fichier chunks généré ici: {result_path}")
