# -*- coding: utf-8 -*-
"""
pdf_ocr_cli.py - Script minimal pour convertir un PDF en texte via OCR
dans une unique fonction main(), sans interface Gradio. Tout se fait
en un seul appel de fonction.
"""

import logging
import time
import contextlib
from pathlib import Path
import nltk
import torch

# Depuis PdfToText.py, on importe :
# - convert_PDF_to_Text : la fonction qui fait l'OCR et renvoie un dict
# - rm_local_text_files : pour nettoyer d'anciens .txt (optionnel)
# - ocr_predictor       : pour charger le modèle docTR
from PdfToText import (
    convert_PDF_to_Text,
    rm_local_text_files,
    ocr_predictor
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Télécharge les stopwords NLTK (si besoin de ponctuels nettoyages)
nltk.download("stopwords", quiet=True)

def main(
    pdf_path: str,
    language: str = "en",
    max_pages: int = 20
) -> str:
    """
    Fonction principale : on lui donne en paramètre le chemin
    d'un PDF, et elle réalise :
      1) Détection GPU
      2) Chargement du modèle OCR
      3) OCR du PDF pour récupérer le texte
      4) Création d'un fichier .txt contenant le texte
      5) Renvoie le chemin du fichier .txt créé

    Args:
        pdf_path (str): Chemin vers le fichier PDF à traiter
        language (str): Langue du document (pour l'OCR, s'il y a un param. multi-lang)
        max_pages (int): Nombre maximum de pages à traiter

    Returns:
        str: Chemin vers le fichier .txt généré
    """
    # 0) Vérifier que c'est bien un .pdf
    file_path = Path(pdf_path)
    if file_path.suffix.lower() != ".pdf":
        logging.error(f"Le fichier {file_path} n'est pas un PDF.")
        return ""

    # 1) Détection du GPU
    use_GPU = torch.cuda.is_available()
    logging.info(f"GPU disponible : {use_GPU}")

    # 2) Chargement du modèle OCR (docTR)
    logging.info("Chargement du modèle OCR...")
    with contextlib.redirect_stdout(None):
        ocr_model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_mobilenet_v3_large",
            pretrained=True,
            assume_straight_pages=True,
        )

    # (Optionnel) Supprimer d'anciens .txt dont le nom contient "RESULT_"
    rm_local_text_files(name_contains="RESULT_")

    # 3) OCR : convertit le PDF en texte
    start_time = time.perf_counter()
    logging.info(f"Début OCR sur {pdf_path}")
    conversion_stats = convert_PDF_to_Text(
        PDF_file=file_path,
        ocr_model=ocr_model,
        max_pages=max_pages,
    )

    # Récupérer le texte
    converted_txt = conversion_stats["converted_text"]
    num_pages = conversion_stats["num_pages"]
    was_truncated = conversion_stats["truncated"]

    # Chrono OCR terminé
    elapsed_minutes = round((time.perf_counter() - start_time) / 60, 2)
    logging.info(
        f"Traitement terminé en {elapsed_minutes} minutes, pages traitées: {num_pages}, "
        f"troncation: {was_truncated}"
    )

    # 4) Création du fichier texte
    output_name = f"RESULT_{file_path.stem}_OCR.txt"
    with open(output_name, "w", encoding="utf-8", errors="ignore") as f:
        f.write(converted_txt)

    logging.info(f"Fichier texte généré : {output_name}")

    # 5) Retourne le chemin du fichier .txt
    return str(Path(output_name).resolve())


# ------------------
# Exemple d'utilisation : si tu l'exécutes directement (python pdf_ocr_cli.py)
# ------------------
if __name__ == "__main__":
    # Exemple local : PDF dans un sous-dossier
    example_pdf = Path(__file__).parent / "Documents_PDF/Deep Learning with Python, François Chollet-min.pdf"
    example_pdf = str(example_pdf.resolve())

    output_txt_path = main(
        pdf_path=example_pdf,
        language="en",
        max_pages=20
    )
    print(f"Fichier texte final : {output_txt_path}")
