# -*- coding: utf-8 -*-
"""
pdf_ocr_cli.py - Script minimal pour convertir un PDF en texte via OCR
sans interface Gradio. Tout se fait depuis le terminal.
"""

import logging
import time
import contextlib
from pathlib import Path
import nltk
import torch

# 1) Importer les fonctions d'un autre fichier (pdf2text.py)
#    C'est ici qu'on appelle convert_PDF_to_Text, rm_local_text_files
from PdfToText import (
    convert_PDF_to_Text,
    rm_local_text_files,
    ocr_predictor
)



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Optionnel : si nécessaire pour certaines fonctions nltk
nltk.download("stopwords")  # TODO : ajustez si besoin

def convert_PDF(
    pdf_path: str,
    language: str = "en",
    max_pages: int = 20,
):
    """
    convert_PDF - convertit un PDF en texte depuis le terminal (pas de Gradio).
    Args:
        pdf_path (str): chemin vers le PDF
        language (str, optional): Langue à utiliser pour l'OCR. Par défaut "en".
        max_pages (int, optional): Nombre max de pages à traiter. Par défaut 20.

    Returns:
        str: le contenu textuel du PDF après OCR
    """
    # Moment où on nettoie les éventuels fichiers .txt déjà présents (optionnel)
    rm_local_text_files()

    # Contrôle du suffixe pour s'assurer qu'on a bien un PDF
    file_path = Path(pdf_path)
    if not file_path.suffix.lower() == ".pdf":
        logging.error(f"Le fichier {file_path} n'est pas un PDF.")
        return None

    # Démarrage du chrono
    start_time = time.perf_counter()

    # 2) Appel de la fonction d’un autre fichier (pdf2text.py)
    #    pour faire l'OCR et récupérer le texte
    conversion_stats = convert_PDF_to_Text(
        PDF_file=file_path,
        ocr_model=ocr_model,
        max_pages=max_pages,
    )

    converted_txt = conversion_stats["converted_text"]
    num_pages = conversion_stats["num_pages"]
    was_truncated = conversion_stats["truncated"]

    # Fin du chrono
    elapsed_minutes = round((time.perf_counter() - start_time) / 60, 2)
    logging.info(
        f"Traitement terminé en {elapsed_minutes} minutes, pages traitées: {num_pages}, "
        f"troncation: {was_truncated}"
    )

    # 3) Moment où l’on crée le fichier .txt de sortie
    output_name = f"RESULT_{file_path.stem}_OCR.txt"
    with open(output_name, "w", encoding="utf-8", errors="ignore") as f:
        f.write(converted_txt)

    logging.info(f"Fichier texte généré : {output_name}")
    return converted_txt


if __name__ == "__main__":
    # Détection du GPU
    use_GPU = torch.cuda.is_available()
    logging.info(f"GPU disponible : {use_GPU}")

    # Chargement du modèle OCR (docTR)
    logging.info("Chargement du modèle OCR...")
    with contextlib.redirect_stdout(None):
        ocr_model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_mobilenet_v3_large",
            pretrained=True,
            assume_straight_pages=True,
        )

    # Path to pdf
    example_pdf = Path(__file__).parent / "Documents_PDF/Deep Learning with Python, François Chollet-min.pdf"
    example_pdf = str(example_pdf.resolve())

    # Exécution du script sur ce PDF en le limitant à 20 pages
    convert_PDF(
        pdf_path=example_pdf,
        language="en",
        max_pages=20
    )

    # Astuce : dans une vraie utilisation en ligne de commande,
    # vous pourriez utiliser argparse pour passer --pdf <fichier.pdf>
    # et --max_pages <int> directement depuis le terminal.
