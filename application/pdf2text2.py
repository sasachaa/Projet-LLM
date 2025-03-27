# -*- coding: utf-8 -*-
"""
pdf_ocr_unified.py - Fichier unique pour convertir un PDF en texte via OCR docTR,
avec une fonction main(pdf_path) -> chemin du .txt généré.

Il combine :
- le code "pdftotext.py" (OCR, nettoyage, etc.)
- le code "pdf_ocr_cli.py" (orchestration, création fichier .txt)
"""

import logging
import os
import re
import shutil
import time
import contextlib
from datetime import date, datetime
from os.path import basename, dirname, join
from pathlib import Path

import nltk
import torch
from cleantext import clean
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from libretranslatepy import LibreTranslateAPI
from natsort import natsorted
from spellchecker import SpellChecker
from tqdm.auto import tqdm

# ------------------------------------------------------------------------------
# 1) Configuration du logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
)

# Télécharge éventuellement les stopwords NLTK si besoin
nltk.download("stopwords", quiet=True)

# ------------------------------------------------------------------------------
# 2) Fonctions pour l'OCR et le nettoyage (ex-"pdftotext.py")
# ------------------------------------------------------------------------------
def simple_rename(filepath, target_ext=".txt"):
    _fp = Path(filepath)
    basename = _fp.stem
    return f"OCR_{basename}_{target_ext}"

def rm_local_text_files(name_contains="RESULT_"):
    """
    Supprime les anciens fichiers .txt dont le nom contient `name_contains`.
    """
    files = [
        f
        for f in Path.cwd().iterdir()
        if f.is_file() and f.suffix == ".txt" and name_contains in f.name
    ]
    logging.info(f"removing {len(files)} text files")
    for f in files:
        os.remove(f)
    logging.info("done")

def corr(
    s: str,
    add_space_when_numerics=False,
    exceptions=["e.g.", "i.e.", "etc.", "cf.", "vs.", "p."],
) -> str:
    """Corrige des espaces superflus, ponctuations, etc."""
    if add_space_when_numerics:
        s = re.sub(r"(\d)\.(\d)", r"\1. \2", s)

    s = re.sub(r"\s+", " ", s)
    s = re.sub(r'\s([?.!"](?:\s|$))', r"\1", s)
    s = re.sub(r"\s\'", r"'", s)   # espace avant apostrophe
    s = re.sub(r"'\s", r"'", s)    # espace après apostrophe
    s = re.sub(r"\s,", r",", s)    # espace avant virgule

    for e in exceptions:
        expected_sub = re.sub(r"\s", "", e)
        s = s.replace(expected_sub, e)

    return s

def fix_punct_spaces(string):
    """
    Remplace certains espaces autour de ponctuation.
    Ex: "hello , there" -> "hello, there"
    """
    fix_spaces = re.compile(r"\s*([?!.,]+(?:\s+[?!.,]+)*)\s*")
    string = fix_spaces.sub(lambda x: "{} ".format(x.group(1).replace(" ", "")), string)
    string = string.replace(" ' ", "'")
    string = string.replace(' " ', '"')
    return string.strip()

def clean_OCR(ugly_text: str):
    """
    Nettoyage de base : supprime \n, \t, espaces multiples, '- ' etc.
    """
    cleaned_text = ugly_text.replace("\n", " ")
    cleaned_text = cleaned_text.replace("\t", " ")
    cleaned_text = cleaned_text.replace("  ", " ")
    cleaned_text = cleaned_text.lstrip()
    cleaned_text = cleaned_text.replace("- ", "")
    cleaned_text = cleaned_text.replace(" -", "")
    return fix_punct_spaces(cleaned_text)

def move2completed(from_dir, filename, new_folder="completed", verbose=False):
    old_filepath = join(from_dir, filename)
    new_filedirectory = join(from_dir, new_folder)
    if not os.path.isdir(new_filedirectory):
        os.mkdir(new_filedirectory)
        if verbose:
            print("created new directory:", new_filedirectory)
    new_filepath = join(new_filedirectory, filename)
    try:
        shutil.move(old_filepath, new_filepath)
        logging.info(f"successfully moved {filename} to /completed.")
    except:
        logging.info(f"ERROR! unable to move file to {new_filepath}.")

custom_replace_list = {
    "t0": "to",
    "'$": "'s",
    ",,": ", ",
    "_ ": " ",
    " '": "'",
}

replace_corr_exceptions = {
    "i. e.": "i.e.",
    "e. g.": "e.g.",
    "e. g": "e.g.",
    " ,": ",",
}

spell = SpellChecker()

def check_word_spelling(word: str) -> bool:
    misspelled = spell.unknown([word])
    return len(misspelled) == 0

def eval_and_replace(text: str, match_token: str = "- ") -> str:
    """
    Remplace '- ' quand ça forme un mot correct, sinon laisse un espace.
    """
    try:
        if match_token not in text:
            return text
        else:
            while True:
                if match_token not in text:
                    break
                split_once = text.split(match_token, maxsplit=1)
                if len(split_once) == 1:
                    break
                full_before_text, full_after_text = split_once[0], split_once[1]
                before_text = [c for c in full_before_text.split()[-1] if c.isalpha()]
                after_text = [c for c in full_after_text.split()[0] if c.isalpha()]
                joined = "".join(before_text + after_text)
                if check_word_spelling(joined):
                    text = full_before_text + full_after_text
                else:
                    text = full_before_text + " " + full_after_text
    except Exception as e:
        logging.error(f"Error spell-checking: {e}")
    return text

def cleantxt_ocr(ugly_text, lower=False, lang: str = "en") -> str:
    """
    En plus du clean_OCR, on applique clean-text 
    (avec fix_unicode, to_ascii, etc.).
    """
    from cleantext import clean
    cleaned_text = clean(
        ugly_text,
        fix_unicode=True,
        to_ascii=True,
        lower=lower,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=False,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        replace_with_punct="",
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUM>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        lang=lang,
    )
    return cleaned_text

def format_ocr_out(OCR_data):
    """
    Transforme la liste OCR en string, 
    puis applique cleantxt_ocr + corr.
    """
    if isinstance(OCR_data, list):
        text = " ".join(OCR_data)
    else:
        text = str(OCR_data)
    _clean = cleantxt_ocr(text)
    return corr(_clean)

def postprocess(text: str) -> str:
    """
    Applique des remplacements custom 
    puis retire le token '- ' si besoin.
    """
    proc = corr(cleantxt_ocr(text))
    for k, v in custom_replace_list.items():
        proc = proc.replace(str(k), str(v))
    proc = corr(proc)
    for k, v in replace_corr_exceptions.items():
        proc = proc.replace(str(k), str(v))
    proc = eval_and_replace(proc)
    return proc

def result2text(result, as_text=False) -> str or list:
    """
    Convertit le résultat docTR en un gros string (ou une liste).
    """
    full_doc = []
    for i, page in enumerate(result.pages, start=1):
        text = ""
        for block in page.blocks:
            text += "\n\t"
            for line in block.lines:
                for word in line.words:
                    text += word.value + " "
        full_doc.append(text)
    if as_text:
        return "\n".join(full_doc)
    else:
        return full_doc

def convert_PDF_to_Text(
    PDF_file,
    ocr_model=None,
    max_pages: int = 20,
):
    """
    Effectue la lecture OCR sur PDF_file (via docTR),
    assemble + nettoie le texte,
    renvoie un dict (converted_text, truncated, etc.)
    """
    start = time.perf_counter()
    pdf_path = Path(PDF_file)

    if ocr_model is None:
        ocr_model = ocr_predictor(pretrained=True)

    logging.info(f"starting OCR on {pdf_path.name}")
    doc = DocumentFile.from_pdf(pdf_path)

    truncated = False
    if len(doc) > max_pages:
        logging.warning(f"PDF has {len(doc)} pages (> {max_pages}), truncating.")
        doc = doc[:max_pages]
        truncated = True

    logging.info(f"running OCR on {len(doc)} pages")
    result = ocr_model(doc)

    # Convert docTR -> raw text
    raw_text = result2text(result)
    # Nettoyage
    proc_text = [format_ocr_out(r) for r in raw_text]
    fin_text = [postprocess(t) for t in proc_text]
    ocr_results = "\n\n".join(fin_text)

    runtime = round(time.perf_counter() - start, 2)
    logging.info("OCR complete.")

    return {
        "num_pages": len(doc),
        "runtime": runtime,
        "date": str(date.today()),
        "converted_text": ocr_results,
        "truncated": truncated,
        "length": len(ocr_results),
    }

# Autre : fonctions de traduction
lt = LibreTranslateAPI("https://translate.astian.org/")

def translate_text(text, source_l, target_l="en"):
    return str(lt.translate(text, source_l, target_l))

def translate_doc(filepath, lang_start, lang_end="en", verbose=False):
    src_folder = dirname(filepath)
    src_folder = Path(src_folder)
    trgt_folder = src_folder / f"translated_{lang_end}"
    trgt_folder.mkdir(exist_ok=True)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        foreign_t = f.readlines()
    in_name = basename(filepath)
    translated_doc = []
    for line in tqdm(foreign_t, total=len(foreign_t), desc=f"translating {in_name[:10]}..."):
        translated_line = translate_text(line, lang_start, lang_end)
        translated_doc.append(translated_line)
    t_out_name = f"[To {lang_end}]{simple_rename(in_name)}.txt"
    out_path = join(trgt_folder, t_out_name)
    with open(out_path, "w", encoding="utf-8", errors="ignore") as f_o:
        f_o.writelines(translated_doc)
    if verbose:
        print("finished translating the document! - ", datetime.now())
    return out_path

# ------------------------------------------------------------------------------
# 3) Fonctions "CLI" (ex-"pdf_ocr_cli.py"), + main()
# ------------------------------------------------------------------------------

def convert_PDF(
    pdf_path: str,
    language: str = "en",
    max_pages: int = 20,
):
    """
    convert_PDF - convertit un PDF en texte en ligne de commande (sans interface web).
    1) Supprime anciens .txt
    2) Vérifie l'extension .pdf
    3) OCR
    4) Écrit un fichier RESULT_XXX_OCR.txt
    5) Retourne la chaîne de texte
    """
    rm_local_text_files(name_contains="RESULT_")

    file_path = Path(pdf_path)
    if file_path.suffix.lower() != ".pdf":
        logging.error(f"{file_path} n'est pas un PDF.")
        return ""

    start_time = time.perf_counter()

    # Charger docTR
    with contextlib.redirect_stdout(None):
        ocr_model_local = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_mobilenet_v3_large",
            pretrained=True,
            assume_straight_pages=True,
        )

    # OCR
    conversion_stats = convert_PDF_to_Text(
        PDF_file=file_path,
        ocr_model=ocr_model_local,
        max_pages=max_pages,
    )
    converted_txt = conversion_stats["converted_text"]
    num_pages = conversion_stats["num_pages"]
    truncated = conversion_stats["truncated"]

    elapsed_minutes = round((time.perf_counter() - start_time) / 60, 2)
    logging.info(
        f"Traitement terminé en {elapsed_minutes} minutes, "
        f"pages traitées: {num_pages}, truncated={truncated}"
    )

    # Écrire le .txt
    output_name = f"RESULT_{file_path.stem}_OCR.txt"
    with open(output_name, "w", encoding="utf-8", errors="ignore") as f:
        f.write(converted_txt)

    logging.info(f"Fichier texte généré : {output_name}")
    return converted_txt

def main(pdf_path: str, max_pages: int = 20) -> str:
    """
    Fonction PRINCIPALE qui 
    1) prend un PDF en entrée 
    2) effectue l'OCR 
    3) génère RESULT_<fichier>_OCR.txt 
    4) retourne le chemin du .txt créé

    Usage exemple:
        texte_path = main("/path/to/file.pdf")
    """
    # 0) Vérif
    p = Path(pdf_path)
    if p.suffix.lower() != ".pdf":
        logging.error(f"{p} n'est pas un fichier .pdf")
        return ""

    # 1) On lance la conversion + on récupère le texte
    text_content = convert_PDF(pdf_path, max_pages=max_pages)

    # 2) On construit le nom du fichier .txt
    txt_path = Path.cwd() / f"RESULT_{p.stem}_OCR.txt"

    # 3) On renvoie le chemin
    return str(txt_path.resolve())

# ------------------------------------------------------------------------------
# 4) Test si on exécute directement
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # (Optionnel) Détection GPU
    use_gpu = torch.cuda.is_available()
    logging.info(f"GPU disponible: {use_gpu}")

    # Exemple local
    example_pdf = Path(__file__).parent / "Documents_PDF" / "DL-Exemple.pdf"
    example_pdf = str(example_pdf.resolve())

    # Appel de main()
    result_txt_path = main(example_pdf, max_pages=20)
    print("Le fichier .txt final se trouve ici:", result_txt_path)

