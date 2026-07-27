"""
pdf_loader.py — Extraction et nettoyage de contenu PDF
Auteure : TinhinaneBA
"""

import re
import pdfplumber
from pathlib import Path


def extract_pdf_content(pdf_path: Path) -> dict:
    """
    Extrait le contenu textuel d'un PDF page par page.

    Args:
        pdf_path : chemin vers le fichier PDF

    Returns:
        dict avec filename, n_pages, pages, total_words, total_chars
    """
    pages_content = []

    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)

        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages_content.append({
                    'page'   : i + 1,
                    'text'   : text.strip(),
                    'n_words': len(text.split()),
                    'n_chars': len(text)
                })

    return {
        'filename'     : pdf_path.name,
        'n_pages'      : n_pages,
        'pages'        : pages_content,
        'total_words'  : sum(p['n_words'] for p in pages_content),
        'total_chars'  : sum(p['n_chars'] for p in pages_content),
        'pages_parsed' : len(pages_content)
    }


def clean_text(text: str) -> str:
    """
    Nettoie le texte extrait d'un PDF.
    Supprime les artefacts courants des rapports financiers.

    Args:
        text : texte brut extrait

    Returns:
        texte nettoye
    """
    # Supprimer les sauts de ligne multiples
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Supprimer les espaces multiples
    text = re.sub(r' {2,}', ' ', text)

    # Supprimer les numeros de page isoles
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Supprimer les caracteres speciaux parasites
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\%\€\$\/\n]', ' ', text)

    return text.strip()


def load_reports_from_dir(reports_dir: Path) -> dict:
    """
    Charge et extrait tous les PDFs d'un dossier.

    Args:
        reports_dir : dossier contenant les PDFs

    Returns:
        dict {nom_fichier: contenu_extrait}
    """
    all_docs = {}
    pdf_files = list(reports_dir.glob('*.pdf'))

    if not pdf_files:
        raise FileNotFoundError(f"Aucun PDF trouve dans {reports_dir}")

    for pdf_path in pdf_files:
        doc = extract_pdf_content(pdf_path)

        # Appliquer le nettoyage sur chaque page
        for page in doc['pages']:
            page['text_clean'] = clean_text(page['text'])

        all_docs[pdf_path.stem] = doc
        print(f" {pdf_path.name} — {doc['pages_parsed']} pages "
              f"/ {doc['total_words']:,} mots")

    return all_docs


if __name__ == "__main__":
    import sys
    reports_dir = Path(sys.argv[1]) if len(sys.argv) > 1 \
                  else Path('data/reports')
    docs = load_reports_from_dir(reports_dir)
    print(f"\n{len(docs)} rapport(s) charge(s)")