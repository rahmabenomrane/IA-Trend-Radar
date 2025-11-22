import os
import csv
import re

import nltk
from nltk.corpus import stopwords

# À lancer une seule fois dans un terminal Python :
# import nltk
# nltk.download("stopwords")

DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "raw", "articles_raw.csv")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
CLEAN_PATH = os.path.join(CLEAN_DIR, "articles_clean.csv")


def clean_text(text, stop_words):
    """
    Nettoie un texte :
    - supprime HTML, URLs, ponctuation
    - met en minuscules
    - enlève les stopwords
    """
    if text is None:
        return ""

    # Supprimer balises HTML
    text = re.sub(r"<.*?>", " ", text)

    # Supprimer URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Garder lettres / chiffres / espaces
    text = re.sub(r"[^a-zA-ZÀ-ÖØ-öø-ÿ0-9\s]", " ", text)

    # Minuscules
    text = text.lower()

    # Tokenisation simple
    tokens = text.split()

    # Supprimer stopwords et les mots trop courts
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    return " ".join(tokens)


def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)

    # Stopwords anglais + français (pour tech / IT FR & EN)
    stop_words = set(stopwords.words("english")) | set(
        stopwords.words("french"))

    with open(RAW_PATH, "r", encoding="utf-8") as fin, \
            open(CLEAN_PATH, "w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin)
        fieldnames = [
            "id",
            "title",
            "url",
            "source",
            "published_at",
            "clean_text"
        ]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            raw = row.get("raw_text", "")
            cleaned = clean_text(raw, stop_words)

            writer.writerow({
                "id": row.get("id", ""),
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "source": row.get("source", ""),
                "published_at": row.get("published_at", ""),
                "clean_text": cleaned
            })

    print(f"[OK] Articles nettoyés sauvegardés dans {CLEAN_PATH}")


if __name__ == "__main__":
    main()
