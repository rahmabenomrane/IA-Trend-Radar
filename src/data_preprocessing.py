import os
import csv
import re
import nltk
from nltk.corpus import stopwords

DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "raw", "articles_raw.csv")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
CLEAN_PATH = os.path.join(CLEAN_DIR, "articles_clean.csv")

# Tech keywords qu'il NE FAUT PAS supprimer
TECH_WHITELIST = {
    "ai", "ml", "it", "vr", "ar", "iot", "devops",
    "cloud", "cybersecurity", "blockchain",
    "data", "neural", "network", "robotics", "software"
}


def clean_text(text, stop_words):
    text = text or ""

    # Remplacement des abréviations tech
    text = text.replace("A.I.", "AI").replace("I.A.", "AI")

    # Supprimer HTML
    text = re.sub(r"<.*?>", " ", text)

    # Supprimer URL
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Conserver lettres, chiffres, tirets
    text = re.sub(r"[^a-zA-Z0-9\- ]", " ", text)

    text = text.lower()
    tokens = text.split()

    clean_tokens = []
    for t in tokens:
        # Ne jamais supprimer les mots tech
        if t in TECH_WHITELIST:
            clean_tokens.append(t)
            continue

        # Supprimer stopwords
        if t in stop_words:
            continue

        # Supprimer mots trop courts
        if len(t) <= 2:
            continue

        clean_tokens.append(t)

    return " ".join(clean_tokens)


def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)

    stop_words = set(stopwords.words("english")) | set(
        stopwords.words("french"))

    with open(RAW_PATH, "r", encoding="utf-8") as fin, \
            open(CLEAN_PATH, "w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=[
            "id", "title", "url", "source", "published_at", "clean_text"
        ])
        writer.writeheader()

        for row in reader:
            cleaned = clean_text(row["content"], stop_words)

            writer.writerow({
                "id": row["id"],
                "title": row["title"],
                "url": row["url"],
                "source": row["source"],
                "published_at": row["published_at"],
                "clean_text": cleaned,
            })

    print(f"[OK] Texte nettoyé : {CLEAN_PATH}")


if __name__ == "__main__":
    main()
