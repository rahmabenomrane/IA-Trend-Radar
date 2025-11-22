import os
import csv
from transformers import pipeline

DATA_DIR = "data"
CLEAN_PATH = os.path.join(DATA_DIR, "clean", "articles_clean.csv")
SUM_DIR = os.path.join(DATA_DIR, "summaries")
SUM_PATH = os.path.join(SUM_DIR, "articles_with_summaries.csv")


def load_clean_articles(path):
    articles = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            articles.append(row)
    return articles


def chunk_text(text, max_words=300):
    """
    Découpe un texte long en segments de max_words mots.
    On travaille en mots (pas tokens), donc on prend une marge de sécurité < 1024 tokens.
    """
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])


def summarize_long_text(text, summarizer, max_summary_tokens=120):
    """
    Résume un texte long :
    - on le découpe en petits morceaux
    - on résume chaque morceau
    - on concatène les résumés partiels
    """
    text = text.strip()
    if not text:
        return ""

    chunks = list(chunk_text(text, max_words=300))

    partial_summaries = []
    for idx, chunk in enumerate(chunks):
        try:
            result = summarizer(
                chunk,
                max_length=max_summary_tokens,
                min_length=40,
                do_sample=False,
                truncation=True,   # 🔹 Très important pour éviter l'erreur de longueur
            )[0]["summary_text"]
            partial_summaries.append(result)
        except Exception as e:
            print(f"[WARN] Erreur lors du résumé d'un chunk ({idx}) : {e}")
            continue

    if not partial_summaries:
        return ""

    final_summary = " ".join(partial_summaries)
    return final_summary


def main():
    os.makedirs(SUM_DIR, exist_ok=True)

    print("[INFO] Chargement des articles nettoyés...")
    articles = load_clean_articles(CLEAN_PATH)

    if not articles:
        print("[WARN] Aucun article trouvé dans articles_clean.csv")
        return

    print("[INFO] Chargement du modèle de résumé (ça peut prendre un peu de temps)...")
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

    with open(SUM_PATH, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id",
            "title",
            "url",
            "source",
            "published_at",
            "clean_text",
            "summary"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for art in articles:
            clean_text = art.get("clean_text", "")
            art_id = art.get("id", "")
            art_title = (art.get("title") or "")[:60]

            print(f"[INFO] Résumé de l'article id={art_id} : {art_title}...")

            try:
                summary = summarize_long_text(clean_text, summarizer)
            except Exception as e:
                print(f"[WARN] Erreur sur l'article id={art_id} : {e}")
                summary = ""

            writer.writerow({
                "id": art.get("id", ""),
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "source": art.get("source", ""),
                "published_at": art.get("published_at", ""),
                "clean_text": clean_text,
                "summary": summary
            })

    print(f"[OK] Fichier avec résumés générés : {SUM_PATH}")


if __name__ == "__main__":
    main()
