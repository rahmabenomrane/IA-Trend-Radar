
import os
import csv
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from newspaper import Article
from transformers import pipeline

NEWS_API_KEY = "8f22d79a156449e290120071071f6524"  
DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "raw", "articles_raw.csv")


# ---------- 1. Thèmes Tech/IT ----------

TECH_TOPICS = [
    "artificial intelligence",
    "machine learning",
    "cloud computing",
    "cybersecurity",
    "blockchain",
    "big data",
    "software engineering",
    "devops",
    "internet of things",
]


# ---------- 2. Appel API d'actualités ----------

def search_articles_for_topic(topic, page_size=5):
    """
    Utilise une API de news pour récupérer des articles sur un thème donné.
    Exemple avec un schéma type NewsAPI.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
    }
    headers = {"Authorization": f"Bearer {NEWS_API_KEY}"}

    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    articles = data.get("articles", [])
    results = []
    for art in articles:
        results.append({
            "title": art.get("title", ""),
            "url": art.get("url", ""),
            "source": art.get("source", {}).get("name", ""),
            "published_at": art.get("publishedAt", ""),
            "description": art.get("description", ""),
        })
        print(f"API returned {len(results)} articles for topic '{topic}'")

    return results


# ---------- 3. Scraping du contenu HTML ----------

def get_html(url, timeout=10):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; IA-Trend-Radar/1.0)"
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def scrape_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"[WARN] Newspaper failed {url}: {e}")
        return ""


# ---------- 4. Classificateur IA Tech/IT ----------

def build_tech_classifier():
    clf = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
    return clf


def is_tech_article(text, classifier, threshold=0.5):
    if not text or len(text.split()) < 20:
        return False

    tech_labels = [
        "Technology",
        "Artificial Intelligence",
        "Machine Learning",
        "Cybersecurity",
        "Software Engineering",
        "Blockchain"
    ]

    result = classifier(text[:1000], candidate_labels=tech_labels)
    top_label = result["labels"][0]
    top_score = result["scores"][0]

    # Debug (afficher ce que le modèle décide)
    print(f"    -> Predicted: {top_label} ({top_score:.2f})")

    return top_label in tech_labels and top_score >= threshold


# ---------- 5. Pipeline principal ----------

def main():
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)

    classifier = build_tech_classifier()

    all_rows = []
    article_id = 1

    for topic in TECH_TOPICS:
        print(f"\n[INFO] Recherche d'articles pour le thème : {topic}")
        api_articles = search_articles_for_topic(topic, page_size=5)

        for meta in api_articles:
            url = meta["url"]
            print(f"[INFO] Scraping : {url}")

            raw_text = scrape_content(url)
            time.sleep(1)  # pour ne pas spammer

            if not raw_text:
                continue

            # IA : filtrage Tech/IT
            if not is_tech_article(raw_text, classifier):
                print("    -> Rejeté (non Tech/IT)")
                continue

            print("    -> Accepté (article Tech/IT)")

            row = {
                "id": article_id,
                "title": meta["title"],
                "url": url,
                "source": meta["source"],
                "published_at": meta["published_at"],
                "raw_text": raw_text
            }
            all_rows.append(row)
            article_id += 1

    # Sauvegarde
    with open(RAW_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "title", "url",
                        "source", "published_at", "raw_text"]
        )
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(
        f"\n[OK] {len(all_rows)} articles Tech/IT sauvegardés dans {RAW_PATH}")


if __name__ == "__main__":
    main()
