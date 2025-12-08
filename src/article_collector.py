import os
import csv
import requests
from transformers import pipeline

DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "raw", "articles_raw.csv")

# 🔑 API KEY GNews 
GNEWS_API_KEY = "6709734e89c5f27677bffc620134cf7f"


TECH_TOPICS = [
    "artificial intelligence",
    "machine learning",
    "cloud computing",
    "cybersecurity",
    "blockchain",
    "devops",
    "software engineering",
    "robotics",
    "internet of things",
    "data science",
]


def fetch_articles(topic):
    url = f"https://gnews.io/api/v4/search"
    params = {
        "q": topic,
        "lang": "en",
        "max": 5,
        "apikey": GNEWS_API_KEY
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json().get("articles", [])


def build_tech_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def main():
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    classifier = build_tech_classifier()

    with open(RAW_PATH, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "title", "url",
                      "source", "published_at", "content"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        article_id = 1

        for topic in TECH_TOPICS:
            print(f"[INFO] Recherche : {topic}")
            articles = fetch_articles(topic)

            for art in articles:
                content = art.get("content", "")

                if not content:
                    continue

                # IA: vérification tech
                labels = [
                    "Artificial Intelligence",
                    "Machine Learning",
                    "Cybersecurity",
                    "Cloud Computing",
                    "Software Engineering",
                    "Technology",
                    "Robotics",
                    "IoT",
                    "DevOps",
                ]
                result = classifier(content[:1000], candidate_labels=labels)

                if result["scores"][0] < 0.5:
                    print(" → Rejeté (non tech)")
                    continue

                print(" → Accepté (tech)")

                writer.writerow({
                    "id": article_id,
                    "title": art["title"],
                    "url": art["url"],
                    "source": art["source"]["name"],
                    "published_at": art["publishedAt"],
                    "content": content,
                })
                article_id += 1

    print(f"[OK] Articles sauvegardés : {RAW_PATH}")


if __name__ == "__main__":
    main()
