import pandas as pd
from transformers import pipeline
from pathlib import Path

# IMPORTANT : reproduire exactement la structure attendue par trend_analysis.py
ROOT = Path(__file__).resolve().parent.parent   # remonte d’un dossier
SUMMARIES_DIR = ROOT / "src" / "data" / "summaries"
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = SUMMARIES_DIR / "articles_with_summaries.csv"


def summarize_articles(input_csv):
    df = pd.read_csv(input_csv)

    # Ton code doit créer la colonne EXACTEMENT nommée "summary"
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    summaries = []
    for txt in df["content"]:   # OU "cleaned_text" selon ton scraping
        if not isinstance(txt, str):
            summaries.append("")
            continue
        try:
            s = summarizer(txt, max_length=180, min_length=50, do_sample=False)
            summaries.append(s[0]["summary_text"])
        except:
            summaries.append("")

    df["summary"] = summaries

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] CSV produit pour trend_analysis.py : {OUTPUT_CSV}")


if __name__ == "__main__":
    summarize_articles("src/data/clean/articles_clean.csv")
