#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trend_analysis.py
- Lit src/data/summaries/articles_with_summaries.csv
- Extrait keywords (TF-IDF, et KeyBERT si disponible)
- Calcule top words (fréquence)
- Detecte tendances émergentes (comparaison ancien vs récent)
- Exporte JSON/CSV dans src/data/trends/
"""

import os
import json
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import re
import string

# NLP
try:
    import nltk
    from nltk.corpus import stopwords
except Exception:
    nltk = None

from sklearn.feature_extraction.text import TfidfVectorizer

# Optional KeyBERT
try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except Exception:
    KEYBERT_AVAILABLE = False

# -----------------------
# Configuration
# -----------------------
ROOT = Path(__file__).resolve().parent
SUMMARIES_CSV = ROOT / "data" / "summaries" / "articles_with_summaries.csv"
TRENDS_DIR = ROOT / "data" / "trends"
TRENDS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_STOPWORDS = set([
    "le","la","les","de","des","du","et","un","une","en","à","a","pour","par","sur",
    "est","sont","avec","que","qui","dans","au","aux","ce","ces","se","sa","son",
    "plus","ne","pas","ou","mais","comme","ou","ans","année","années"
])

# -----------------------
# Utilitaires
# -----------------------
def ensure_nltk_stopwords():
    global DEFAULT_STOPWORDS
    if nltk is None:
        return
    try:
        nltk.data.find("corpora/stopwords")
    except Exception:
        nltk.download("stopwords")
    try:
        DEFAULT_STOPWORDS = set(stopwords.words("french"))
    except Exception:
        DEFAULT_STOPWORDS = set(stopwords.words("english"))

def load_summaries(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Colonnes dans le CSV: {list(df.columns)}")
    for col in ['summary', 'summaries', 'cleaned_text', 'clean_text', 'text', 'content']:
        if col in df.columns:
            df['__summary_text__'] = df[col].astype(str)
            return df
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if text_cols:
        df['__summary_text__'] = df[text_cols[0]].astype(str)
        return df
    raise ValueError("Aucune colonne texte trouvée dans le CSV.")

def simple_preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_remove_stopwords(text, stopwords_set):
    toks = [t for t in re.split(r'\W+', text) if t and len(t) > 1]
    toks = [t for t in toks if t not in stopwords_set]
    return toks

# -----------------------
# TF-IDF keywords
# -----------------------
def get_top_tfidf_keywords(documents, top_n=10, max_features=5000, stopwords_set=None, ngram_range=(1,1)):
    if stopwords_set is None:
        stopwords_set = set()
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=list(stopwords_set),  # <-- correction : convertir set en liste
        ngram_range=ngram_range
    )
    X = vectorizer.fit_transform(documents)
    feature_names = np.array(vectorizer.get_feature_names_out())
    mean_tfidf = X.mean(axis=0).A1
    top_indices = mean_tfidf.argsort()[::-1][:top_n]
    top_terms = [{"term": feature_names[i], "score": float(mean_tfidf[i])} for i in top_indices]
    return top_terms

# -----------------------
# KeyBERT keywords per document (optional)
# -----------------------
def get_keybert_keywords(documents, num_keywords=5):
    if not KEYBERT_AVAILABLE:
        print("[WARN] keyBERT non installé — passe au TF-IDF uniquement.")
        return None
    model = KeyBERT()
    doc_kws = []
    for doc in documents:
        try:
            kws = model.extract_keywords(doc, top_n=num_keywords, keyphrase_ngram_range=(1,2))
            doc_kws.append([kw for kw,score in kws])
        except Exception:
            doc_kws.append([])
    return doc_kws

# -----------------------
# Fréquences et top words
# -----------------------
def compute_word_frequencies(documents, stopwords_set=None):
    if stopwords_set is None:
        stopwords_set = set()
    cnt = Counter()
    for d in documents:
        toks = tokenize_and_remove_stopwords(simple_preprocess(d), stopwords_set)
        cnt.update(toks)
    return cnt

# -----------------------
# Détection tendances émergentes
# -----------------------
def detect_emerging_trends(df, text_col='__summary_text__', date_col_candidates=None, stopwords_set=None, top_n=20):
    if stopwords_set is None:
        stopwords_set = set()
    if date_col_candidates is None:
        date_col_candidates = ['date', 'published', 'pub_date', 'timestamp', 'created_at']
    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col:
        print(f"[INFO] Colonne date utilisée pour trier: {date_col}")
        df = df.sort_values(by=date_col)
    else:
        print("[INFO] Aucune colonne date trouvée — split par index.")

    texts = df[text_col].astype(str).tolist()
    if len(texts) < 2:
        return []

    mid = len(texts) // 2
    old_texts = texts[:mid]
    new_texts = texts[mid:]

    old_freq = compute_word_frequencies(old_texts, stopwords_set)
    new_freq = compute_word_frequencies(new_texts, stopwords_set)

    all_keys = set(list(old_freq.keys()) + list(new_freq.keys()))
    trends = []
    for k in all_keys:
        o = old_freq.get(k, 0)
        n = new_freq.get(k, 0)
        if o == 0 and n > 0:
            growth = float('inf')
        elif o == 0 and n == 0:
            growth = 0.0
        else:
            growth = (n - o) / (o + 1e-9)
        trends.append({"word": k, "old_count": int(o), "new_count": int(n), "growth": growth})

    trends_sorted = sorted(trends, key=lambda x: (float('inf') if x['growth']==float('inf') else x['growth']), reverse=True)
    return trends_sorted[:top_n]

# -----------------------
# Export helpers
# -----------------------
def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {path}")

def save_csv_from_counter(counter: Counter, path: Path, top_n=None):
    items = counter.most_common(top_n)
    df = pd.DataFrame(items, columns=['word', 'count'])
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}")

# -----------------------
# Main pipeline
# -----------------------
def run_pipeline(csv_path=SUMMARIES_CSV,
                 top_n_words=50,
                 top_n_keywords=10,
                 use_keybert=False):
    if nltk is not None:
        ensure_nltk_stopwords()
    stopwords_set = DEFAULT_STOPWORDS

    df = load_summaries(Path(csv_path))
    documents = df['__summary_text__'].fillna("").astype(str).tolist()
    print(f"[INFO] Documents chargés: {len(documents)}")

    docs_clean = [simple_preprocess(d) for d in documents]

    # Top words
    freq = compute_word_frequencies(docs_clean, stopwords_set)
    save_csv_from_counter(freq, TRENDS_DIR / "top_words.csv", top_n=top_n_words)
    top_words_json = [{"word": w, "count": c} for w, c in freq.most_common(top_n_words)]
    save_json(top_words_json, TRENDS_DIR / "top_words.json")

    # TF-IDF keywords
    tfidf_kws = get_top_tfidf_keywords(docs_clean, top_n=top_n_keywords, stopwords_set=stopwords_set, ngram_range=(1,2))
    save_json(tfidf_kws, TRENDS_DIR / "tfidf_keywords.json")

    # KeyBERT per document (optionnel)
    if use_keybert and KEYBERT_AVAILABLE:
        keybert_kws = get_keybert_keywords(documents, num_keywords=5)
        rows = [{"index": idx, "keywords": kws} for idx, kws in enumerate(keybert_kws)]
        save_json(rows, TRENDS_DIR / "keybert_keywords_per_doc.json")
    elif use_keybert and not KEYBERT_AVAILABLE:
        print("[WARN] keyBERT demandé mais non installé. Ignoré.")

    # Emerging trends
    trends = detect_emerging_trends(df, text_col='__summary_text__', stopwords_set=stopwords_set, top_n=50)
    for t in trends:
        if t['growth'] == float('inf'):
            t['growth'] = "inf"
        else:
            t['growth'] = float(t['growth'])
    save_json(trends, TRENDS_DIR / "emerging_trends.json")

    print("[DONE] Pipeline terminé. Résultats dans:", TRENDS_DIR)

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Trend analysis pipeline (Personne 2)")
    ap.add_argument("--csv", type=str, default=str(SUMMARIES_CSV),
                    help="Chemin vers articles_with_summaries.csv")
    ap.add_argument("--top_words", type=int, default=50, help="Nombre top mots (freq)")
    ap.add_argument("--top_keywords", type=int, default=20, help="Nombre top keywords TF-IDF")
    ap.add_argument("--use_keybert", action="store_true", help="Si set, tente KeyBERT (nécessite installation)")
    args = ap.parse_args()

    run_pipeline(csv_path=Path(args.csv), top_n_words=args.top_words, top_n_keywords=args.top_keywords, use_keybert=args.use_keybert)
