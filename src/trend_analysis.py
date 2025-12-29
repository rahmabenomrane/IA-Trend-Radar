#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
trend_analysis.py (version ingénieur + agent IA)
- Lit src/data/summaries/articles_with_summaries.csv
- Extrait candidats keywords (TF-IDF ngrams)
- Filtre IT strict (lexique/whitelist + topics + règles)
- Valide via IA (Zero-shot classifier) que le terme est Tech
- Normalise les termes (ai -> artificial intelligence, iot -> internet of things, etc.)
- Calcule tendances émergentes (ancien vs récent)
- Exporte JSON/CSV dans src/data/trends/
"""

import json
import argparse
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

# NLP stopwords
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

# Zero-shot IA (agent validation)
try:
    from transformers import pipeline
    ZS_AVAILABLE = True
except Exception:
    ZS_AVAILABLE = False


# -----------------------
# Configuration
# -----------------------
ROOT = Path(__file__).resolve().parent
SUMMARIES_CSV = ROOT / "data" / "summaries" / "articles_with_summaries.csv"
TRENDS_DIR = ROOT / "data" / "trends"
TRENDS_DIR.mkdir(parents=True, exist_ok=True)

# Tes topics & whitelist
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

TECH_WHITELIST = {
    "ai", "ml", "it", "vr", "ar", "iot", "devops",
    "cloud", "cybersecurity", "blockchain",
    "data", "neural", "network", "robotics", "software"
}

# Normalisation (important pour éviter les doublons)
NORMALIZATION_MAP = {
    "ai": "artificial intelligence",
    "a.i": "artificial intelligence",
    "ml": "machine learning",
    "iot": "internet of things",
    "infosec": "cybersecurity",
    "secops": "cybersecurity",
    "devsecops": "devsecops",
    "llm": "large language model",
    "llms": "large language model",
    "genai": "generative ai",
    "gpt": "generative ai",
    "k8s": "kubernetes",
    "ci/cd": "cicd",
    "ci": "continuous integration",
    "cd": "continuous delivery",
}

DEFAULT_STOPWORDS = set([
    # FR basics
    "le","la","les","de","des","du","et","un","une","en","à","a","pour","par","sur",
    "est","sont","avec","que","qui","dans","au","aux","ce","ces","se","sa","son",
    "plus","ne","pas","ou","mais","comme","ans","année","années",
    # EN basics
    "the","a","an","and","or","of","to","in","on","for","with","as","is","are","was","were",
    "this","that","these","those","from","by","at","it","its","be","been","has","have",
    # news noise
    "said","says","report","reports","today","week","month","year","new","latest"
])

# Mots interdits (souvent “parachutés”)
NOISE_WORDS = set([
    "company","companies","market","stock","investors","investment","growth","money","people",
    "government","industry","business","world","global","news","media","platform"  # trop généraux
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
        DEFAULT_STOPWORDS |= set(stopwords.words("english"))
        DEFAULT_STOPWORDS |= set(stopwords.words("french"))
    except Exception:
        pass


def load_summaries(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Colonnes dans le CSV: {list(df.columns)}")

    # Trouver la colonne texte
    for col in ["summary", "summaries", "clean_text", "cleaned_text", "content", "text"]:
        if col in df.columns:
            df["__summary_text__"] = df[col].astype(str)
            break
    if "__summary_text__" not in df.columns:
        raise ValueError("Aucune colonne texte trouvée (summary/clean_text/content...).")

    # Convertir date si possible
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

    return df


def simple_preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_term(term: str) -> str:
    t = term.strip().lower()
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"\s+", " ", t)
    t = NORMALIZATION_MAP.get(t, t)
    return t


def looks_like_tech(term: str) -> bool:
    """Filtre rapide (règles + whitelist + topics) avant l'IA."""
    t = normalize_term(term)

    if not t:
        return False
    if len(t) <= 2 and t not in TECH_WHITELIST:
        return False
    if any(ch.isdigit() for ch in t) and t not in ("web3", "k8s"):
        # éviter les tokens bruités (sauf exceptions)
        return False
    if t in DEFAULT_STOPWORDS or t in NOISE_WORDS:
        return False

    # whitelist (termes courts IT)
    if t in TECH_WHITELIST:
        return True

    # contient un topic tech ?
    for topic in TECH_TOPICS:
        if topic in t or t in topic:
            return True

    # heuristiques: suffix/pattern IT
    tech_patterns = [
        r"\b(ai|ml|iot|cloud|cyber|blockchain|devops|robot|data|software)\b",
        r"\b(kubernetes|docker|microservices|cicd|slam|lidar|nav2)\b",
        r"\b(encryption|rsa|authentication|zero trust)\b",
        r"\b(neural|transformer|llm|agent|yolo)\b",
    ]
    for p in tech_patterns:
        if re.search(p, t):
            return True

    return False


# -----------------------
# Extraction candidats (TF-IDF)
# -----------------------
def extract_candidates_tfidf(documents, top_n=80, stopwords_set=None, ngram_range=(1, 2)):
    if stopwords_set is None:
        stopwords_set = set()

    vectorizer = TfidfVectorizer(
        max_features=8000,
        stop_words=list(stopwords_set),
        ngram_range=ngram_range,
        min_df=1
    )
    X = vectorizer.fit_transform(documents)
    feature_names = np.array(vectorizer.get_feature_names_out())
    mean_tfidf = X.mean(axis=0).A1
    top_indices = mean_tfidf.argsort()[::-1][:top_n]
    candidates = [(feature_names[i], float(mean_tfidf[i])) for i in top_indices]
    return candidates


# -----------------------
# Agent IA : validation “Tech ou non ?”
# -----------------------
class TechTrendAgent:
    """
    Agent simple (agentic pipeline):
    1) prend des candidats
    2) filtre par règles/lexique
    3) valide via IA zero-shot (si dispo)
    4) renvoie des trends propres
    """

    def __init__(self, threshold=0.55):
        self.threshold = threshold
        self.zs = None
        if ZS_AVAILABLE:
            try:
                self.zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            except Exception:
                self.zs = None

        self.labels = [
            "Technology / IT",
            "Artificial Intelligence",
            "Cybersecurity",
            "Cloud Computing",
            "Software Engineering",
            "Robotics",
            "Blockchain",
            "Data Science",
            "Non-technical / General news"
        ]

    def validate_term_with_ai(self, term: str) -> bool:
        """Zero-shot sur un mini contexte synthétique."""
        if self.zs is None:
            # fallback: si IA indispo, on accepte seulement via règles
            return True

        premise = f"This term is about: {term}."
        res = self.zs(premise, candidate_labels=self.labels)
        best_label = res["labels"][0]
        best_score = res["scores"][0]

        # Refuse si le modèle pense que c'est non-tech
        if best_label == "Non-technical / General news" and best_score >= self.threshold:
            return False

        # Accepte si score tech suffisamment haut
        return best_score >= self.threshold

    def filter_and_normalize(self, candidates):
        kept = []
        for term, score in candidates:
            t = normalize_term(term)

            if not looks_like_tech(t):
                continue

            if not self.validate_term_with_ai(t):
                continue

            kept.append((t, score))

        # Merge doublons
        merged = defaultdict(float)
        for t, s in kept:
            merged[t] += s
        return sorted(merged.items(), key=lambda x: x[1], reverse=True)


# -----------------------
# Emerging trends (ancien vs récent)
# -----------------------
def compute_word_frequencies(documents, stopwords_set=None):
    if stopwords_set is None:
        stopwords_set = set()
    cnt = Counter()
    for d in documents:
        text = simple_preprocess(d)
        toks = [t for t in re.split(r"\W+", text) if t and t not in stopwords_set]
        toks = [normalize_term(t) for t in toks]
        toks = [t for t in toks if looks_like_tech(t)]
        cnt.update(toks)
    return cnt


def detect_emerging_trends(df, text_col="__summary_text__", stopwords_set=None, top_n=30):
    if stopwords_set is None:
        stopwords_set = set()

    # ✅ correction: inclure published_at
    if "published_at" in df.columns and df["published_at"].notna().any():
        df = df.sort_values(by="published_at")
    else:
        print("[INFO] Aucune date exploitable — split par index.")

    texts = df[text_col].astype(str).tolist()
    if len(texts) < 4:
        return []

    mid = len(texts) // 2
    old_texts = texts[:mid]
    new_texts = texts[mid:]

    old_freq = compute_word_frequencies(old_texts, stopwords_set)
    new_freq = compute_word_frequencies(new_texts, stopwords_set)

    all_keys = set(old_freq.keys()) | set(new_freq.keys())
    trends = []
    for k in all_keys:
        o = old_freq.get(k, 0)
        n = new_freq.get(k, 0)
        growth = (n - o) / (o + 1e-9) if o > 0 else (float("inf") if n > 0 else 0.0)
        trends.append({"term": k, "old_count": int(o), "new_count": int(n), "growth": growth})

    trends_sorted = sorted(trends, key=lambda x: (float("inf") if x["growth"] == float("inf") else x["growth"]), reverse=True)
    return trends_sorted[:top_n]


# -----------------------
# Export helpers
# -----------------------
def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {path}")


def save_csv_terms(terms, path: Path):
    df = pd.DataFrame(terms, columns=["term", "score"])
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}")


# -----------------------
# Main pipeline
# -----------------------
def run_pipeline(csv_path=SUMMARIES_CSV,
                 top_candidates=120,
                 top_trends=40,
                 use_keybert=False,
                 agent_threshold=0.55):

    if nltk is not None:
        ensure_nltk_stopwords()
    stopwords_set = DEFAULT_STOPWORDS

    df = load_summaries(Path(csv_path))
    documents = df["__summary_text__"].fillna("").astype(str).tolist()
    docs_clean = [simple_preprocess(d) for d in documents]
    print(f"[INFO] Documents chargés: {len(docs_clean)}")

    # 1) Candidats TF-IDF
    candidates = extract_candidates_tfidf(docs_clean, top_n=top_candidates, stopwords_set=stopwords_set, ngram_range=(1,2))

    # 2) Agent IA (filtrage + validation + normalisation)
    agent = TechTrendAgent(threshold=agent_threshold)
    trends = agent.filter_and_normalize(candidates)
    trends = trends[:top_trends]

    # Exports principaux (pour le front)
    save_csv_terms(trends, TRENDS_DIR / "tech_trends.csv")
    save_json([{"term": t, "score": s} for t, s in trends], TRENDS_DIR / "tech_trends.json")

    # 3) Option KeyBERT (bonus)
    if use_keybert and KEYBERT_AVAILABLE:
        kb = KeyBERT()
        per_doc = []
        for idx, doc in enumerate(documents):
            try:
                kws = kb.extract_keywords(doc, top_n=5, keyphrase_ngram_range=(1,2))
                kws = [normalize_term(k) for k, _ in kws if looks_like_tech(k)]
                per_doc.append({"index": idx, "keywords": kws})
            except Exception:
                per_doc.append({"index": idx, "keywords": []})
        save_json(per_doc, TRENDS_DIR / "keybert_keywords_per_doc.json")

    # 4) Emerging trends (ancien vs récent)
    emerging = detect_emerging_trends(df, stopwords_set=stopwords_set, top_n=30)
    for e in emerging:
        if e["growth"] == float("inf"):
            e["growth"] = "inf"
        else:
            e["growth"] = float(e["growth"])
    save_json(emerging, TRENDS_DIR / "emerging_trends.json")

    print("[DONE] Résultats dans:", TRENDS_DIR)


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Trend analysis (Agent IA) - Veille technologique")
    ap.add_argument("--csv", type=str, default=str(SUMMARIES_CSV), help="Chemin vers articles_with_summaries.csv")
    ap.add_argument("--top_candidates", type=int, default=120, help="Nombre de candidats TF-IDF avant filtrage")
    ap.add_argument("--top_trends", type=int, default=40, help="Nombre de trends finaux exportés")
    ap.add_argument("--use_keybert", action="store_true", help="Active KeyBERT (optionnel)")
    ap.add_argument("--agent_threshold", type=float, default=0.55, help="Seuil de validation IA (0.5-0.7)")
    args = ap.parse_args()

    run_pipeline(
        csv_path=Path(args.csv),
        top_candidates=args.top_candidates,
        top_trends=args.top_trends,
        use_keybert=args.use_keybert,
        agent_threshold=args.agent_threshold
    )
