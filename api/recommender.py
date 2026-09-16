"""Content-based recommendation with graduated fallback.

Four tiers, tried in order, so a search only comes back empty if the
catalog itself has nothing left to offer:

  1. Exact match on a normalized product name - the fast path for
     autocomplete-driven searches.
  2. Fuzzy match on product name - absorbs typos and partial titles
     ("grlic oil capsule" -> the real product).
  3. Free-text similarity - the query is cleaned with the same
     lowercase/stopword/lemmatize pipeline used on the catalog, projected
     into the fitted TF-IDF space, and compared by cosine similarity
     against every product. This is what actually serves descriptive
     queries that were never going to match a product *name*.
  4. Sub-category fallback - when even the best free-text score is weak,
     fall back to the highest-rated items in the closest-matching
     sub-category rather than surfacing a handful of near-zero scores.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity

from api.text_utils import clean_text

logger = logging.getLogger("bigbasket_recommender")

FUZZY_NAME_THRESHOLD = 82
MIN_TOKEN_COVERAGE = 0.75
PER_TOKEN_MATCH_THRESHOLD = 78
WEAK_MATCH_FLOOR = 0.08
DEFAULT_TOP_N = 5

_WORD_RE = re.compile(r"[a-z0-9]+")


def _token_coverage(query: str, candidate: str) -> float:
    """Fraction of the query's words that have a close per-word match in `candidate`.

    A plain whole-string fuzzy ratio between a short query and a long
    catalog title (or vice versa) is easy to fool: with 23k candidates,
    something will always align well by coincidence somewhere in the
    string. Checking word by word instead asks the question that actually
    matters here - "is every real word in this query accounted for in the
    title" - which a coincidental character-level alignment won't satisfy.
    """
    query_tokens = _WORD_RE.findall(query.lower())
    candidate_tokens = _WORD_RE.findall(candidate.lower())
    if not query_tokens or not candidate_tokens:
        return 0.0

    matched = sum(
        1
        for qt in query_tokens
        if max(fuzz.ratio(qt, ct) for ct in candidate_tokens) >= PER_TOKEN_MATCH_THRESHOLD
    )
    return matched / len(query_tokens)


@dataclass
class RecommendationResult:
    recommendations: list
    note: Optional[str] = None


class ProductRecommender:
    """Loads catalog + TF-IDF artifacts once and serves ranked lookups."""

    def __init__(self, models_dir: Path):
        self.catalog = joblib.load(models_dir / "df_recommender.pkl")
        self.tfidf_matrix = joblib.load(models_dir / "tfidf_matrix.pkl")
        self.vectorizer = joblib.load(models_dir / "tfidf_vectorizer.pkl")

        if self.catalog.shape[0] != self.tfidf_matrix.shape[0]:
            raise ValueError(
                f"Catalog has {self.catalog.shape[0]} rows but the TF-IDF matrix "
                f"has {self.tfidf_matrix.shape[0]} - they must describe the same catalog."
            )

        normalized_names = self.catalog["product"].str.strip().str.lower()
        name_lookup = pd.Series(self.catalog.index, index=normalized_names)
        self._name_lookup = name_lookup[~name_lookup.index.duplicated(keep="first")]
        self._product_names = self._name_lookup.index.tolist()

        # Per sub-category, row positions sorted by rating desc (unrated
        # last), precomputed once so tier 4 is a slice, not a per-request sort.
        self._subcategory_rank = {
            sub_cat: group.sort_values("rating", ascending=False, na_position="last").index.tolist()
            for sub_cat, group in self.catalog.groupby("sub_category")
        }
        self._subcategory_names = list(self._subcategory_rank.keys())

        # WordNet lazily indexes its corpus files on first use (a multi-second
        # stall) - pay that cost once here instead of on some user's first
        # free-text search.
        clean_text("warm up")

        logger.info(
            "Recommender ready: %s products across %s sub-categories.",
            f"{len(self.catalog):,}",
            len(self._subcategory_names),
        )

    def recommend(self, raw_query: str, top_n: int = DEFAULT_TOP_N) -> RecommendationResult:
        query = raw_query.strip()

        exact_idx = self._exact_match(query)
        if exact_idx is not None:
            return self._similar_to_row(exact_idx, top_n)

        matched_name, fuzzy_idx = self._fuzzy_name_match(query)
        if fuzzy_idx is not None:
            result = self._similar_to_row(fuzzy_idx, top_n)
            result.note = f'Showing results for "{matched_name}"'
            return result

        best_idx, best_score, ranked = self._free_text_search(query, top_n)
        if best_score >= WEAK_MATCH_FLOOR:
            return RecommendationResult(recommendations=ranked)

        return self._subcategory_fallback(query, top_n, weak_hit_idx=best_idx)

    def _exact_match(self, query: str) -> Optional[int]:
        key = query.lower()
        return int(self._name_lookup[key]) if key in self._name_lookup.index else None

    def _fuzzy_name_match(self, query: str):
        # WRatio over the full catalog is the cheap way to find a plausible
        # candidate among 23k names; it's also easy for it to fool itself on
        # a query it shouldn't have matched at all, so that candidate still
        # has to clear the word-level coverage check below before it's used.
        match = process.extractOne(query.lower(), self._product_names, scorer=fuzz.WRatio)
        if match is None or match[1] < FUZZY_NAME_THRESHOLD:
            return None, None

        matched_name = match[0]
        if _token_coverage(query, matched_name) < MIN_TOKEN_COVERAGE:
            return None, None

        idx = int(self._name_lookup[matched_name])
        return self.catalog.iloc[idx]["product"], idx

    def _similar_to_row(self, idx: int, top_n: int) -> RecommendationResult:
        sim_row = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        return RecommendationResult(recommendations=self._rank(sim_row, {idx}, top_n))

    def _free_text_search(self, query: str, top_n: int):
        cleaned = clean_text(query)
        if not cleaned:
            return None, 0.0, []

        query_vector = self.vectorizer.transform([cleaned])
        sim_row = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        if not sim_row.size:
            return None, 0.0, []

        best_idx = int(sim_row.argmax())
        return best_idx, float(sim_row[best_idx]), self._rank(sim_row, set(), top_n)

    def _subcategory_fallback(self, query: str, top_n: int, weak_hit_idx: Optional[int]) -> RecommendationResult:
        if weak_hit_idx is not None:
            target = self.catalog.iloc[weak_hit_idx]["sub_category"]
        else:
            match = process.extractOne(query.lower(), self._subcategory_names, scorer=fuzz.WRatio)
            target = match[0] if match else self._subcategory_names[0]

        positions = self._subcategory_rank[target][:top_n]
        return RecommendationResult(
            recommendations=self._records(positions, scores=None),
            note=f'No close match for "{query}" - here are top-rated picks in {target}',
        )

    def _rank(self, sim_row, exclude: set, top_n: int) -> list:
        order = sim_row.argsort()[::-1]
        positions = [int(i) for i in order if i not in exclude][:top_n]
        scores = [round(float(sim_row[i]), 4) for i in positions]
        return self._records(positions, scores)

    def _records(self, positions: list, scores: Optional[list]) -> list:
        rows = self.catalog.iloc[positions]
        records = rows[["product", "category", "sub_category", "sale_price"]].to_dict(orient="records")
        for i, record in enumerate(records):
            record["similarity"] = scores[i] if scores else None
            rating = rows.iloc[i]["rating"]
            record["rating"] = None if pd.isna(rating) else float(rating)
        return records
