# Model artifacts

| File | Size | What it holds |
|---|---|---|
| `df_recommender.pkl` | 9.9 MB | 23,449 products: `product`, `category`, `sub_category`, `sale_price`, `rating` (32% null), `tags_clean` |
| `tfidf_matrix.pkl` | 10.3 MB | Sparse TF-IDF matrix over `tags_clean`, 0.77% dense |
| `tfidf_vectorizer.pkl` | 0.2 MB | The fitted `TfidfVectorizer` itself - needed to project a live search query into the same vector space, not just to look up existing rows |

`../nltk_data/` (36 MB, repo root) ships the WordNet corpus the lemmatizer
needs, so there's no network call to a corpus server at build or first
request. `api/text_utils.py` points NLTK at it directly by path.

## What changed in this revision

The retrieval logic moved from "look up this exact product, return similar
items" to a four-tier search (exact -> fuzzy name -> free-text TF-IDF ->
sub-category fallback) - see `api/recommender.py`'s module docstring for the
full design. That required two changes here:

- `category`, `sub_category`, and `rating` are kept as real columns now,
  not folded into the text blob and discarded. The fallback tier and the
  frontend's category tag both need them.
- The fitted `TfidfVectorizer` is saved (`tfidf_vectorizer.pkl`), so a raw
  search string can be `.transform()`-ed into the same space the catalog
  lives in, rather than only ever comparing existing rows to each other.
  Catalog text is now also lemmatized before fitting (`tags_clean`), so a
  query goes through the identical cleaning step in `api/text_utils.py` and
  lands in a vocabulary that actually matches it.

`rating` is retained with nulls as-is (32% of products don't have one, per
the source data) rather than imputed - the sub-category fallback sorts
rated items first and treats unrated ones as lowest priority, instead of
guessing a rating that isn't there.

## Background: why these files were rebuilt from the CSV rather than reused

The `df_recommender.pkl` this project originally shipped with couldn't be
unpickled with any numpy/pandas/joblib version tried. Its raw bytes showed
`EF BF BD` repeated throughout - the UTF-8 encoding of the Unicode
replacement character, the signature of a binary file having been decoded
as text somewhere (with invalid bytes replaced) and re-saved. That's
destructive; no version-matching could have recovered it. Both this file
and its predecessor were instead rebuilt by re-running the documented
cleaning steps against the original `BigBasket_Products.csv`, verified to
match the source notebook's own recorded row count (23,449).

That first rebuild also computed `similarity_matrix.pkl`, the notebook's
precomputed dense 23,449 x 23,449 cosine-similarity array - a verified
4.4 GB as float64, too large for GitHub or Render's free tier. That's the
reason this project takes the on-demand `cosine_similarity(query_vector,
tfidf_matrix)` approach rather than precomputing and storing every pairwise
score: it was true before the retrieval logic changed and it's still true
now, independent of how the query side works.
