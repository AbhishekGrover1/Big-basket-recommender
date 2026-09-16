"""Text normalization shared between the offline catalog build and live queries.

Kept as plain functions (no custom callables baked into the vectorizer) so
the fitted TfidfVectorizer stays a plain, portable pickle - passing a
tokenizer/preprocessor callable into TfidfVectorizer ties the pickle to
this exact module path forever, which is the same class of bug that broke
the model files this project shipped with originally.
"""

import re
from pathlib import Path

import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Point at the corpora baked in at build time (see render.yaml's buildCommand)
# rather than trusting NLTK_DATA to be set in whatever environment this runs
# in - this is the same "resolve paths off __file__, not the environment"
# rule the rest of this project follows for model artifacts.
_NLTK_DATA_DIR = Path(__file__).resolve().parent.parent / "nltk_data"
if _NLTK_DATA_DIR.exists():
    nltk.data.path.insert(0, str(_NLTK_DATA_DIR))

from nltk.stem import WordNetLemmatizer  # noqa: E402  (path must be set first)

_TOKEN_PATTERN = re.compile(r"[a-z]+")
_MIN_TOKEN_LENGTH = 3
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, tokenize, drop stopwords/short tokens, lemmatize.

    Applied identically to catalog tags at build time and to every
    incoming search query, so both sides land in the same vocabulary.
    """
    if not text:
        return ""

    tokens = _TOKEN_PATTERN.findall(text.lower())
    kept = (t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) >= _MIN_TOKEN_LENGTH)
    return " ".join(_lemmatizer.lemmatize(t) for t in kept)
