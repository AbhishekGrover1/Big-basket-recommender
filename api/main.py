"""BigBasket AI Recommender - Flask API.

Thin HTTP layer: request parsing and status codes live here, retrieval
logic lives in api/recommender.py. Both are constructed once at process
startup (see ProductRecommender's own docstring for why) and held for the
life of the worker.
"""

import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from api.recommender import ProductRecommender

BASE_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bigbasket_recommender")

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

try:
    recommender = ProductRecommender(BASE_DIR / "models")
except FileNotFoundError as exc:
    logger.error("Model artifact not found: %s", exc)
    raise
except Exception as exc:
    logger.error("Failed to initialize recommender (%s): %s", type(exc).__name__, exc)
    raise


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "catalog_size": int(len(recommender.catalog))})


@app.get("/api/products")
def list_products():
    """Product names for the search box's autocomplete."""
    return jsonify({"products": recommender.catalog["product"].tolist()})


@app.post("/recommend")
def recommend():
    payload = request.get_json(silent=True) or {}
    query = (payload.get("product_name") or "").strip()

    if not query:
        return jsonify({"error": "Enter a product or search term."}), 400

    result = recommender.recommend(query)
    response = {"query": query, "recommendations": result.recommendations}
    if result.note:
        response["note"] = result.note
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
