(function () {
  "use strict";

  const form = document.getElementById("search-form");
  const input = document.getElementById("product-input");
  const suggestionsList = document.getElementById("suggestions");
  const noteEl = document.getElementById("result-note");
  const resultsSection = document.getElementById("results");
  const searchBtn = document.getElementById("search-btn");

  const MAX_SUGGESTIONS = 8;
  const MIN_CHARS_FOR_SUGGESTIONS = 2;

  let catalog = [];
  let activeSuggestions = [];
  let activeIndex = -1;
  let debounceTimer = null;

  /** Progressive enhancement only - search still works via the API's own
   *  fallback tiers even if this list never loads. */
  async function loadCatalog() {
    try {
      const res = await fetch("/api/products");
      const data = await res.json();
      catalog = Array.isArray(data.products) ? data.products : [];
    } catch (err) {
      console.warn("Autocomplete list unavailable:", err);
    }
  }

  function closeSuggestions() {
    suggestionsList.hidden = true;
    suggestionsList.innerHTML = "";
    activeSuggestions = [];
    activeIndex = -1;
  }

  function renderSuggestions(query) {
    const trimmed = query.trim().toLowerCase();
    if (trimmed.length < MIN_CHARS_FOR_SUGGESTIONS) {
      closeSuggestions();
      return;
    }

    const matches = [];
    for (let i = 0; i < catalog.length && matches.length < MAX_SUGGESTIONS; i++) {
      if (catalog[i].toLowerCase().includes(trimmed)) matches.push(catalog[i]);
    }

    activeSuggestions = matches;
    activeIndex = -1;
    suggestionsList.innerHTML = "";

    if (matches.length === 0) {
      closeSuggestions();
      return;
    }

    matches.forEach((name, i) => {
      const li = document.createElement("li");
      li.textContent = name;
      li.setAttribute("role", "option");
      li.addEventListener("mousedown", (e) => {
        e.preventDefault();
        selectSuggestion(name);
      });
      li.addEventListener("mouseenter", () => setActiveIndex(i));
      suggestionsList.appendChild(li);
    });

    suggestionsList.hidden = false;
  }

  function setActiveIndex(i) {
    const items = suggestionsList.querySelectorAll("li");
    items.forEach((item) => item.classList.remove("is-active"));
    activeIndex = i;
    if (i >= 0 && items[i]) {
      items[i].classList.add("is-active");
      items[i].scrollIntoView({ block: "nearest" });
    }
  }

  function selectSuggestion(name) {
    input.value = name;
    closeSuggestions();
    input.focus();
  }

  input.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    const value = input.value;
    debounceTimer = setTimeout(() => renderSuggestions(value), 100);
  });

  input.addEventListener("keydown", (e) => {
    if (suggestionsList.hidden) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex(Math.min(activeIndex + 1, activeSuggestions.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex(Math.max(activeIndex - 1, 0));
    } else if (e.key === "Enter" && activeIndex >= 0) {
      e.preventDefault();
      selectSuggestion(activeSuggestions[activeIndex]);
    } else if (e.key === "Escape") {
      closeSuggestions();
    }
  });

  document.addEventListener("click", (e) => {
    if (!form.contains(e.target)) closeSuggestions();
  });

  function tagFor(item) {
    if (typeof item.similarity === "number") {
      return Math.round(item.similarity * 100) + "% match";
    }
    if (typeof item.rating === "number") {
      return "\u2605 " + item.rating.toFixed(1);
    }
    return "Top pick";
  }

  function setLoading(isLoading) {
    searchBtn.disabled = isLoading;
    searchBtn.classList.toggle("is-loading", isLoading);
    if (isLoading) {
      noteEl.hidden = true;
      resultsSection.innerHTML = "";
      const loading = document.createElement("p");
      loading.className = "loading-text";
      loading.textContent = "Searching...";
      resultsSection.appendChild(loading);
    }
  }

  function renderMessage(text, isError) {
    noteEl.hidden = true;
    resultsSection.innerHTML = "";
    const message = document.createElement("p");
    message.className = "state-message" + (isError ? " is-error" : "");
    message.textContent = text;
    resultsSection.appendChild(message);
  }

  function renderResults(data) {
    resultsSection.innerHTML = "";

    if (data.note) {
      noteEl.textContent = data.note;
      noteEl.hidden = false;
    } else {
      noteEl.hidden = true;
    }

    if (!data.recommendations || data.recommendations.length === 0) {
      renderMessage("No matches found.", false);
      return;
    }

    data.recommendations.forEach((item) => {
      const card = document.createElement("article");
      card.className = "result-card";

      const category = document.createElement("p");
      category.className = "result-category";
      category.textContent = item.sub_category || item.category || "";

      const title = document.createElement("h3");
      title.className = "result-title";
      title.textContent = item.product;

      const footer = document.createElement("div");
      footer.className = "result-footer";

      const tag = document.createElement("span");
      tag.className = "result-tag";
      tag.textContent = tagFor(item);
      footer.appendChild(tag);

      card.appendChild(category);
      card.appendChild(title);
      card.appendChild(footer);
      resultsSection.appendChild(card);
    });
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    closeSuggestions();
    setLoading(true);

    try {
      const res = await fetch("/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ product_name: query }),
      });
      const data = await res.json();

      if (!res.ok) {
        renderMessage(data.error || "Something went wrong. Please try again.", true);
      } else {
        renderResults(data);
      }
    } catch (err) {
      renderMessage("Could not reach the recommendation service.", true);
    } finally {
      setLoading(false);
    }
  });

  loadCatalog();
})();
