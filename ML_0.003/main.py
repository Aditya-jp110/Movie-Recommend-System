import warnings
warnings.filterwarnings("ignore")

import ast
import numpy as np
import pandas as pd

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- 1. SAFE HELPERS ----------

def safe_literal_eval(obj):
    """Safely parse string representation of list/dict; return [] on error."""
    if isinstance(obj, list):
        return obj
    if pd.isna(obj):
        return []
    try:
        return ast.literal_eval(obj)
    except (ValueError, SyntaxError):
        return []


def convert_genres_or_keywords(obj):
    """Extract 'name' from list of dicts (genres/keywords)."""
    data = safe_literal_eval(obj)
    result = []
    for d in data:
        if isinstance(d, dict):
            name = d.get("name")
            if name:
                # remove spaces inside names like "Science Fiction" -> "ScienceFiction"
                result.append(name.replace(" ", ""))
    return result


def convert_cast(obj, max_actors=3):
    """Extract up to max_actors cast names."""
    data = safe_literal_eval(obj)
    result = []
    for d in data:
        if len(result) >= max_actors:
            break
        if isinstance(d, dict):
            name = d.get("name")
            if name:
                result.append(name.replace(" ", ""))
    return result


def fetch_director(obj):
    """Extract director name from crew."""
    data = safe_literal_eval(obj)
    for d in data:
        if isinstance(d, dict) and d.get("job") == "Director":
            name = d.get("name")
            if name:
                return [name.replace(" ", "")]
    return []


ps = PorterStemmer()


def stem(text: str) -> str:
    """Apply Porter stemming to each word."""
    return " ".join(ps.stem(word) for word in text.split())


# ---------- 2. LOAD & PREPARE DATA ----------

def load_and_prepare_data(movies_path="movies.csv", credits_path="credits.csv"):
    print("Loading CSV files...")

    movies = pd.read_csv(movies_path, encoding="latin1", errors="ignore")
    credits = pd.read_csv(credits_path, encoding="latin1", errors="ignore")

    # Merge on common column 'title'
    movies = movies.merge(credits, on="title")

    # Keep only required columns
    movies = movies[['genres', 'id', 'keywords', 'title', 'overview', 'cast', 'crew']]

    # Drop rows with missing important fields
    movies.dropna(inplace=True)

    # Convert overview to list of words
    movies['overview'] = movies['overview'].astype(str).apply(lambda x: x.split())

    # Convert JSON-like text fields to lists of names
    movies['genres'] = movies['genres'].apply(convert_genres_or_keywords)
    movies['keywords'] = movies['keywords'].apply(convert_genres_or_keywords)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(fetch_director)

    # Create 'tags' by combining lists
    movies['tags'] = (
        movies['overview'] +
        movies['genres'] +
        movies['keywords'] +
        movies['cast'] +
        movies['crew']
    )

    # New dataframe with only necessary fields
    new_df = movies[['id', 'title', 'tags']].copy()

    # Convert list -> string and lowercase
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

    # Apply stemming
    new_df['tags'] = new_df['tags'].apply(stem)

    return new_df


# ---------- 3. BUILD VECTORS & SIMILARITY MATRIX ----------

def build_similarity_matrix(new_df):
    print("Vectorizing tags and building similarity matrix...")

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    similarity = cosine_similarity(vectors)
    return similarity


# ---------- 4. RECOMMEND FUNCTION ----------

def create_recommender(new_df, similarity):
    """Return a recommend(movie_name) function bound to this dataset."""

    def recommend(movie):
        # Make search case-insensitive
        movie_clean = movie.strip().lower()

        # Find title ignoring case
        matches = new_df[new_df['title'].str.lower() == movie_clean]

        if matches.empty:
            return [f"'{movie}' not found in database. Check spelling or try another movie."]

        movie_index = matches.index[0]
        distances = similarity[movie_index]

        movies_list = sorted(
            list(enumerate(distances)),
            reverse=True,
            key=lambda x: x[1]
        )[1:6]  # skip the movie itself

        recommendations = [new_df.iloc[i[0]].title for i in movies_list]
        return recommendations

    return recommend


# ---------- 5. MAIN (COMMAND-LINE APP) ----------

def main():
    # Load, process, and build model once
    new_df = load_and_prepare_data()
    similarity = build_similarity_matrix(new_df)
    recommend = create_recommender(new_df, similarity)

    print("\n🎬 Movie Recommender System")
    print("Type a movie name and get similar movies.")
    print("Type 'q' or 'quit' to exit.\n")

    while True:
        movie_name = input("Enter movie name: ").strip()
        if movie_name.lower() in ("q", "quit", "exit"):
            print("Goodbye! 👋")
            break

        results = recommend(movie_name)
        print("\nRecommended movies:")
        for r in results:
            print("  -", r)
        print()  # blank line

if __name__ == "__main__":
    main()
