"""Build the movie table and similarity matrix from the two TMDB CSV files."""

from __future__ import annotations

import argparse
import ast
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


PROJECT_ROOT = Path(__file__).resolve().parent
TOKEN_CLEANER = re.compile(r"[^a-z0-9_]+")


def parse_names(value: str, limit: int | None = None) -> list[str]:
    """Extract names from a TMDB JSON-style list stored as text."""
    if pd.isna(value):
        return []
    records = ast.literal_eval(value)
    names = [str(record["name"]) for record in records if record.get("name")]
    return names[:limit] if limit else names


def parse_director(value: str) -> list[str]:
    if pd.isna(value):
        return []
    for record in ast.literal_eval(value):
        if record.get("job") == "Director" and record.get("name"):
            return [str(record["name"])]
    return []


def entity_token(value: str) -> str:
    """Keep multiword people and genre names together as one token."""
    normalized = value.lower().replace(" ", "_")
    return TOKEN_CLEANER.sub("", normalized)


def make_tags(row: pd.Series) -> str:
    overview = str(row["overview"]).lower()
    genres = [entity_token(value) for value in row["genre_names"]]
    keywords = [entity_token(value) for value in row["keyword_names"]]
    cast = [entity_token(value) for value in row["cast_names"]]
    directors = [entity_token(value) for value in row["director_names"]]

    # Repetition gives structured fields more influence than incidental words in
    # a synopsis while keeping the model simple enough to explain.
    tokens = genres * 3 + keywords * 2 + cast * 2 + directors * 3
    return " ".join([overview, *tokens])


def prepare_movies(movies_csv: Path, credits_csv: Path) -> pd.DataFrame:
    movies = pd.read_csv(movies_csv)
    credits = pd.read_csv(credits_csv)

    credits = credits[["movie_id", "cast", "crew"]]
    merged = movies.merge(
        credits,
        left_on="id",
        right_on="movie_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.dropna(subset=["overview"]).copy()

    merged["genre_names"] = merged["genres"].apply(parse_names)
    merged["keyword_names"] = merged["keywords"].apply(parse_names)
    merged["cast_names"] = merged["cast"].apply(lambda value: parse_names(value, limit=3))
    merged["director_names"] = merged["crew"].apply(parse_director)

    prepared = pd.DataFrame(
        {
            "movie_id": merged["id"].astype("int64"),
            "title": merged["title"].astype(str),
            "year": pd.to_datetime(merged["release_date"], errors="coerce").dt.year.astype("Int64"),
            "genres": merged["genre_names"].apply(lambda names: " · ".join(names)),
            "overview": merged["overview"].astype(str),
            "rating": merged["vote_average"].fillna(0).astype("float32"),
            "popularity": merged["popularity"].fillna(0).astype("float32"),
        }
    )
    prepared["tags"] = merged.apply(make_tags, axis=1)
    prepared = prepared.reset_index(drop=True)

    if prepared["movie_id"].duplicated().any():
        raise ValueError("Prepared movie table contains duplicate movie IDs")
    return prepared


def calculate_similarity(tags: pd.Series) -> tuple[np.ndarray, int]:
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        min_df=2,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        dtype=np.float32,
    )
    vectors = vectorizer.fit_transform(tags)
    similarity = linear_kernel(vectors, vectors).astype(np.float32, copy=False)
    np.fill_diagonal(similarity, 1.0)
    return similarity, len(vectorizer.get_feature_names_out())


def save_artifacts(movies: pd.DataFrame, similarity: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "movies_dict.pkl").open("wb") as movie_file:
        pickle.dump(movies.to_dict(orient="list"), movie_file, protocol=pickle.HIGHEST_PROTOCOL)
    with (output_dir / "movies.pkl").open("wb") as legacy_movie_file:
        pickle.dump(movies, legacy_movie_file, protocol=pickle.HIGHEST_PROTOCOL)
    with (output_dir / "similarity.pkl").open("wb") as similarity_file:
        pickle.dump(similarity, similarity_file, protocol=pickle.HIGHEST_PROTOCOL)


def build(data_dir: Path = PROJECT_ROOT / "Data", output_dir: Path = PROJECT_ROOT) -> None:
    movies = prepare_movies(
        data_dir / "tmdb_5000_movies.csv",
        data_dir / "tmdb_5000_credits.csv",
    )
    similarity, feature_count = calculate_similarity(movies["tags"])
    save_artifacts(movies, similarity, output_dir)
    print(
        f"Built {len(movies):,} movies with {feature_count:,} TF-IDF features. "
        f"Similarity matrix: {similarity.shape}, {similarity.dtype}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "Data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build(arguments.data_dir, arguments.output_dir)
