"""Model loading and recommendation logic shared by the app and tests."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "movie_id",
    "title",
    "year",
    "genres",
    "overview",
    "rating",
    "popularity",
    "tags",
}


@dataclass(frozen=True)
class Movie:
    movie_id: int
    title: str
    year: int | None
    genres: str
    overview: str
    rating: float


@dataclass(frozen=True)
class Recommendation(Movie):
    rank: int
    score: float


class MovieRecommender:
    """Read-only content-based movie recommender."""

    def __init__(self, movies: pd.DataFrame, similarity: np.ndarray) -> None:
        missing = REQUIRED_COLUMNS.difference(movies.columns)
        if missing:
            raise ValueError(f"Movie data is missing columns: {sorted(missing)}")
        if movies.empty:
            raise ValueError("Movie data is empty")
        if movies["movie_id"].duplicated().any():
            raise ValueError("Movie IDs must be unique")
        if similarity.shape != (len(movies), len(movies)):
            raise ValueError(
                "Similarity matrix shape does not match the movie table: "
                f"{similarity.shape} versus {len(movies)} rows"
            )
        if not np.isfinite(similarity).all():
            raise ValueError("Similarity matrix contains non-finite values")

        self._movies = movies.reset_index(drop=True).copy()
        self._similarity = similarity
        self._index_by_id = {
            int(movie_id): index
            for index, movie_id in enumerate(self._movies["movie_id"])
        }
        title_counts = self._movies["title"].value_counts()
        self._duplicate_titles = set(title_counts[title_counts > 1].index)

    @classmethod
    def from_files(cls, movies_path: Path, similarity_path: Path) -> "MovieRecommender":
        # These artifacts are produced locally by build_model.py. Pickle files from
        # unknown sources must not be loaded because pickle can execute code.
        with movies_path.open("rb") as movies_file:
            movie_data = pickle.load(movies_file)
        with similarity_path.open("rb") as similarity_file:
            similarity = pickle.load(similarity_file)

        movies = movie_data if isinstance(movie_data, pd.DataFrame) else pd.DataFrame(movie_data)
        return cls(movies, np.asarray(similarity))

    @property
    def movie_ids(self) -> list[int]:
        return [int(movie_id) for movie_id in self._movies["movie_id"]]

    def default_index(self, title: str) -> int:
        matches = self._movies.index[self._movies["title"] == title]
        return int(matches[0]) if len(matches) else 0

    def display_title(self, movie_id: int) -> str:
        row = self._row(movie_id)
        if row["title"] in self._duplicate_titles and pd.notna(row["year"]):
            return f"{row['title']} ({int(row['year'])})"
        return str(row["title"])

    def movie(self, movie_id: int) -> Movie:
        row = self._row(movie_id)
        return Movie(
            movie_id=int(row["movie_id"]),
            title=str(row["title"]),
            year=self._optional_year(row["year"]),
            genres=str(row["genres"]),
            overview=str(row["overview"]),
            rating=float(row["rating"]),
        )

    def recommend(self, movie_id: int, limit: int = 5) -> list[Recommendation]:
        if limit < 1:
            raise ValueError("Recommendation limit must be at least 1")

        selected_index = self._index(movie_id)
        scores = self._similarity[selected_index]
        popularity = self._movies["popularity"].to_numpy(dtype=float)
        candidates = np.arange(len(self._movies))
        candidates = candidates[candidates != selected_index]

        order = np.lexsort((-popularity[candidates], -scores[candidates]))
        selected_candidates = candidates[order[: min(limit, len(candidates))]]

        recommendations: list[Recommendation] = []
        for rank, index in enumerate(selected_candidates, start=1):
            row = self._movies.iloc[index]
            recommendations.append(
                Recommendation(
                    movie_id=int(row["movie_id"]),
                    title=str(row["title"]),
                    year=self._optional_year(row["year"]),
                    genres=str(row["genres"]),
                    overview=str(row["overview"]),
                    rating=float(row["rating"]),
                    rank=rank,
                    score=float(scores[index]),
                )
            )
        return recommendations

    def _index(self, movie_id: int) -> int:
        try:
            return self._index_by_id[int(movie_id)]
        except (KeyError, ValueError) as exc:
            raise KeyError(f"Unknown movie ID: {movie_id}") from exc

    def _row(self, movie_id: int) -> pd.Series:
        return self._movies.iloc[self._index(movie_id)]

    @staticmethod
    def _optional_year(value: object) -> int | None:
        return None if pd.isna(value) else int(value)
