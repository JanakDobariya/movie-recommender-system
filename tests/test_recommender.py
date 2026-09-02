import unittest

import numpy as np
import pandas as pd

from recommender import MovieRecommender


def sample_movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movie_id": [10, 20, 30],
            "title": ["Same Name", "Same Name", "Another Film"],
            "year": [1980, 2005, 2010],
            "genres": ["Drama", "Comedy", "Drama"],
            "overview": ["First", "Second", "Third"],
            "rating": [6.0, 7.0, 8.0],
            "popularity": [3.0, 1.0, 2.0],
            "tags": ["space planet adventure", "crime city", "space planet crime"],
        }
    )


class MovieRecommenderTests(unittest.TestCase):
    def test_recommendations_exclude_selection_and_descend_by_score(self) -> None:
        recommender = MovieRecommender(sample_movies())

        results = recommender.recommend(10, limit=2)

        self.assertEqual([result.movie_id for result in results], [30, 20])
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertGreater(results[0].score, results[1].score)
        self.assertTrue(all(np.isfinite(item.score) for item in results))

    def test_duplicate_titles_are_disambiguated_by_year(self) -> None:
        recommender = MovieRecommender(sample_movies())

        self.assertEqual(recommender.display_title(10), "Same Name (1980)")
        self.assertEqual(recommender.display_title(20), "Same Name (2005)")
        self.assertEqual(recommender.display_title(30), "Another Film")

    def test_rejects_duplicate_movie_ids(self) -> None:
        movies = sample_movies()
        movies.loc[1, "movie_id"] = 10

        with self.assertRaisesRegex(ValueError, "unique"):
            MovieRecommender(movies)

    def test_rejects_missing_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing columns"):
            MovieRecommender(sample_movies().drop(columns="tags"))

    def test_popularity_breaks_ties(self) -> None:
        movies = sample_movies()
        movies["tags"] = "space planet"
        results = MovieRecommender(movies).recommend(10)
        self.assertEqual([item.movie_id for item in results], [30, 20])

    def test_invalid_id_and_limit(self) -> None:
        recommender = MovieRecommender(sample_movies())
        with self.assertRaisesRegex(KeyError, "Unknown movie ID"):
            recommender.recommend(-1)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            recommender.recommend(10, limit=0)

    def test_empty_catalogue_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            MovieRecommender(sample_movies().iloc[:0])


if __name__ == "__main__":
    unittest.main()
