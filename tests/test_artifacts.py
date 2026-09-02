import unittest
from pathlib import Path

import numpy as np

from recommender import MovieRecommender


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recommender = MovieRecommender.from_file(
            PROJECT_ROOT / "movies_dict.pkl",
        )

    def test_catalogue_contains_4800_unique_movies(self) -> None:
        movie_ids = self.recommender.movie_ids
        self.assertEqual(len(movie_ids), 4_800)
        self.assertEqual(len(set(movie_ids)), 4_800)

    def test_real_recommendations_are_ranked_and_unique(self) -> None:
        avatar_id = self.recommender.movie_ids[
            self.recommender.default_index("Avatar")
        ]
        recommendations = self.recommender.recommend(avatar_id, limit=5)

        self.assertEqual(len(recommendations), 5)
        self.assertNotIn(avatar_id, [item.movie_id for item in recommendations])
        self.assertEqual(len({item.movie_id for item in recommendations}), 5)
        self.assertEqual([item.rank for item in recommendations], [1, 2, 3, 4, 5])
        self.assertTrue(
            np.all(np.diff([item.score for item in recommendations]) <= 0)
        )


if __name__ == "__main__":
    unittest.main()
