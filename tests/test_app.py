"""Regression checks for the movie artifact and Streamlit form."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from recommender import MovieRecommender


PROJECT_DIR = Path(__file__).resolve().parents[1]


class MovieSmokeTests(unittest.TestCase):
    def test_recommendations(self):
        recommender = MovieRecommender.from_file(PROJECT_DIR / "movies_dict.pkl")
        self.assertEqual(len(recommender.movie_ids), 4800)
        selected_id = 155  # The Dark Knight in the included TMDB data.
        results = recommender.recommend(selected_id, limit=5)
        self.assertEqual(len(results), 5)
        self.assertNotIn(selected_id, [item.movie_id for item in results])
        self.assertEqual(len({item.movie_id for item in results}), 5)
        self.assertEqual([item.rank for item in results], [1, 2, 3, 4, 5])
        self.assertEqual(
            [item.score for item in results],
            sorted([item.score for item in results], reverse=True),
        )

    def test_app_form(self):
        app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        rendered = "\n".join(item.value for item in app.markdown)
        self.assertIn("The Dark Knight Rises", rendered)
        self.assertEqual(rendered.count('<article class="movie-card">'), 5)


if __name__ == "__main__":
    unittest.main()
