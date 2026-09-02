"""Regression checks for the movie artifact and Streamlit form."""

from pathlib import Path
import os
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from recommender import MovieRecommender
from posters import PosterLookupError


PROJECT_DIR = Path(__file__).resolve().parents[1]


class MovieSmokeTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"TMDB_API_KEY": ""})
        environment.start()
        self.addCleanup(environment.stop)

    def make_app(self, api_key=""):
        app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=30)
        app.secrets["TMDB_API_KEY"] = api_key
        return app

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
        app = self.make_app().run()
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        rendered = "\n".join(item.value for item in app.markdown)
        self.assertIn("The Dark Knight Rises", rendered)
        self.assertEqual(rendered.count('<article class="movie-card">'), 5)
        self.assertTrue(any("TMDB API key" in item.value for item in app.info))

    def test_selection_can_change(self):
        app = self.make_app().run()
        app.selectbox[0].select(559).run()  # Spider-Man 3
        app.button[0].click().run()
        self.assertFalse(app.exception)
        rendered = "\n".join(item.value for item in app.markdown)
        self.assertIn("Because you picked Spider-Man 3", rendered)
        self.assertIn("Spider-Man 2", rendered)
        app.run()
        self.assertIn("Because you picked Spider-Man 3", "\n".join(item.value for item in app.markdown))

    def test_posters_and_retry_after_a_temporary_failure(self):
        app = self.make_app(api_key="test-retry-key").run()
        with patch("posters.request_poster", side_effect=PosterLookupError("Temporary poster failure")):
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.warning), 1)
        with patch("posters.request_poster", return_value="https://image.tmdb.org/t/p/w500/test.jpg") as lookup:
            app.button(key="retry_posters").click().run()
            self.assertEqual(lookup.call_count, 5)
        self.assertFalse(app.exception)
        self.assertFalse(app.warning)
        rendered = "\n".join(item.value for item in app.markdown)
        self.assertEqual(rendered.count('<img class="poster"'), 5)

    def test_missing_artifact_has_actionable_message(self):
        import streamlit as st
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        with patch.object(MovieRecommender, "from_file", side_effect=FileNotFoundError):
            app = self.make_app().run()
        self.assertFalse(app.exception)
        self.assertIn("python build_model.py", app.error[0].value)

    def test_stale_selection_does_not_crash_after_catalogue_change(self):
        app = self.make_app()
        app.session_state["selected_movie_id"] = -1
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("catalogue changed" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()
