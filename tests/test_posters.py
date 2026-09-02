import unittest
from unittest.mock import Mock, patch

import requests

from posters import PosterLookupError, request_poster


class PosterTests(unittest.TestCase):
    def response(self, status=200, data=None):
        result = Mock(status_code=status)
        result.json.return_value = data
        if status >= 400:
            result.raise_for_status.side_effect = requests.HTTPError("redacted")
        return result

    @patch("posters.requests.get")
    def test_missing_key_does_not_call_api(self, get):
        self.assertIsNone(request_poster(155, ""))
        get.assert_not_called()

    @patch("posters.requests.get")
    def test_poster_uses_tmdb_id(self, get):
        get.return_value = self.response(data={"poster_path": "/poster.jpg"})
        self.assertEqual(request_poster(155, "test-key"), "https://image.tmdb.org/t/p/w500/poster.jpg")
        self.assertEqual(get.call_args.args[0], "https://api.themoviedb.org/3/movie/155")
        self.assertEqual(get.call_args.kwargs["timeout"], (3, 5))

    @patch("posters.requests.get")
    def test_movie_without_a_poster(self, get):
        for response in (self.response(data={"poster_path": None}), self.response(status=404)):
            get.return_value = response
            self.assertIsNone(request_poster(155, "test-key"))

    @patch("posters.requests.get")
    def test_invalid_key_has_a_safe_message(self, get):
        for status in (401, 403):
            get.return_value = self.response(status=status)
            with self.assertRaisesRegex(PosterLookupError, "TMDB rejected") as error:
                request_poster(155, "private-test-key")
            self.assertNotIn("private-test-key", str(error.exception))

    @patch("posters.requests.get")
    def test_timeout_and_http_failures_do_not_leak_credentials(self, get):
        for failure in (requests.Timeout("private-test-key"), requests.ConnectionError("private-test-key")):
            get.side_effect = failure
            with self.assertRaises(PosterLookupError) as error:
                request_poster(155, "private-test-key")
            self.assertNotIn("private-test-key", str(error.exception))
        get.side_effect = None
        for status in (429, 500, 503):
            get.return_value = self.response(status=status)
            with self.assertRaises(PosterLookupError):
                request_poster(155, "test-key")

    @patch("posters.requests.get")
    def test_invalid_responses_can_be_retried(self, get):
        for data in ([], {"poster_path": "https://example.com/poster.jpg"}, {"poster_path": 123}):
            get.return_value = self.response(data=data)
            with self.assertRaises(PosterLookupError):
                request_poster(155, "test-key")
        get.return_value = self.response()
        get.return_value.json.side_effect = ValueError("invalid JSON")
        with self.assertRaises(PosterLookupError):
            request_poster(155, "test-key")


if __name__ == "__main__":
    unittest.main()
