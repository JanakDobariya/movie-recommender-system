"""TMDB poster lookup. Failures never prevent movie recommendations."""

from __future__ import annotations

import re

import requests


POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{movie_id}"


class PosterLookupError(Exception):
    """A safe error message that does not include credentials or request URLs."""


def request_poster(movie_id: int, api_key: str) -> str | None:
    if not api_key:
        return None

    try:
        response = requests.get(
            TMDB_MOVIE_URL.format(movie_id=int(movie_id)),
            params={"api_key": api_key, "language": "en-US"},
            timeout=(3, 5),
        )
        if response.status_code in (401, 403):
            raise PosterLookupError(
                "TMDB rejected the poster API key. Check TMDB_API_KEY in the app's secrets."
            )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        raise PosterLookupError(
            "TMDB could not load some posters. You can retry without changing your recommendations."
        ) from None

    if not isinstance(data, dict):
        raise PosterLookupError("TMDB returned an unexpected response. Please retry the posters.")
    path = data.get("poster_path")
    if not path:
        return None
    if not isinstance(path, str) or not re.fullmatch(r"/[A-Za-z0-9._-]+", path):
        raise PosterLookupError("TMDB returned an invalid poster path. Please retry the posters.")
    return f"{POSTER_BASE_URL}{path}"
