"""Streamlit interface for the movie recommender."""

from __future__ import annotations

import html
import os
import pickle
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from posters import PosterLookupError, request_poster
from recommender import MovieRecommender, Recommendation


PROJECT_ROOT = Path(__file__).resolve().parent


st.set_page_config(
    page_title="Movie recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _tmdb_api_key() -> str:
    """Read the API key without keeping it in source control."""
    environment_key = os.getenv("TMDB_API_KEY", "").strip()
    if environment_key:
        return environment_key

    try:
        return str(st.secrets.get("TMDB_API_KEY", "")).strip()
    except (FileNotFoundError, KeyError, StreamlitSecretNotFoundError):
        return ""


@st.cache_resource(show_spinner="Loading the recommendation model...")
def load_recommender() -> MovieRecommender:
    return MovieRecommender.from_file(PROJECT_ROOT / "movies_dict.pkl")


@st.cache_data(ttl=86_400, max_entries=5000, show_spinner=False)
def fetch_poster(movie_id: int, api_key: str) -> str | None:
    # Streamlit caches returned values, but not raised exceptions. A temporary
    # TMDB failure can therefore be retried immediately.
    return request_poster(movie_id, api_key)


def poster_result(movie_id: int, api_key: str) -> tuple[str | None, str | None]:
    try:
        return fetch_poster(movie_id, api_key), None
    except PosterLookupError as exc:
        return None, str(exc)


def movie_card(recommendation: Recommendation, poster_url: str | None) -> str:
    title = html.escape(recommendation.title)
    year = str(recommendation.year) if recommendation.year else "Year unknown"
    genres = html.escape(recommendation.genres or "Genre unavailable")
    score = round(recommendation.score * 100)

    if poster_url:
        artwork = (
            f'<img class="poster" src="{html.escape(poster_url, quote=True)}" '
            f'alt="Poster for {title}">'
        )
    else:
        initial = html.escape(recommendation.title[:1].upper() or "?")
        artwork = (
            '<div class="poster poster-fallback" role="img" '
            f'aria-label="Poster unavailable for {title}">'
            f'<span>{initial}</span><small>Poster unavailable</small></div>'
        )

    return textwrap.dedent(f"""
        <article class="movie-card">
            <div class="poster-wrap">
                {artwork}
                <span class="rank">{recommendation.rank:02d}</span>
            </div>
            <div class="card-copy">
                <h3>{title}</h3>
                <p class="movie-meta">{year} <span>·</span> {genres}</p>
                <p class="similarity">{score}% content similarity</p>
            </div>
        </article>
    """).strip()


def render_cards(recommendations: list[Recommendation], api_key: str) -> None:
    if api_key:
        with st.spinner("Loading posters..."):
            with ThreadPoolExecutor(max_workers=5) as executor:
                posters = list(executor.map(
                    lambda item: poster_result(item.movie_id, api_key), recommendations
                ))
    else:
        posters = [(None, None)] * len(recommendations)

    cards = [
        movie_card(item, poster_url)
        for item, (poster_url, _) in zip(recommendations, posters)
    ]
    st.markdown(f'<div class="movie-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    if not api_key:
        st.info(
            "Posters need a TMDB API key. On Streamlit Cloud, add TMDB_API_KEY "
            "under Manage app → Settings → Secrets. Recommendations work without it."
        )
    errors = list(dict.fromkeys(error for _, error in posters if error))
    if errors:
        st.warning(" ".join(errors))
        if st.button("Retry posters", key="retry_posters"):
            st.rerun()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #f7f2e8;
            --muted: #a9a9b2;
            --panel: rgba(20, 23, 31, 0.88);
            --line: rgba(255, 255, 255, 0.10);
            --accent: #f4bd4a;
            --accent-strong: #ffcf66;
        }

        .stApp {
            background:
                radial-gradient(circle at 78% 4%, rgba(131, 53, 44, 0.26), transparent 32rem),
                radial-gradient(circle at 8% 22%, rgba(43, 65, 92, 0.25), transparent 28rem),
                #080a0f;
            color: var(--ink);
        }

        [data-testid="stHeader"] { background: rgba(8, 10, 15, 0.96); }
        [data-testid="stMainBlockContainer"] {
            max-width: 1240px;
            padding-top: 3.5rem;
            padding-bottom: 5rem;
        }

        .hero { max-width: 780px; margin: 1.25rem 0 2.4rem; }
        .eyebrow {
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.18em;
            margin-bottom: 0.85rem;
            text-transform: uppercase;
        }
        .hero h1 {
            color: var(--ink);
            font-size: clamp(3rem, 7vw, 6.2rem);
            font-weight: 760;
            letter-spacing: -0.065em;
            line-height: 0.92;
            margin: 0;
        }
        .hero h1 em { color: var(--accent); font-style: normal; }
        .hero-copy {
            color: #b9b8bf;
            font-size: clamp(1rem, 2vw, 1.2rem);
            line-height: 1.6;
            margin-top: 1.25rem;
            max-width: 620px;
        }

        [data-testid="stForm"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 20px;
            box-shadow: 0 24px 70px rgba(0,0,0,0.22);
            padding: 1.1rem 1.2rem 0.45rem;
        }
        .stSelectbox label p { color: #dedbe0; font-weight: 650; }
        [data-baseweb="select"] > div {
            background: #0e1118;
            border-color: rgba(255,255,255,0.14);
            border-radius: 12px;
            color: var(--ink);
            min-height: 3.2rem;
        }
        .stButton > button, [data-testid="stFormSubmitButton"] > button {
            background: var(--accent);
            border: 0;
            border-radius: 12px;
            color: #17130b;
            font-weight: 760;
            min-height: 3.2rem;
            transition: transform 120ms ease, background 120ms ease;
            width: 100%;
        }
        [data-testid="stFormSubmitButton"] { width: 100%; }
        .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
            background: var(--accent-strong);
            color: #17130b;
            transform: translateY(-1px);
        }
        .stButton > button:focus, [data-testid="stFormSubmitButton"] > button:focus {
            box-shadow: 0 0 0 3px rgba(244,189,74,0.28);
        }

        .section-heading { margin: 3rem 0 1.2rem; }
        .section-heading h2 {
            color: var(--ink);
            font-size: clamp(1.65rem, 3vw, 2.35rem);
            letter-spacing: -0.035em;
            margin: 0.25rem 0 0;
        }
        .section-heading p { color: var(--muted); margin: 0.45rem 0 0; }

        .selection-card {
            align-items: start;
            background: linear-gradient(135deg, rgba(244,189,74,0.12), rgba(255,255,255,0.025));
            border: 1px solid rgba(244,189,74,0.20);
            border-radius: 16px;
            display: grid;
            gap: 1.4rem;
            grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
            margin-top: 1rem;
            padding: 1.25rem 1.35rem;
        }
        .selection-card h2 { color: var(--ink); font-size: 1.25rem; margin: 0; }
        .selection-card p { color: #b8b6bc; line-height: 1.55; margin: 0.55rem 0 0; }
        .selection-facts { color: var(--accent); font-size: 0.86rem; overflow-wrap: anywhere; }

        .movie-grid {
            display: grid;
            gap: 1.05rem;
            grid-template-columns: repeat(5, minmax(0, 1fr));
        }
        .movie-card {
            background: rgba(18, 21, 28, 0.94);
            border: 1px solid var(--line);
            border-radius: 16px;
            min-width: 0;
            overflow: hidden;
            transition: border-color 160ms ease, transform 160ms ease;
        }
        .movie-card:hover { border-color: rgba(244,189,74,0.42); transform: translateY(-4px); }
        .poster-wrap { aspect-ratio: 2 / 3; background: #141820; overflow: hidden; position: relative; }
        .poster { display: block; height: 100%; object-fit: cover; width: 100%; }
        .poster-fallback {
            align-items: center;
            background: linear-gradient(145deg, #263143, #181c25 60%, #3d2924);
            color: rgba(255,255,255,0.72);
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .poster-fallback span { color: rgba(244,189,74,0.7); font-size: 4.5rem; font-weight: 800; }
        .poster-fallback small { color: #acaab0; margin-top: 0.5rem; }
        .rank {
            align-items: center;
            background: var(--accent);
            border-radius: 999px;
            color: #15110a;
            display: flex;
            font-size: 0.72rem;
            font-weight: 800;
            height: 2rem;
            justify-content: center;
            left: 0.75rem;
            position: absolute;
            top: 0.75rem;
            width: 2rem;
        }
        .card-copy { padding: 1rem; }
        .card-copy h3 {
            color: var(--ink);
            font-size: 1rem;
            line-height: 1.35;
            margin: 0;
            min-height: 2.7em;
        }
        .movie-meta {
            color: #aaa8b0;
            font-size: 0.78rem;
            line-height: 1.4;
            margin: 0.65rem 0 0;
            min-height: 2.2em;
        }
        .movie-meta span { color: #5e5f67; padding: 0 0.15rem; }
        .similarity { color: var(--accent); font-size: 0.74rem; font-weight: 700; margin: 0.8rem 0 0; }

        [data-testid="stExpander"] {
            background: rgba(18,21,28,0.6);
            border-color: var(--line);
            margin-top: 3.5rem;
        }
        [data-testid="stExpander"] p { color: #aaa8b0; font-size: 0.88rem; }

        @media (max-width: 980px) {
            .movie-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        }
        @media (max-width: 640px) {
            [data-testid="stMainBlockContainer"] { padding-top: 3.5rem; }
            .hero h1 { font-size: 3.4rem; }
            .movie-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .selection-card { grid-template-columns: 1fr; }
            .selection-card p {
                display: -webkit-box;
                -webkit-box-orient: vertical;
                -webkit-line-clamp: 6;
                overflow: hidden;
            }
            .selection-facts { white-space: normal; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    try:
        recommender = load_recommender()
    except (OSError, ValueError, ImportError, pickle.UnpicklingError, EOFError):
        st.error(
            "The movie catalogue could not be loaded. Run `python build_model.py` "
            "with the project dependencies installed, then commit movies_dict.pkl and reboot the app."
        )
        st.stop()
    api_key = _tmdb_api_key()

    st.markdown(
        f"""
        <header class="hero">
            <p class="eyebrow">Content-based movie discovery</p>
            <h1>Find your next <em>watch.</em></h1>
            <p class="hero-copy">
                Choose a film you already like. FrameFinder compares its story,
                genres, cast, keywords, and director across {len(recommender.movie_ids):,} titles.
            </p>
        </header>
        """,
        unsafe_allow_html=True,
    )

    with st.form("recommendation_form"):
        left, right = st.columns([4, 1], vertical_alignment="bottom")
        with left:
            selected_id = st.selectbox(
                "Pick a movie",
                options=recommender.movie_ids,
                format_func=recommender.display_title,
                index=recommender.default_index("The Dark Knight"),
                help="Start typing to search the catalogue.",
            )
        with right:
            submitted = st.form_submit_button("Find similar films")

    if submitted:
        st.session_state["selected_movie_id"] = selected_id

    active_id = st.session_state.get("selected_movie_id")
    if active_id is not None and active_id not in recommender.movie_ids:
        st.session_state.pop("selected_movie_id", None)
        st.info("The catalogue changed. Pick a movie to see fresh recommendations.")
        active_id = None
    if active_id is not None:
        selected = recommender.movie(active_id)
        overview = html.escape(selected.overview or "No overview is available for this title.")
        facts = " · ".join(
            part
            for part in [
                str(selected.year) if selected.year else "",
                selected.genres,
                f"TMDB {selected.rating:.1f}/10" if selected.rating else "",
            ]
            if part
        )
        st.markdown(
            f"""
            <section class="selection-card">
                <div>
                    <h2>Because you picked {html.escape(selected.title)}</h2>
                    <p>{overview}</p>
                </div>
                <div class="selection-facts">{html.escape(facts)}</div>
            </section>
            """,
            unsafe_allow_html=True,
        )

        recommendations = recommender.recommend(active_id, limit=5)
        st.markdown(
            """
            <div class="section-heading">
                <p class="eyebrow">Your five closest matches</p>
                <h2>Worth adding to the queue</h2>
                <p>Ordered by similarity to the film you selected.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_cards(recommendations, api_key)

    with st.expander("About FrameFinder"):
        st.write(
            "The model turns each film's overview, genres, keywords, top-billed "
            "cast, and director into a TF-IDF vector. Cosine similarity measures "
            "how closely those vectors point in the same direction. Ratings and "
            "popularity do not influence the recommendation score."
        )
        st.image(str(PROJECT_ROOT / "assets" / "tmdb.svg"), width=110)
        st.caption(
            "Movie metadata and poster images are supplied by TMDB. "
            "This product uses the TMDB API but is not endorsed or certified by TMDB."
        )
        if not api_key:
            st.caption("Add a TMDB API key to display posters. The recommendation model works without one.")


if __name__ == "__main__":
    main()
