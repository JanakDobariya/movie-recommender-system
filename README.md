# FrameFinder

FrameFinder is a content-based movie recommender built with the TMDB 5000 dataset. Pick a film and the app returns five titles with similar stories, genres, keywords, cast members, and directors.

## Live demo

[Open FrameFinder on Streamlit](https://movie-recommendation-system31.streamlit.app/)

The model does not use ratings to decide what to recommend. It creates a sparse TF-IDF index from each film's metadata and calculates cosine similarity for the selected film when a recommendation is requested. Ratings and popularity are shown as context only. Popularity breaks an exact similarity tie.

## Run the app

Use Python 3.12 or newer. Create a virtual environment and install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Posters are optional. To display them, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add a TMDB API key. The recommender still works when no key is configured.

Start Streamlit from the project directory:

```bash
streamlit run app.py
```

## Smoke tests

```bash
python -m unittest discover -s tests
```

The tests check the saved movie data, recommendation results, and the Streamlit
form. They do not require a TMDB API key.

## Rebuild the model

The generated movie artifact is already included. Rebuild it after changing the preprocessing code or replacing either compressed CSV file:

```bash
python build_model.py
```

`build_model.py` joins the source files by TMDB movie ID. It removes the three records that have no overview, leaving 4,800 unique movies, and writes `movies_dict.pkl` to the project directory. The app builds a sparse TF-IDF index from that artifact at startup; a large pairwise similarity file is not stored in the repository.

Only load these pickle files when they were produced locally. Python pickle files can run code while loading, so they are not a safe format for artifacts from an unknown source.

## Project files

- `app.py`: Streamlit interface and poster lookup
- `recommender.py`: artifact validation and recommendation logic
- `build_model.py`: data cleaning, feature extraction, and artifact generation
- `movie_recommender_system.ipynb`: a readable walkthrough of the model pipeline
- `assets/`: the approved TMDB attribution mark used by the app
- `Data/`: gzip-compressed TMDB movie and credit source files

Movie metadata and poster images come from TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.
