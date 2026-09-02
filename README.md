# FrameFinder

FrameFinder is a content-based movie recommender built with the TMDB 5000 dataset. Pick a film and the app returns five titles with similar stories, genres, keywords, cast members, and directors.

## Live demo

[Open FrameFinder on Streamlit](https://movie-recommendation-system31.streamlit.app/)

The model does not use ratings to decide what to recommend. It creates a sparse TF-IDF index from each film's metadata and calculates cosine similarity for the selected film when a recommendation is requested. Ratings and popularity are shown as context only. Popularity breaks an exact similarity tie.

## Run the app

Use Python 3.12 for the same environment used in the deployment checks. Create a virtual environment and install the pinned dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Posters need a TMDB API key. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your key, or set the `TMDB_API_KEY` environment variable. Keep the secrets file out of Git. Recommendations still work without a key, and the app shows a setup message instead of posters.

Start Streamlit from the project directory:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Use these settings when creating the app:

- Repository: `JanakDobariya/movie-recommender-system`
- Branch: `main`
- Main file path: `app.py`
- Python version in Advanced settings: `3.12`

Copy the contents of your local `.streamlit/secrets.toml` into the deployment's Secrets field. For an existing app, open **Manage app → Settings → Secrets**. The local secrets file is ignored by Git, so a code push does not upload the poster key.

The app loads the included `movies_dict.pkl` and builds its sparse index at startup. You do not need to run the notebook or upload `similarity.pkl`. Streamlit installs `requirements.txt` automatically.

If the page remains on "Your app is waking up" or "taking longer than normal," open the app's logs through **Manage app**, or use the app menu in your [Streamlit workspace](https://share.streamlit.io/). Read the error before changing files. After fixing a build error, reboot the app from that menu. A working local app does not confirm that the cloud deployment has started successfully.

For missing posters, check the key in Secrets. An invalid key or a temporary TMDB failure produces a warning with a **Retry posters** button. Failed requests are not cached, and poster failures do not stop recommendations.

See Streamlit's [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [secrets guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

## Smoke tests

```bash
python -m unittest discover -s tests
```

The tests check the saved movie data, ranking, movie selection, missing artifacts, and poster failures and retries. Network requests are mocked, so the tests do not need a TMDB API key. Run them before pushing changes to the deployed app.

## Rebuild the model

The generated movie artifact is already included. Rebuild it after changing the preprocessing code or replacing either compressed CSV file:

```bash
python build_model.py
```

`build_model.py` joins the source files by TMDB movie ID. It removes the three records that have no overview, leaving 4,800 unique movies, and writes `movies_dict.pkl` to the project directory. The app builds a sparse TF-IDF index from that artifact at startup; a large pairwise similarity file is not stored in the repository.

Only load these pickle files when they were produced locally. Python pickle files can run code while loading, so they are not a safe format for artifacts from an unknown source.

To run the notebook, install Jupyter in the same virtual environment with `python -m pip install jupyterlab`, then launch `jupyter lab` from the project directory and select that environment's Python kernel.

## Project files

- `app.py`: Streamlit interface, state, and poster caching
- `posters.py`: TMDB requests, timeouts, and safe error messages
- `recommender.py`: artifact validation and recommendation logic
- `build_model.py`: data cleaning, feature extraction, and artifact generation
- `movies_dict.pkl`: prepared movie metadata used by the app
- `movie_recommender_system.ipynb`: a readable walkthrough of the model pipeline
- `requirements.txt`: pinned application dependencies
- `.streamlit/config.toml`: server and dark-theme settings
- `.streamlit/secrets.toml.example`: a template for your private TMDB key
- `tests/`: regression checks for the app, catalogue, and poster requests
- `assets/`: the approved TMDB attribution mark used by the app
- `Data/`: gzip-compressed TMDB movie and credit source files

Local secrets, virtual environments, screenshots, audit output, and redundant model files are excluded from Git.

Movie metadata and poster images come from TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.
