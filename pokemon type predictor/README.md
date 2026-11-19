# PokéType

A two-page Flask app that predicts a Pokémon's primary type from its stats and
secondary type. At startup, the app trains a Random Forest using the included
`G2 (1).csv` dataset and the tuned parameters in `model_info.json`.

## Run locally

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open the local address printed by Flask (usually `http://127.0.0.1:5000`).
The home page is `/`, and the input form is `/predict`.
