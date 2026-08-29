# Reverse Dictionary — Gradle Frontend

Two services, both on localhost:

```
[browser] --> localhost:8080 (Spring Boot / Gradle, static HTML+JS)
                    |  proxies /api/*
                    v
              localhost:8000 (Python FastAPI, wraps sif_query.py /
                               sentence_transformer_encoder.py / bert_cls.py)
```

The browser only ever talks to the Gradle-built app on :8080 — it proxies
`/api/query` and `/api/encoders` to the Python service, so there's no CORS
setup needed and the Python address never appears in client-side code.

## 1. Start the Python backend (the actual model)

```bash
cd backend
pip install -r requirements.txt

# Point at wherever the OPTED processed CSVs live (opted_train/valid/test.csv).
# Defaults to /mnt/project if unset.
export REVDICT_DATA_DIR=/path/to/your/processed/data

uvicorn app:app --host 0.0.0.0 --port 8000
```

The **SIF** encoder works out of the box with just `requirements.txt`
(scikit-learn only) and is what the app builds at startup. **Sentence-
Transformer** and **BERT [CLS]** are optional — install
`sentence-transformers` or `torch` + `transformers` to turn them on; until
then the dropdown will mark them "(unavailable)" and the API returns a 503
with an install hint if you try to use them anyway.

Check it's up:

```bash
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/query \
    -H "Content-Type: application/json" \
    -d '{"text": "a large gray animal with a trunk", "encoder": "sif", "top_k": 10}'
```

## 2. Start the Gradle frontend

```bash
cd frontend-gradle
gradle bootRun
# or, if you don't have Gradle installed locally, generate the wrapper once:
#   gradle wrapper --gradle-version 8.8
#   ./gradlew bootRun
```

If the Python service is running somewhere other than
`http://localhost:8000`, override it:

```bash
gradle bootRun -PrevdictApiBase=http://localhost:9000
```

Then open **http://localhost:8080** — you'll see a text field for the
definition, a dropdown to pick the encoder (SIF / Sentence-Transformer /
BERT [CLS]), and a results-count dropdown (Top 10 / Top 100 / Custom).

## Project layout

```
reverse-dict-app/
  backend/
    app.py               FastAPI service — builds the SIF index at startup,
                          lazily loads neural encoders on first request
    requirements.txt
  frontend-gradle/
    build.gradle          Spring Boot (Gradle) build
    settings.gradle
    src/main/java/com/revdict/frontend/
      FrontendApplication.java
      WebClientConfig.java   points at the Python API base URL
      QueryController.java   proxies /api/query, /api/encoders, /api/health
      QueryDtos.java
    src/main/resources/
      application.properties
      static/
        index.html          text field + encoder dropdown + top-k dropdown
        app.js
        style.css
```

## Notes / next steps

- The BiLSTM encoder (`bilstm_encoder.py`) isn't wired into the API yet —
  it needs a trained checkpoint + `bilstm_vocab.py` vocabulary to load,
  which weren't part of this data drop. Add a `bilstm` branch to
  `_load_neural_encoder` in `app.py` once you have a saved model file.
- `SIF_SUBSET_FRAC` (env var `REVDICT_SIF_SUBSET_FRAC`, default `0.25`)
  controls how much of the data the SIF index is built from — raise it for
  better recall, lower it for a faster startup.
- This is a local dev setup (CORS wide open, no auth) — don't expose
  `:8000` or `:8080` beyond localhost as-is.
