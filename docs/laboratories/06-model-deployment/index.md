---
authors: domonkosadam
---

# 06 - Deploying a model using FastAPI and Docker

## Goal

The main goal of this laboratory is to turn a local LLM into a production-style service: a REST API built with FastAPI, containerized with Docker, backed by a SQLite conversation history, used from a Streamlit chat UI, and made faster with semantic caching.

The structure of this laboratory is as follows:

1. Installing the required dependencies and creating the project skeleton.
1. FastAPI Backend [5 points in total]
    1. Define Pydantic models [0.5 point]
    1. Implement `/health` and `/models` endpoints [0.5 point]
    1. Implement `/generate` POST endpoint [2 points]
    1. Implement `/generate/stream` POST endpoint [2 points]
1. Docker Containerization [3 points in total]
    1. Write the Dockerfile [1 point]
    1. Write docker-compose.yml [1 point]
    1. Read the Ollama URL from an environment variable [1 point]
1. Postman: Raw API Inspection [1 point]
1. Conversation Endpoint + SQLite History [4 points in total]
    1. Implement the database schema [1 point]
    1. Implement session and message operations [1 point]
    1. Implement `POST /conversation` [1 point]
    1. Implement `GET /conversation/{session_id}` [1 point]
1. Streamlit Chat UI [3 points]
1. Semantic Caching [4 points in total]
    1. Implement embedding generation [1 point]
    1. Implement cosine similarity [0.5 point]
    1. Implement model-aware cache check and store [1 point]
    1. Add semantic caching to `POST /generate` [1 point]
    1. Threshold experiment [0.5 point]
1. Giving feedback [+1 point]

!!! info "Grading"
    In order to pass this laboratory, you must obtain at least 8 points out of 20.

!!! important "Screenshot requirement"
    At the end of **every** exercise (marked with a :camera: **Screenshot** note), you must take a screenshot of your working solution, including its output, and upload it together with your solution, using the exact filename given in that note (e.g. `f1_1.png`, `f4_2.png`). Solutions missing the required screenshots, or using the wrong filename, will not be accepted.

    If an output does not fit on one screen, split it into several screenshots with a numeric suffix (e.g. `f3_1a_1.png`, `f3_1a_2.png`).

## Preparation

Don't forget to follow the assignment submission process described under [GitHub](../../information/github.md) while working on this laboratory.

!!! important "Reviewer"
    When creating the pull request for this laboratory, assign it to the `domonkosadam` GitHub user.

!!! tip "Project files"
    This laboratory is a small Python project: you work in plain `.py` files (plus a `Dockerfile`, a `.dockerignore` and a `docker-compose.yml`), not in a notebook. Create the files from the skeletons in the [Project skeleton](#project-skeleton) section in the root of your repository. Regardless of how you test your code (terminal, Postman, a notebook for experiments), make sure that **everything** (code, `answers.md` and the required screenshots) is committed and pushed to your solution branch. Do **not** commit your virtual environment, `data/` or any `*.db` file.

!!! important "Written answers"
    Some exercises contain questions (marked with :pencil: **Written answer**). Answer them in a file called `answers.md` in the root of your repository, under a heading with the exercise number (e.g. `## Exercise 1.2`). Your feedback at the end of the laboratory also goes into this file (under `## Feedback`). Make sure `answers.md` is committed and pushed together with your solution and screenshots.

## Setup

### Prerequisites

Make sure the following are available on your machine before starting:

- **Ollama** running locally (`ollama serve`, or the desktop app)
- The generation model: `ollama pull llama3.2`
- The embedding model: `ollama pull nomic-embed-text`
- **Docker Desktop** (or Docker Engine with the Compose plugin) installed and running
- **Postman** installed
- **Python 3.11 – 3.14** (on Windows on ARM, install the x64 build of Python)

Verify that Ollama is working:

```bash
curl http://localhost:11434/api/tags
```

!!! important "Mandatory"
    Create a virtual environment and install the pinned dependencies. Both requirements files pin the exact version of **every** package, including all indirect dependencies, and they are locked together, so they can be installed into the same environment. If `pip` reports success but installed almost nothing (check with `pip list`), your Python version / platform combination is not supported (see above).

    ```bash
    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r https://raw.githubusercontent.com/VIAUMSMA001/laboratories/main/docs/laboratories/06-model-deployment/requirements_backend.txt
    pip install -r https://raw.githubusercontent.com/VIAUMSMA001/laboratories/main/docs/laboratories/06-model-deployment/requirements_frontend.txt
    ```

    Also download both files into your project folder: the Docker image of Exercise 2.1 installs `requirements_backend.txt`.

    ```bash
    curl -O https://raw.githubusercontent.com/VIAUMSMA001/laboratories/main/docs/laboratories/06-model-deployment/requirements_backend.txt
    curl -O https://raw.githubusercontent.com/VIAUMSMA001/laboratories/main/docs/laboratories/06-model-deployment/requirements_frontend.txt
    ```

!!! tip "Windows users"
    The `curl` examples use bash quoting. Run them in **Git Bash** or **WSL**; in PowerShell use `curl.exe` and escape the inner double quotes, or use `Invoke-RestMethod`, e.g.

    ```powershell
    Invoke-RestMethod -Method Post -Uri http://localhost:8000/generate -ContentType "application/json" -Body '{"prompt": "Hi"}'
    ```

The main packages in the requirements files are:

```
aiosqlite==0.22.1
fastapi==0.141.1
httpx==0.28.1
pydantic==2.13.5
uvicorn==0.54.0
requests==2.34.2
streamlit==1.64.0
```

Useful references:

- FastAPI: <https://fastapi.tiangolo.com/>
- HTTPX (async client, streaming): <https://www.python-httpx.org/async/>
- Ollama REST API: <https://github.com/ollama/ollama/blob/main/docs/api.md>
- aiosqlite: <https://aiosqlite.omnilib.dev/>
- Dockerfile reference: <https://docs.docker.com/reference/dockerfile/>
- Compose file reference: <https://docs.docker.com/reference/compose-file/>

### Project skeleton

Create the following files. They contain only the provided infrastructure, function signatures and docstrings: **everything marked `TODO` is yours to write**, following the exercises below.

```
.
├── app.py                     # Streamlit chat UI (Part 5)
├── cache.py                   # semantic cache (Part 6)
├── config.py                  # settings (Exercise 2.3)
├── database.py                # SQLite history (Part 4)
├── main.py                    # FastAPI backend (Parts 1, 4, 6)
├── Dockerfile                 # Exercise 2.1
├── .dockerignore              # Exercise 2.1
├── docker-compose.yml         # Exercise 2.2
├── requirements_backend.txt   # downloaded above
├── requirements_frontend.txt  # downloaded above
└── answers.md                 # written answers and feedback
```

```python title="config.py"
"""
config.py — Settings shared by main.py and cache.py
"""

# TODO 2.3 — make the Ollama URL configurable through the OLLAMA_BASE_URL environment variable
OLLAMA_BASE_URL = "http://localhost:11434"
```

```python title="main.py"
"""
main.py — FastAPI backend for local LLM deployment

Endpoints (the full specification is in the laboratory description):
    GET  /health                      — service + Ollama status               (Exercise 1.2)
    GET  /models                      — list available Ollama models          (Exercise 1.2)
    POST /generate                    — single-turn generation                (Exercise 1.3, cache in 6.4)
    POST /generate/stream             — single-turn streaming generation      (Exercise 1.4)
    POST /conversation                — multi-turn chat with SQLite history   (Exercise 4.3)
    GET  /conversation/{session_id}   — retrieve conversation history         (Exercise 4.4)
    GET  /cache/stats                 — number of cached entries              (provided)
"""

import json
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from cache import check_cache, store_in_cache, get_embedding, cache_size
from config import OLLAMA_BASE_URL
from database import init_db, create_session, save_message, get_history, session_exists


# TODO 1.1 — request/response models


# ---------------------------------------------------------------------------
# Startup — initialize the database via lifespan (provided)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="LLM API Lab", version="2.0.0", lifespan=lifespan)


# TODO 1.2 — GET /health and GET /models


# TODO 1.3 — POST /generate


# TODO 1.4 — POST /generate/stream


# TODO 4.3 — POST /conversation


# TODO 4.4 — GET /conversation/{session_id}


# ---------------------------------------------------------------------------
# Utility endpoint (provided)
# ---------------------------------------------------------------------------

@app.get("/cache/stats")
async def cache_stats():
    return {"cached_entries": cache_size()}
```

```python title="database.py"
"""
database.py — SQLite conversation history using aiosqlite

Schema:
    sessions:  id (TEXT PK), created_at (TEXT)
    messages:  id (INTEGER PK AUTOINCREMENT), session_id (TEXT FK → sessions.id),
               role (TEXT), content (TEXT), timestamp (TEXT)

All timestamps are UTC ISO-8601 strings. The full specification is in the laboratory description (Part 4).
"""

import os
from datetime import datetime, timezone

import aiosqlite

DB_PATH = os.getenv("DB_PATH", "./conversations.db")


async def init_db() -> None:
    """Create both tables if they do not exist yet (safe to call on every startup)."""
    # TODO 4.1 — this runs at server startup; until you implement it, it does nothing.
    return None


async def create_session(session_id: str) -> None:
    """Insert a new session. Creating an already existing session must not raise an error."""
    # TODO 4.2a
    raise NotImplementedError


async def session_exists(session_id: str) -> bool:
    """Return True if a session with the given id exists."""
    # TODO 4.2b
    raise NotImplementedError


async def save_message(session_id: str, role: str, content: str) -> None:
    """Append a message ("user" or "assistant") to a session's history."""
    # TODO 4.2c
    raise NotImplementedError


async def get_history(session_id: str) -> list[dict]:
    """Return the session's messages in the order they were saved: [{"role": ..., "content": ...}, ...]."""
    # TODO 4.2d
    raise NotImplementedError
```

```python title="cache.py"
"""
cache.py — Semantic caching using Ollama embeddings and cosine similarity

How it works:
    1. Incoming prompt → embed with nomic-embed-text → 768-dim vector
    2. Compare against the cached prompt embeddings **in the same namespace** using cosine similarity
    3. If the best similarity >= threshold → return the cached response (cache HIT)
    4. Otherwise → call the LLM and store the result in the cache (cache MISS)

The cache is stored in memory as a module-level list.
In production it would be replaced by a vector database (e.g. Redis, Qdrant, pgvector).
The full specification is in the laboratory description (Part 6).
"""

import math

import httpx

from config import OLLAMA_BASE_URL

EMBED_MODEL = "nomic-embed-text"
DEFAULT_THRESHOLD = 0.92

# In-memory cache: list of {"embedding": [...], "prompt": str, "namespace": str, "response": str}
# The namespace groups interchangeable answers (e.g. same model and same length limit; see Exercise 6.4).
_cache: list[dict] = []


async def get_embedding(text: str) -> list[float]:
    """Return the embedding vector of `text`, computed by EMBED_MODEL through Ollama."""
    # TODO 6.1
    raise NotImplementedError


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of a and b (standard library only, no numpy)."""
    # TODO 6.2
    raise NotImplementedError


def check_cache(
    prompt_embedding: list[float],
    namespace: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[bool, str, float]:
    """
    Find the most similar cached prompt in the same `namespace`.

    Returns (hit, cached_response, best_similarity):
        hit             — True if best_similarity >= threshold
        cached_response — the response of the best match on a hit, otherwise ""
        best_similarity — the highest similarity found (0.0 if there is no candidate)
    """
    # TODO 6.3a
    raise NotImplementedError


def store_in_cache(prompt: str, namespace: str, embedding: list[float], response: str) -> None:
    """Add a new entry to the cache."""
    # TODO 6.3b
    raise NotImplementedError


def cache_size() -> int:
    return len(_cache)


def clear_cache() -> None:
    _cache.clear()
```

```python title="app.py"
"""
app.py — Streamlit frontend for the LLM API Lab

Chat — ChatGPT-like interface using the /conversation endpoint.
The full specification is in the laboratory description (Exercise 5.1).
"""

import uuid

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="LLM API Lab", page_icon="🤖", layout="wide")
st.title("🤖 LLM API Lab")


def get_models() -> list[str]:
    """Return the model names offered by the backend (provided helper)."""
    try:
        r = requests.get(f"{API_BASE}/models", timeout=5)
        return r.json().get("models", ["llama3.2"])
    except Exception:
        return ["llama3.2"]


st.header("Conversational Chat")
st.caption("Messages are persisted per session using the /conversation endpoint.")

# TODO 5.1a — session state

# TODO 5.1b — session controls, model selector, chat history

# TODO 5.1c — handle a new message
```

```dockerfile title="Dockerfile"
# Dockerfile
#
# Exercise 2.1 — Build a container image for the FastAPI backend.
# Write the Dockerfile from scratch; the requirements are listed in the laboratory description (Exercise 2.1).
```

```yaml title="docker-compose.yml"
# docker-compose.yml
#
# Exercise 2.2 — Define the `api` service.
# The requirements are listed in the laboratory description (Exercise 2.2).
```

## Part 1 – FastAPI Backend

FastAPI is a modern Python web framework for building APIs with automatic validation. It uses **Pydantic** models to define the shape of incoming and outgoing data, and **uvicorn** as the ASGI server that handles the HTTP connections.

In an ML deployment, FastAPI sits between your model and any client. This gives you:

- a stable, versioned interface that clients depend on (not Python internals),
- input validation before anything reaches the model,
- a place to add logging, caching, auth and monitoring later,
- language-agnostic access: any client that speaks HTTP can use it.

All calls to Ollama must be **asynchronous** (`httpx.AsyncClient`), so that a slow LLM call does not block the server. The Ollama base URL comes from `config.py` (`OLLAMA_BASE_URL`); do not hard-code it anywhere else.

**Error handling (applies to every endpoint that calls Ollama):**

- Ollama cannot be reached (connection error, timeout) → **HTTP 503**
- Ollama answers with an error status (e.g. unknown model) → **HTTP 502**, with Ollama's error message in `detail`
- never let such a failure surface as an unhandled 500 error

Start the app locally with:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The interactive API documentation generated from your models is available at http://localhost:8000/docs.

!!! tip "Tip"
    Work through Part 1 entirely with `uvicorn` running locally before moving to Docker in Part 2. If something breaks after containerizing, you then know that the code itself is correct.


### Exercise 1.1 – Define Pydantic Models

In `main.py`, define the following four models:

| Model | Field | Type | Default |
|---|---|---|---|
| `GenerateRequest` | `prompt` | str | – |
| | `model` | str | `"llama3.2"` |
| | `temperature` | float | `0.7` |
| | `max_tokens` | int | `512` |
| `GenerateResponse` | `response` | str | – |
| | `model` | str | – |
| | `latency_ms` | float | – |
| | `token_count` | int | – |
| | `cache_hit` | bool | `False` |
| | `similarity_score` | float | `0.0` |
| `ConversationRequest` | `session_id` | str | – |
| | `message` | str | – |
| | `model` | str | `"llama3.2"` |
| `ConversationResponse` | `response` | str | – |
| | `session_id` | str | – |
| | `latency_ms` | float | – |

**Expected outcome:** the models can be created and apply their defaults:
```bash
python -c "from main import GenerateRequest; print(GenerateRequest(prompt='hi'))"
# prompt='hi' model='llama3.2' temperature=0.7 max_tokens=512
```
(FastAPI only lists a model under *Schemas* at http://localhost:8000/docs once an endpoint uses it, so they appear there as you implement the endpoints.)

!!! note ":camera: Screenshot"
    Take a screenshot of the output of the `python -c` command, and upload it together with your solution as `f1_1.png`.


### Exercise 1.2 – Implement `/health` and `/models` endpoints

**`GET /health`** returns `{"status": "ok", "ollama": true|false}`:

- `ollama` is `true` if Ollama's model list endpoint (`GET /api/tags`) answers with HTTP 200 within 5 seconds, otherwise `false`.
- The endpoint itself must never fail, even if Ollama is not running.

**`GET /models`** returns `{"models": [...]}`:

- the names of the installed Ollama models that can **generate text** (from `GET /api/tags`). Embedding models such as `nomic-embed-text` cannot answer prompts and must not be listed: ask Ollama for each model's capabilities (`POST /api/show`) and keep the models with the `completion` capability,
- respond with **HTTP 503** and an explanatory `detail` if Ollama cannot be reached.

**Expected outcome:**
```bash
curl http://localhost:8000/health
# {"status":"ok","ollama":true}

curl http://localhost:8000/models
# {"models":["llama3.2:latest", ...]}          (no nomic-embed-text)
```
Then make Ollama unreachable and check that `/health` reports `"ollama": false` and `/models` returns 503. Either stop Ollama (quit the Ollama desktop app on macOS/Windows, or run `sudo systemctl stop ollama` on Linux), or temporarily set `OLLAMA_BASE_URL` in `config.py` to `http://localhost:1` (an address where nothing listens). Undo it afterwards.

!!! note ":camera: Screenshot"
    Take a screenshot of the `/health` and `/models` responses while Ollama is running, and both responses while Ollama is unreachable, and upload it together with your solution as `f1_2.png`.


### Exercise 1.3 – Implement `/generate` POST endpoint

`POST /generate` takes a `GenerateRequest` and returns a `GenerateResponse`:

- Generate a single, **non-streamed** completion with Ollama's `POST /api/generate` endpoint, passing the request's `model`, `prompt`, `temperature` and `max_tokens` (in Ollama this option is called `num_predict`).
- `token_count` is the number of generated tokens reported by Ollama.
- `latency_ms` is the total time spent in the endpoint, in milliseconds, rounded to 2 decimals.
- Follow the error-handling rules above (502 / 503).
- LLM responses can be slow: use a generous timeout (120 s).

Leave `cache_hit` and `similarity_score` at their defaults for now; semantic caching is added in Exercise 6.4.

**Expected outcome:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "model": "llama3.2"}'
# {"response":"The capital of France is Paris.","model":"llama3.2","latency_ms":1243.5,
#  "token_count":8,"cache_hit":false,"similarity_score":0.0}
```
Also check that `{"prompt": "Hi", "model": "no-such-model"}` returns 502, that a request without `prompt` returns 422, and that with Ollama unreachable (e.g. temporarily set `OLLAMA_BASE_URL` in `config.py` to `http://localhost:1`) the endpoint returns 503.

!!! note ":camera: Screenshot"
    Take a screenshot of the successful `/generate` response and the 502, 422 and 503 cases, and upload it together with your solution as `f1_3.png`.


### Exercise 1.4 – Implement `/generate/stream` POST endpoint

`POST /generate/stream` takes a `GenerateRequest` and returns the generated text as a **plain-text stream**, forwarding each piece of text as soon as Ollama produces it:

- Call Ollama with streaming enabled. Ollama then sends one JSON object per line; the text is in the `response` field and the last object has `"done": true`.
- Use FastAPI's `StreamingResponse` with an async generator, and consume Ollama's stream with `httpx`'s streaming API. Do **not** read the whole response into memory first.
- Respect `model`, `temperature` and `max_tokens`.
- Errors must be reported with a proper status code (502 / 503) **before** any text is streamed: once a `StreamingResponse` has started, its status code can no longer change. So check Ollama's response status before you return the `StreamingResponse`, and make sure the Ollama connection is closed when the stream ends.

**Expected outcome:** the response arrives token by token:
```bash
curl --no-buffer -X POST http://localhost:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Count slowly from 1 to 10"}'
```
and `{"prompt": "Hi", "model": "no-such-model"}` returns 502 (not 200 with an empty body).

!!! note ":camera: Screenshot"
    Take a screenshot of the streamed `curl` output and the 502 response for the unknown model, and upload it together with your solution as `f1_4.png`.


## Part 2 – Docker Containerization

Docker packages your application and all its dependencies into a **container**: an isolated, reproducible environment. A model-serving container behaves identically regardless of the host machine's Python version, OS or installed packages.

**Networking:** Ollama runs on your *host* machine, not inside the container. Inside a container, `localhost` means the container itself. Docker Desktop (Windows, macOS) provides the special hostname `host.docker.internal` to reach the host.

**Linux with Docker Engine (no Docker Desktop):** two extra steps are needed.

1. Map `host.docker.internal` to the host (Exercise 2.2).
2. By default Ollama only accepts connections from `127.0.0.1`, so requests from a container are refused. Make it listen on all interfaces: `sudo systemctl edit ollama`, add
    ```
    [Service]
    Environment="OLLAMA_HOST=0.0.0.0:11434"
    ```
    then `sudo systemctl daemon-reload && sudo systemctl restart ollama`. If a firewall is active, allow port 11434 from the Docker bridge (e.g. `sudo ufw allow in on docker0 to any port 11434`). Note that this exposes Ollama to your local network, unless the firewall blocks it.

**SQLite and Docker:** SQLite stores the database in a plain file and needs no database server. Mounting a host directory into the container makes the file survive container restarts and rebuilds.

Key Docker commands:
```bash
docker build -t lab-llm-api .                  # build an image from the Dockerfile in this folder
docker run -p 8000:8000 lab-llm-api            # run a container from the image
docker ps                                      # list running containers
docker logs <container_id>                     # view logs
docker compose up --build                      # build and start the services in docker-compose.yml
docker compose down                            # stop and remove them
```


### Exercise 2.1 – Write the Dockerfile

Write `Dockerfile` from scratch. Requirements:

1. Base image: the official **slim** image of **Python 3.12**.
2. The application lives in `/app` inside the image.
3. Dependencies are installed from `requirements_backend.txt` **in a separate layer before the source code is copied**, so that changing a `.py` file does not reinstall all packages on the next build. Do not keep pip's download cache in the image.
4. The rest of the source code is copied into the image.
5. The image documents that the app listens on port `8000`.
6. Starting a container runs the app with uvicorn on `0.0.0.0:8000` (no `--reload`).
7. Add a `.dockerignore` so that local-only files are **not** copied into the image: your virtual environment, `__pycache__` folders, the `data/` folder and any `*.db` file.

**Expected outcome:** `docker build -t lab-llm-api .` completes without errors, and `docker image ls lab-llm-api` shows an image of roughly 250–300 MB (a much larger image usually means that your `.venv` was copied). Change a comment in `main.py`, rebuild, and verify in the build output that the dependency layer is `CACHED`.

!!! note ":camera: Screenshot"
    Take a screenshot of the output of the second `docker build` (with the `CACHED` dependency step) and of `docker image ls lab-llm-api`, and upload it together with your solution as `f2_1.png`.


### Exercise 2.2 – Write docker-compose.yml

Define one service called `api` in `docker-compose.yml`:

- built from the `Dockerfile` in the current directory,
- host port `8000` mapped to container port `8000`,
- environment variables:
    - `OLLAMA_BASE_URL=http://host.docker.internal:11434`
    - `DB_PATH=/data/conversations.db`
- the local `./data` directory mounted at `/data` in the container,
- `host.docker.internal` must resolve to the host also on Linux (look up `extra_hosts` and the special value `host-gateway`).

**Expected outcome:** `docker compose up --build` starts the API at http://localhost:8000. Stop your local `uvicorn` first: both use port 8000, and Docker cannot publish a port that is already in use ("port is already allocated").

!!! note "Note"
    At this point the server starts, but every endpoint that calls Ollama fails (and `/health` reports `"ollama": false`), because `OLLAMA_BASE_URL` in `config.py` is still hard-coded to `localhost`. This is expected and is fixed in Exercise 2.3.

!!! note ":camera: Screenshot"
    Take a screenshot of the running `docker compose up --build` output showing that the API started, and upload it together with your solution as `f2_2.png`.


### Exercise 2.3 – Read the Ollama URL from an environment variable

Make `OLLAMA_BASE_URL` in `config.py` configurable: use the value of the `OLLAMA_BASE_URL` environment variable if it is set, and fall back to `http://localhost:11434` otherwise. (We deliberately do not use `OLLAMA_HOST`: Ollama itself reads that variable as its *listen address*, e.g. `0.0.0.0`, which is not a valid URL for a client.)

**Expected outcome:** `uvicorn main:app` works locally **and** in `docker compose up --build`: `/health` reports `"ollama": true` in both. Check them one after the other (stop one before starting the other, as both use port 8000).

!!! note ":camera: Screenshot"
    Take a screenshot of the `/health` response with `"ollama": true` from `uvicorn` running locally and, after stopping it, from the Docker container, and upload it together with your solution as `f2_3.png`.


## Part 3 – Postman: Raw API Inspection

Before building a frontend, it is valuable to interact with your API at the raw HTTP level. You see exactly what JSON travels over the wire: the same data your Streamlit app will send.

This part uses your running Docker container from Part 2.


### Exercise 3.1 – Test your endpoints in Postman

Create a Postman Collection called `LLM API Lab` with the following requests:

1. **Health check:** `GET http://localhost:8000/health`
2. **List models:** `GET http://localhost:8000/models`
3. **Generate (non-streaming):** `POST http://localhost:8000/generate` with the raw JSON body
    ```json
    {"prompt": "Explain what a REST API is in two sentences.", "model": "llama3.2", "temperature": 0.7}
    ```

4. **Generate (streaming):** `POST http://localhost:8000/generate/stream` with the same body. Postman may buffer the response and show it all at once; use `curl --no-buffer` to observe the actual streaming.
5. **Validation error:** `POST http://localhost:8000/generate` with a body that has no `prompt`. Look at the 422 response FastAPI generates.

!!! note ":camera: Screenshot"
    Take a screenshot of request 1 (health check) in Postman, showing the request and the full response including the status code, and upload it together with your solution as `f3_1a.png`.

!!! note ":camera: Screenshot"
    Take a screenshot of request 2 (list models), and upload it together with your solution as `f3_1b.png`.

!!! note ":camera: Screenshot"
    Take a screenshot of request 3 (generate), and upload it together with your solution as `f3_1c.png`.

!!! note ":camera: Screenshot"
    Take a screenshot of request 4 (generate, streaming), and upload it together with your solution as `f3_1d.png`.

!!! note ":camera: Screenshot"
    Take a screenshot of request 5 (validation error), and upload it together with your solution as `f3_1e.png`.


## Part 4 – Conversation Endpoint + SQLite History

Stateless endpoints like `/generate` treat every request in isolation: the model has no memory of previous turns. A real chat needs **conversation history**: each new message is sent together with all previous messages.

You store the history in **SQLite**, accessed asynchronously with `aiosqlite` so the database does not block FastAPI's event loop. The schema:

```
sessions              messages
─────────────────     ──────────────────────────────────────────
id (TEXT PK)          id (INTEGER PK AUTOINCREMENT)
created_at (TEXT)     session_id (TEXT, FK → sessions.id)
                      role (TEXT: "user" or "assistant")
                      content (TEXT)
                      timestamp (TEXT)
```

For multi-turn conversations, use Ollama's **chat** endpoint (`POST /api/chat`), which takes the full message list:

```json
{
  "model": "llama3.2",
  "messages": [
    {"role": "user",      "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user",      "content": "What is Python?"}
  ],
  "stream": false
}
```


### Exercise 4.1 – Implement the database schema

Implement `init_db()` in `database.py`:

- create both tables exactly as in the schema above; every column except the primary keys is `NOT NULL`,
- it must be safe to call on every server start (it must not fail or delete data if the tables already exist),
- the foreign key must actually be **enforced**: SQLite ignores foreign keys unless they are switched on for every connection (`PRAGMA foreign_keys = ON`). Inserting a message for a session that does not exist must fail.

`main.py` already calls `init_db()` on startup through the `lifespan` handler.

**Expected outcome:** the database file (`DB_PATH`, default `./conversations.db`) is created and contains both tables. Save the following script as `show_schema.py` and run it with `python show_schema.py`; it prints the two `CREATE TABLE` statements.

```python
import asyncio
import sqlite3

import database

asyncio.run(database.init_db())
query = """
    SELECT sql FROM sqlite_master
    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
"""
for (sql,) in sqlite3.connect(database.DB_PATH).execute(query):
    print(sql)
```

!!! note ":camera: Screenshot"
    Take a screenshot of the output of `show_schema.py`, and upload it together with your solution as `f4_1.png`.


### Exercise 4.2 – Implement session and message operations

Implement the remaining functions in `database.py`. Use **parameterized queries** (never build SQL with f-strings) and commit after every write.

- **4.2a `create_session(session_id)`:** insert a session with the current UTC time (ISO-8601). Creating an existing session again must silently do nothing.
- **4.2b `session_exists(session_id) -> bool`**
- **4.2c `save_message(session_id, role, content)`:** insert a message with the current UTC time.
- **4.2d `get_history(session_id) -> list[dict]`:** the messages of the session as `[{"role": ..., "content": ...}, ...]` **in the order they were saved** (even if two messages have the same timestamp); an empty list for a session without messages.

**Expected outcome:** save the following script as `check_database.py` and run it with `python check_database.py`. It uses a separate database file and prints `True False`, the two messages in the order they were saved, and that the foreign key is enforced.

```python
import asyncio
import os

os.environ["DB_PATH"] = "check.db"
if os.path.exists("check.db"):
    os.remove("check.db")

import database  # noqa: E402  (imported after DB_PATH is set)


async def main():
    await database.init_db()
    await database.create_session("demo")
    await database.create_session("demo")  # a duplicate must not raise
    await database.save_message("demo", "user", "Hi")
    await database.save_message("demo", "assistant", "Hello!")
    print("exists:", await database.session_exists("demo"), await database.session_exists("nope"))
    print("history:", await database.get_history("demo"))
    try:
        await database.save_message("nope", "user", "orphan")
        print("foreign key NOT enforced")
    except Exception as e:
        print("foreign key enforced:", type(e).__name__, e)


asyncio.run(main())
```

!!! note ":camera: Screenshot"
    Take a screenshot of the output of `check_database.py`, and upload it together with your solution as `f4_2.png`.


### Exercise 4.3 – Implement `POST /conversation`

`POST /conversation` takes a `ConversationRequest` and returns a `ConversationResponse`:

1. If the session does not exist yet, create it.
2. Send the **complete history** of the session plus the new user message to Ollama's chat endpoint (non-streamed).
3. Only when Ollama answered successfully, store **both** the user's message and the assistant's reply. A failed call must not leave a lone user message in the history (otherwise a retry would send the message twice).
4. Return the reply together with the `session_id` and the total `latency_ms`.
5. Follow the error-handling rules of Part 1 (502 / 503).

**Expected outcome:**
```bash
curl -X POST http://localhost:8000/conversation \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test01", "message": "My name is Alice.", "model": "llama3.2"}'

# Second message — the model must remember the name
curl -X POST http://localhost:8000/conversation \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test01", "message": "What is my name?", "model": "llama3.2"}'
# the response mentions "Alice"
```
Restart the server (or the container) and ask again: the history must survive the restart. Then send a message with `"model": "no-such-model"`: you get a 502, and `GET /conversation/test01` shows no new message.

!!! note ":camera: Screenshot"
    Take a screenshot of the two conversation requests (the model remembers the name), the request after the restart, and the failed request followed by `GET /conversation/test01` without the failed message, and upload it together with your solution as `f4_3.png`.


### Exercise 4.4 – Implement `GET /conversation/{session_id}`

- Return `{"session_id": ..., "messages": [...]}` with the full history of the session.
- Respond with **HTTP 404** if the session does not exist.

**Expected outcome:**
```bash
curl http://localhost:8000/conversation/test01
# {"session_id":"test01","messages":[{"role":"user","content":"My name is Alice."}, ...]}

curl -i http://localhost:8000/conversation/does-not-exist
# HTTP/1.1 404 Not Found
```

!!! note ":camera: Screenshot"
    Take a screenshot of the history of `test01` and the 404 response, and upload it together with your solution as `f4_4.png`.


## Part 5 – Streamlit Chat UI

You already know Streamlit from previous labs. Here you build a chat application backed by the `/conversation` endpoint of Part 4. Remember that Streamlit re-runs the whole script on every interaction, so anything that must survive a rerun belongs in `st.session_state`.

Run the Streamlit app with:
```bash
python -m streamlit run app.py
```


### Exercise 5.1 – Implement the Chat interface

`app.py` provides the page setup and the `get_models()` helper. Implement the rest:

**5.1a — Session state**

- Every browser session gets its own conversation id (an 8-character random id, e.g. from `uuid4`), created once and kept across reruns.
- Keep the list of displayed messages in the session state as well.

**5.1b — Session controls, model selector and history**

- Show the current session id, and a **New Session** button that starts a new conversation (new id, empty chat).
- A model selector filled from `get_models()`, with `llama3.2:latest` preselected when it is available.
- Display all previous messages of the session with `st.chat_message`.

**5.1c — New message**

- When the user submits a message (`st.chat_input`), show it immediately.
- Send it to `POST /conversation` with the session id and the selected model (timeout 120 s), showing a spinner while waiting.
- Display the assistant's reply and, below it, the latency in milliseconds.
- If the request fails for any reason, show a readable error message in the chat instead of crashing the app.
- Store both messages so they are shown again after the next rerun.

**Expected outcome:** a working chat where the model remembers earlier messages of the same session, and **New Session** starts a conversation without memory.

!!! note ":camera: Screenshot"
    Take a screenshot of the running chat app with at least two turns in which the model demonstrates memory (e.g. you tell it your name, then ask it to recall it), and upload it together with your solution as `f5_1a.png`.

!!! note ":camera: Screenshot"
    Take a screenshot of the chat after clicking **New Session**, showing that the new session does not remember the previous one, and upload it together with your solution as `f5_1b.png`.


## Part 6 – Semantic Caching

LLM inference is expensive, both in cost (API calls) and time (seconds per request). **Semantic caching** avoids redundant calls by detecting that a new prompt is semantically similar to one that was already answered.

Unlike exact-match caching (which only catches identical strings), semantic caching uses **vector embeddings** to measure similarity of meaning:

1. When a prompt arrives, compute its embedding vector with `nomic-embed-text`.
2. Compare it with the cached prompt embeddings using **cosine similarity**.
3. If the similarity reaches the threshold (default 0.92), return the cached response immediately: no LLM call.
4. Otherwise call the LLM, then store the prompt embedding and the response in the cache.

A cached answer must only be reused where it is interchangeable. In this lab the cache is split into **namespaces**: one per model *and* `max_tokens` value, so an answer generated by one model (or cut off at 10 tokens) is never returned for a request to another model (or with a 512-token limit).

**Cosine similarity** measures the angle between two vectors. Semantically similar sentences point in roughly the same direction in embedding space:
```
similarity = dot(a, b) / (||a|| × ||b||)
# 1.0 = same direction, 0.0 = orthogonal, -1.0 = opposite
```

The cache is kept **in memory** for this lab (a Python list). In production it would be a vector database (e.g. Redis, Qdrant or pgvector).

Ollama's embedding endpoint:
```
POST http://localhost:11434/api/embed
{"model": "nomic-embed-text", "input": "your text here"}
→ {"embeddings": [[0.123, -0.456, ...]], ...}   # one 768-dimensional vector per input
```


### Exercise 6.1 – Implement embedding generation

Implement `get_embedding(text)` in `cache.py`: return the embedding vector of `text` computed by `EMBED_MODEL` through Ollama (async, 30 s timeout; raise an exception on an HTTP error).

**Expected outcome:** `get_embedding("hello world")` returns a list of 768 floats:
```bash
python -c "import asyncio, cache; e = asyncio.run(cache.get_embedding('hello world')); print(len(e), e[:3])"
```

!!! note ":camera: Screenshot"
    Take a screenshot of the output of the embedding command, and upload it together with your solution as `f6_1.png`.


### Exercise 6.2 – Implement cosine similarity

Implement `cosine_similarity(a, b)` using only the standard library (no numpy). Return `0.0` if either vector has zero length.

**Expected outcome:** `cosine_similarity([1, 0], [1, 0]) == 1.0`, `cosine_similarity([1, 0], [0, 1]) == 0.0`, `cosine_similarity([1, 0], [-1, 0]) == -1.0`, `cosine_similarity([0, 0], [1, 0]) == 0.0`:
```bash
python -c "from cache import cosine_similarity as c; print(c([1, 0], [1, 0]), c([1, 0], [0, 1]), c([1, 0], [-1, 0]), c([0, 0], [1, 0]))"
```

!!! note ":camera: Screenshot"
    Take a screenshot of the output of the cosine similarity command, and upload it together with your solution as `f6_2.png`.


### Exercise 6.3 – Implement model-aware cache check and store

- **6.3a `check_cache(prompt_embedding, namespace, threshold)`:** find the most similar cached entry **in the same namespace** and return `(hit, cached_response, best_similarity)` as described in the docstring.
- **6.3b `store_in_cache(prompt, namespace, embedding, response)`:** append a new entry to `_cache`.

**Expected outcome:** save the following script as `check_cache.py` and run it with `python check_cache.py`. Only the first check is a hit.

```python
import cache

cache.clear_cache()
cache.store_in_cache("q1", "llama3.2:latest|512", [1.0, 0.0], "A1")
print("same namespace, same vector: ", cache.check_cache([1.0, 0.0], "llama3.2:latest|512"))
print("same namespace, other vector:", cache.check_cache([0.0, 1.0], "llama3.2:latest|512"))
print("other namespace, same vector:", cache.check_cache([1.0, 0.0], "llama3.2:1b|512"))
```

!!! note ":camera: Screenshot"
    Take a screenshot of the output of `check_cache.py`, and upload it together with your solution as `f6_3.png`.


### Exercise 6.4 – Add semantic caching to `POST /generate`

Extend your `/generate` endpoint from Exercise 1.3:

1. Build the namespace from the model and `max_tokens`. Ollama treats `llama3.2` and `llama3.2:latest` as the same model, so normalize the model name first (add `:latest` when no tag is given). Compute the embedding of the prompt and check the cache.
2. On a **hit**, return the cached response immediately with `cache_hit=true`, `token_count=0` and the similarity score (rounded to 4 decimals), without calling the LLM.
3. On a **miss**, generate as before, store the new response in the cache, and return `cache_hit=false` together with the best similarity that was found.

`latency_ms` must include the embedding time in both cases. `GET /cache/stats` shows the number of cached entries.

The cache is an optimization, not a requirement: if the embedding call fails (e.g. `nomic-embed-text` is not installed), log a warning and serve the request **without** the cache instead of failing it.

**Expected outcome:** sending the same prompt twice returns `cache_hit: true` the second time, with a latency that is an order of magnitude lower. Sending it again with a different `model` or `max_tokens` is a miss; `"llama3.2"` and `"llama3.2:latest"` share the cache.

!!! note ":camera: Screenshot"
    Take a screenshot of the cache miss and the cache hit for the same prompt, the misses for another model and another `max_tokens`, the shared hit for `llama3.2:latest`, and `GET /cache/stats`, and upload it together with your solution as `f6_4.png`.


### Exercise 6.5 – Threshold experiment

Restart the server (to empty the cache), then send these prompts to `/generate` in this order, all with `"model": "llama3.2"`, and record `cache_hit` and `similarity_score` for each:

| # | Prompt | Expected |
|---|--------|----------|
| 1 | `What is the capital of France?` | miss (empty cache) |
| 2 | `What is the capital of France?` | hit (identical) |
| 3 | `Tell me France's capital city` | ? |
| 4 | `Capital of France?` | ? |
| 5 | `What is the capital of Germany?` | ? |
| 6 | `What is the population of France?` | ? |

Then repeat the experiment with the thresholds **0.95** and **0.75** by passing the `threshold` argument to `check_cache()` in `main.py`.

**Deliverable** (in `answers.md`, under `## Exercise 6.5`): a table with the similarity score and the hit/miss result of every prompt for each threshold, and a short answer:

- Which threshold would you use in production and why?
- Did you find a **false positive** (a cache hit for a prompt that needs a different answer)? What would it cost the user?
- The namespace ignores `temperature`. When is that acceptable, and when would you add it to the namespace?

!!! question ":pencil: Written answer"
    Answer this question in `answers.md`, under the heading `## Exercise 6.5`.

!!! note ":camera: Screenshot"
    Take a screenshot of the Postman responses of prompts 3–6 at the default threshold, and upload it together with your solution as `f6_5a.png`.

!!! note ":camera: Screenshot"
    Take a screenshot of the Postman response of a false positive at a lower threshold, and upload it together with your solution as `f6_5b.png`. This screenshot is only required if you found a false positive.

## Giving feedback

Please share your opinion about this laboratory in `answers.md`, under the heading `## Feedback`. You can include:

- Your general impression of the laboratory.
- The things you liked or did not like.
- Whether you found the lab easy or difficult.
- Which exercises you enjoyed the most.
- What was necessary but not included in this laboratory.
- What was unnecessary but included in this laboratory.
