---
authors: domonkosadam
---

# 04 - Building your own RAG pipeline with LangChain

## Goal

The main goal of this laboratory is to understand how Retrieval-Augmented Generation (RAG) works by building a RAG system from scratch (embeddings, vector search, chunking, retrieval, prompting), then rebuilding it with LangChain, extending it into a conversational agent, and finally putting it behind a command-line and a web interface. The knowledge base is Simple English Wikipedia.

The structure of this laboratory is as follows:

1. Installing the required dependencies and downloading the dataset.
1. Dataset preparation [1 point]
1. Vector Databases & FAISS [4 points in total]
    1. Load embedding model [0.5 point]
    1. Create sample embeddings [0.5 point]
    1. Compute similarity [1 point]
    1. Build FAISS index [1 point]
    1. Search with FAISS [1 point]
1. RAG from Scratch [6 points in total]
    1. Document chunking [0.5 point]
    1. Process documents [0.5 point]
    1. Create embeddings & build a cosine-similarity index [1 point]
    1. Implement retrieval function [2 points]
    1. Build complete RAG pipeline [2 points]
1. LangChain & Advanced Patterns [5 points in total]
    1. LangChain document processing [0.5 point]
    1. LangChain vector store [0.5 point]
    1. RAG chain with LCEL [2 points]
    1. Conversational agentic RAG with memory [2 points]
1. Interactive Q&A Interface [4 points in total]
    1. Command-line interface [2 points]
    1. Streamlit web interface [2 points]
1. Giving feedback [+1 point]

!!! info "Grading"
    In order to pass this laboratory, you must obtain at least 8 points out of 20.

!!! important "Screenshot requirement"
    At the end of **every** exercise (marked with a :camera: **Screenshot** note), you must take a screenshot of your working solution, including its output, and upload it together with your solution, using the exact filename given in that note (e.g. `f0.png`, `f2_5.png`). Solutions missing the required screenshots, or using the wrong filename, will not be accepted.

    If an output does not fit on one screen, split it into several screenshots with a numeric suffix (e.g. `f3_4a_1.png`, `f3_4a_2.png`).

## Preparation

Don't forget to follow the assignment submission process described under [GitHub](../../information/github.md) while working on this laboratory.

!!! important "Reviewer"
    When creating the pull request for this laboratory, assign it to the `domonkosadam` GitHub user.

!!! tip "Jupyter notebook or plain Python files"
    You may solve Parts 0–3 either in a Jupyter notebook or in a plain `.py` file, whichever you prefer. The code snippets build on each other, so keep them in **one** notebook or module: in Part 4, the command-line and web applications import your functions from it. Name it `RAG_Lab_Notebook.ipynb` (or `RAG_Lab_Notebook.py`). Regardless of the format you choose, make sure that **everything** (code, `answers.md` and the required screenshots) is committed and pushed to your solution branch. Do **not** commit the dataset (`data/`).

!!! important "Written answers"
    Some exercises contain questions (marked with :pencil: **Written answer**). Answer them in a file called `answers.md` in the root of your repository, under a heading with the exercise number (e.g. `## Exercise 1.2`). Your feedback at the end of the laboratory also goes into this file (under `## Feedback`). Make sure `answers.md` is committed and pushed together with your solution and screenshots.

## Setup

### Ollama

Install Ollama from <https://ollama.com/download> (on macOS also `brew install ollama`, on Linux `curl -fsSL https://ollama.com/install.sh | sh`), then pull the model used in this laboratory:

```bash
ollama pull llama3.2:3b
ollama list
```

### Python dependencies

!!! important "Mandatory"
    You need **Python 3.11 – 3.14** on Windows (x64), Linux (x64/arm64) or macOS 14+ (Apple Silicon). *Intel Mac:* use Python 3.11 or 3.12; the requirements file then installs an older, compatible PyTorch/Transformers stack automatically. *Windows on ARM:* install the x64 build of Python (it runs under emulation), because some packages have no native Windows-ARM builds.

    Create a virtual environment and install the pinned dependencies. The requirements file pins the exact version of **every** package, including all indirect dependencies (with platform-specific versions where needed), so everyone gets the same, tested environment. On Linux, PyTorch comes with its CUDA libraries, so expect about 3 GB of downloads. If `pip` reports success but installed almost nothing (check with `pip list`), your Python version / platform combination is not supported (see above).

    ```bash
    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r https://raw.githubusercontent.com/VIAUMSMA001/laboratories/main/docs/laboratories/04-custom-rag-langchain/requirements.txt
    ```

The main packages in the requirements file are:

```
faiss-cpu==1.13.2
ipywidgets==8.1.9
jupyterlab==4.6.4
langchain-chroma==1.1.0
langchain-core==1.6.5
langchain-huggingface==1.2.2
langchain-ollama==1.1.0
langchain-text-splitters==1.1.2
langchain==1.4.2
langgraph==1.2.12
nbconvert==7.17.1
numpy==2.4.6   (Intel Mac: 1.26.4)
pandas==3.0.6
pyarrow==25.0.1
requests==2.34.2
sentence-transformers==6.1.0   (Intel Mac: 5.7.0)
streamlit==1.64.0
```

!!! warning "Troubleshooting (macOS)"
    If Python crashes with a *segmentation fault* when FAISS and PyTorch are used together, make sure you installed exactly the pinned `faiss-cpu==1.13.2`. As a last resort, start Jupyter (or your script) with the environment variable `OMP_NUM_THREADS=1`, e.g. `OMP_NUM_THREADS=1 jupyter lab`.

## Part 0 – Dataset Preparation

### Step 1: Download the Dataset from Hugging Face

1. Visit: **https://huggingface.co/datasets/wikimedia/wikipedia**
2. Select configuration: **20231101.simple** (Simple English Wikipedia)
3. Go to the "Files and versions" tab
4. Download: `20231101.simple/train-00000-of-00001.parquet` (~150 MB)
5. Save it as `data/wikipedia.parquet` (create a `data/` folder next to your notebook or script)

**Alternative – Hugging Face CLI** (`huggingface_hub` is already installed as a dependency):
```bash
hf download wikimedia/wikipedia 20231101.simple/train-00000-of-00001.parquet --repo-type dataset --local-dir ./data
mv data/20231101.simple/train-00000-of-00001.parquet data/wikipedia.parquet
```

### Step 2: Load the Dataset

```python
import pandas as pd

# True when this code runs in Jupyter or as a script, False when it is imported as a module by the Part 4 apps.
# The demo/test code below only runs when it is True, which keeps the import in Part 4 faster.
RUN_DEMOS = __name__ == "__main__"

DATA_PATH = "data/wikipedia.parquet"

# TODO: Load the parquet file into a DataFrame called `df`.
df = None

print(f"Loaded {len(df)} articles")
print(f"Columns: {df.columns.tolist()}")
```

### Step 3: Convert to Document Format

To keep the lab fast we only use the first `N_DOCS` articles. Feel free to increase it later to try the full dataset.

```python
N_DOCS = 1000

# TODO: Build `documents`: a list of dicts with the keys 'title', 'text' and 'url',
#       one dict per article, limited to the first N_DOCS articles.
documents = []

print(f"Prepared {len(documents)} documents")
```

!!! note ":camera: Screenshot"
    Take a screenshot of the number of loaded articles, the column names and the number of prepared documents, and upload it together with your solution as `f0.png`.

## Part 1 – Vector Databases & FAISS

Vector databases enable semantic search by storing and retrieving high-dimensional embeddings. Unlike keyword search, vector search finds documents based on meaning similarity. FAISS (Facebook AI Similarity Search) is an efficient library for similarity search over millions of vectors, which makes it a good fit for RAG systems.

### Exercise 1.1 – Load Embedding Model

Load `all-MiniLM-L6-v2` (a fast, compact sentence-embedding model) with the `sentence-transformers` library and determine the dimension of the vectors it produces. (Many tutorials use a method name that was renamed recently; if you get a `FutureWarning`, switch to the new name.)

```python
from sentence_transformers import SentenceTransformer

# TODO: Load the 'all-MiniLM-L6-v2' model.
embedding_model = None

# TODO: Store the dimension of the vectors produced by the model (ask the model, do not hard-code it).
EMBEDDING_DIM = None

print(f"Embedding dimension: {EMBEDDING_DIM}")
```

!!! note ":camera: Screenshot"
    Take a screenshot of the printed embedding dimension, and upload it together with your solution as `f1_1.png`.

### Exercise 1.2 – Create Sample Embeddings

```python
sample_texts = [
    "The cat sat on the mat",
    "A feline rested on the rug",
    "Dogs are loyal animals",
    "Python is a programming language",
    "Machine learning uses neural networks",
]

# TODO: Embed all sample_texts in a single call.
sample_embeddings = None

print(f"Shape: {sample_embeddings.shape}")
```

!!! note ":camera: Screenshot"
    Take a screenshot of the printed shape of the sample embeddings, and upload it together with your solution as `f1_2.png`.

### Exercise 1.3 – Compute Similarity

1. Implement `cosine_similarity(vec1, vec2)` with NumPy (do not use a library function that computes it for you).
2. Compute the full similarity matrix of the five sample sentences and print the pair of **different** sentences that is most similar.

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    # TODO
    pass

print(f"Similarity (cat vs feline): {cosine_similarity(sample_embeddings[0], sample_embeddings[1]):.3f}")
print(f"Similarity (cat vs dog):    {cosine_similarity(sample_embeddings[0], sample_embeddings[2]):.3f}")

# TODO: Build the 5x5 similarity matrix and print the most similar pair of different sentences.
```

!!! note ":camera: Screenshot"
    Take a screenshot of the two printed similarities and the most similar pair of different sentences, and upload it together with your solution as `f1_3.png`.

### Exercise 1.4 – Build FAISS Index

Create an exact-search FAISS index that uses **L2 (Euclidean) distance** and add the sample embeddings to it. Exact (flat) search is linear in the number of vectors, which is acceptable below ~1M vectors.

```python
import faiss

# TODO: Create the index and add the sample embeddings.
index = None

print(f"Vectors in index: {index.ntotal}")
```

!!! note ":camera: Screenshot"
    Take a screenshot of the printed number of vectors in the index, and upload it together with your solution as `f1_4.png`.

### Exercise 1.5 – Search with FAISS

Find the 3 sample sentences closest to the query.

```python
query = "The kitten is sleeping"
k = 3

# TODO: Search the index. `distances` and `indices` must be the arrays returned by FAISS.
distances = None
indices = None

print(f"Query: '{query}'\n")
for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
    print(f"{rank}. {sample_texts[idx]} (L2 distance: {dist:.3f})")
```

!!! note ":camera: Screenshot"
    Take a screenshot of the three search results with their L2 distances, and upload it together with your solution as `f1_5.png`.

## Part 2 – Building RAG from Scratch

### Exercise 2.1 – Document Chunking

Chunking breaks long documents into smaller segments that fit the LLM context window and make retrieval more precise. Here we chunk by **characters** (not tokens): 500 characters with a 50-character overlap between neighboring chunks.

Requirements:

- consecutive chunks overlap by exactly `overlap` characters,
- no chunk may be completely contained in the previous one (i.e. no tiny duplicate chunk at the end),
- an empty text yields an empty list,
- raise a `ValueError` if `overlap >= chunk_size`.

The self-check below must pass.

```python
from typing import List

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split `text` into overlapping character chunks."""
    # TODO
    pass
```

```python
# Self-check (do not modify)
assert chunk_text("abcdefghij", chunk_size=4, overlap=1) == ["abcd", "defg", "ghij"]
assert chunk_text("abc", chunk_size=4, overlap=1) == ["abc"]
assert chunk_text("", chunk_size=4, overlap=1) == []
try:
    chunk_text("abc", chunk_size=4, overlap=4)
    raise AssertionError("expected ValueError")
except ValueError:
    pass
print("chunk_text passed all checks")
```

!!! note ":camera: Screenshot"
    Take a screenshot of the passing self-check, and upload it together with your solution as `f2_1.png`.

### Exercise 2.2 – Process Documents

Chunk every document. Keep the chunk texts in `all_chunks` and, **at the same position**, a metadata dict in `chunk_metadata` with the keys `title`, `url` and `chunk_id` (the index of the chunk within its document).

```python
all_chunks = []
chunk_metadata = []

# TODO

print(f"{len(documents)} documents -> {len(all_chunks)} chunks")
```

!!! note ":camera: Screenshot"
    Take a screenshot of the printed number of documents and chunks, and upload it together with your solution as `f2_2.png`.

### Exercise 2.3 – Create Embeddings & Build a Cosine-Similarity Index

In Part 1 you used L2 distance. For text retrieval, **cosine similarity** is the usual choice. FAISS has no dedicated cosine index, but you can get cosine similarity from an **inner-product** index if you prepare the vectors correctly.

1. Embed all chunks (this can take a few minutes – a progress bar helps).
2. Build `rag_index` so that `rag_index.search()` returns cosine similarities (higher = more similar).
3. Answer the question below.

!!! warning "Long-running step"
    Embedding all chunks may take 1–10 minutes depending on your hardware.

```python
# TODO
chunk_embeddings = None
rag_index = None

print(f"Vectors in index: {rag_index.ntotal}")
```

!!! note ":camera: Screenshot"
    Take a screenshot of the printed number of vectors in the cosine-similarity index, and upload it together with your solution as `f2_3.png`.

**Question:** Why does an inner-product index return cosine similarity in your setup? What would go wrong without your preparation step?

!!! question ":pencil: Written answer"
    Answer this question in `answers.md`, under the heading `## Exercise 2.3`.

### Exercise 2.4 – Implement Retrieval Function

Return the `k` most relevant chunks for a query as a list of dicts with the keys `text`, `title`, `url` and `score` (cosine similarity as a Python `float`, most similar first).

```python
from typing import Dict

def retrieve_documents(query: str, k: int = 3) -> List[Dict]:
    # TODO
    pass

results = retrieve_documents("How does a computer work?", k=3)
for rank, doc in enumerate(results, start=1):
    print(f"\n{rank}. {doc['title']} (cosine similarity: {doc['score']:.3f})")
    print(f"   {doc['text'][:150]}...")
```

!!! note ":camera: Screenshot"
    Take a screenshot of the retrieved documents with their cosine similarities, and upload it together with your solution as `f2_4.png`.

### Exercise 2.5 – Build Complete RAG Pipeline

`query_llm()` is provided. Implement `rag_query(question, k)`, which retrieves context, builds a prompt and asks the LLM. It must return a dict with the keys `question`, `answer` and `sources` (list of dicts with `title` and `url`).

Your prompt must make the model answer **only from the retrieved context** and admit when the context does not contain the answer. The second test question checks this.

```python
import requests

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b"

def query_llm(prompt: str, model: str = LLM_MODEL) -> str:
    """Send a prompt to a local Ollama model and return the generated text."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Is it running? Try: ollama serve")
    if response.status_code != 200:
        raise RuntimeError(f"Ollama returned {response.status_code}: {response.text}. Did you run: ollama pull {model}?")
    return response.json()["response"]


def rag_query(question: str, k: int = 3) -> Dict:
    # TODO
    pass


if RUN_DEMOS:
    for question in ["What is photosynthesis?", "Who won the 2030 FIFA World Cup?"]:
        result = rag_query(question)
        print(f"\nQuestion: {question}\nAnswer: {result['answer']}")
        print("Sources:")
        for i, doc in enumerate(result["sources"], start=1):
            print(f"  {i}. {doc['title']} - {doc['url']}")
```

!!! note ":camera: Screenshot"
    Take a screenshot of the answers and sources for both test questions, and upload it together with your solution as `f2_5.png`.

## Part 3 – LangChain & Advanced Patterns

In Part 2 you built a complete RAG system from scratch. Production systems usually rely on a framework instead. In this part you rebuild the system with **LangChain 1.x**:

- **LCEL** (LangChain Expression Language): compose runnables with the `|` operator
- **Vector store integrations**: FAISS no longer has an official LangChain integration (the `langchain-community` package it lived in was sunset in 2026), so we use **Chroma** through the official `langchain-chroma` package. You keep using raw FAISS where you already know it (Parts 1–2).
- **Agents** (`create_agent`) with tools and memory for conversational RAG

Only use the modern packages imported in the code snippets (`langchain`, `langchain_core`, `langchain_chroma`, …). The legacy `langchain_classic` / `langchain_community` packages are not allowed.

Reference documentation: https://docs.langchain.com/oss/python/langchain/overview

### Exercise 3.1 – LangChain Document Processing

Split the LangChain documents into chunks of 500 characters with 50 characters of overlap using the recursive character splitter.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

lc_documents = [
    Document(page_content=doc["text"], metadata={"title": doc["title"], "url": doc["url"]})
    for doc in documents
]

# TODO
text_splitter = None
lc_chunks = None

print(f"{len(lc_documents)} documents -> {len(lc_chunks)} chunks")
```

!!! note ":camera: Screenshot"
    Take a screenshot of the printed number of documents and chunks, and upload it together with your solution as `f3_1.png`.

**Question:** Your `chunk_text()` produced a different number of chunks than the recursive splitter. Why? Which one produces better chunks for retrieval?

!!! question ":pencil: Written answer"
    Answer this question in `answers.md`, under the heading `## Exercise 3.1`.

### Exercise 3.2 – LangChain Vector Store

Create an in-memory Chroma vector store from `lc_chunks`, using the same `all-MiniLM-L6-v2` model through LangChain's Hugging Face embeddings wrapper.

Chroma keeps in-memory collections for the lifetime of the Python process, so a naive implementation adds every chunk **again** each time you re-run the code (and duplicate chunks then fill your top-k results). Your code must be safe to re-run: run it twice and make sure the provided check passes. (In a notebook, run the cell twice. In a `.py` file every run starts a fresh process, so wrap the store-building code in a function and call it twice in the same run.)

Note: Chroma accepts only a limited number of documents in a single `add_documents()` call (5,461 in the pinned version, fewer than our chunks); `Chroma.from_documents()` splits the data into batches for you.

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# TODO
lc_embeddings = None
vectorstore = None

stored = len(vectorstore.get(include=[])["ids"])
assert stored == len(lc_chunks), f"{stored} vectors stored for {len(lc_chunks)} chunks - re-running duplicated the data"

for doc, distance in vectorstore.similarity_search_with_score("How does a computer work?", k=3):
    print(f"{doc.metadata['title']} (distance: {distance:.3f}): {doc.page_content[:100]}...")
```

!!! note ":camera: Screenshot"
    Take a screenshot of the output of this code after building the vector store a **second** time in the same Python process (the duplicate check must pass), and upload it together with your solution as `f3_2.png`.

### Exercise 3.3 – RAG Chain with LCEL

Build `qa_chain`, a single LCEL runnable that takes the question string and returns a dict with:

- `input`: the question,
- `context`: the list of retrieved `Document` objects (top 3),
- `answer`: the LLM answer as a plain string.

Requirements:

- use `vectorstore.as_retriever(...)` for retrieval,
- the prompt is a `ChatPromptTemplate` with a system message (containing the context) and a human message (containing the question),
- the model must only answer from the context,
- build it only from `langchain_core` runnables (e.g. `RunnableParallel`, `RunnablePassthrough`, `.assign()`, `|`).

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama

llm = ChatOllama(model=LLM_MODEL, temperature=0)

# TODO
retriever = None
qa_prompt = None
qa_chain = None

if RUN_DEMOS:
    result = qa_chain.invoke("What is photosynthesis?")
    print(f"Answer: {result['answer']}")
    for i, doc in enumerate(result["context"], start=1):
        print(f"{i}. {doc.metadata['title']} - {doc.metadata['url']}")
```

!!! note ":camera: Screenshot"
    Take a screenshot of the answer and its sources, and upload it together with your solution as `f3_3.png`.

### Exercise 3.4 – Conversational Agentic RAG with Memory

A fixed chain always retrieves with the raw user message, so a follow-up such as *"Which is the largest planet in it?"* retrieves poorly. In **agentic RAG** the LLM decides *when* to search and *what* to search for, so it can turn a follow-up into a self-contained search query. Conversation memory comes from a **checkpointer**: every call with the same `thread_id` continues the same conversation.

Implement:

1. A tool `search_wikipedia(query: str)` that searches `vectorstore` (top 3). The model must receive the text of the chunks (include the article titles), and **your code** must also get the retrieved `Document` objects back, so you can show sources. Hint: read `help(tool)` and look for `response_format`.
2. `conversation_agent`: an agent built with `create_agent` using `agent_llm`, your tool, a system prompt that forces searching before answering, and an in-memory checkpointer.
3. `ask(question, thread_id)`: sends one user message to the agent and returns `{"answer": str, "sources": list[Document]}`, where `sources` contains only the documents retrieved **during this turn**.

Note: tool calling needs a reasonably capable model; `llama3.2:1b`, for example, often produces malformed tool calls.

Reference: https://docs.langchain.com/oss/python/langchain/agents and https://docs.langchain.com/oss/python/langchain/short-term-memory

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

agent_llm = ChatOllama(model=LLM_MODEL, temperature=0)

# TODO 1: search_wikipedia tool


# TODO 2: conversation_agent
conversation_agent = None


def ask(question: str, thread_id: str = "lab_session") -> dict:
    # TODO 3
    pass


if RUN_DEMOS:
    for question in ["What is the Solar System?", "Which is the largest planet in it?"]:
        result = ask(question, thread_id="notebook_test")
        print(f"\nQuestion: {question}\nAnswer: {result['answer']}")
        for i, doc in enumerate(result["sources"], start=1):
            print(f"  {i}. {doc.metadata['title']} - {doc.metadata['url']}")
```

!!! note ":camera: Screenshot"
    Take a screenshot of both questions with their answers and sources, and upload it together with your solution as `f3_4a.png`.

Print the full message list stored for the `notebook_test` thread (message type, content, and the arguments of every tool call).

```python
# TODO
```

!!! note ":camera: Screenshot"
    Take a screenshot of the printed message list, including the tool calls and their arguments, and upload it together with your solution as `f3_4b.png`.

**Question:** Which search query did the agent use for the follow-up question, and why does this work better than retrieving with the raw question? Compare the answers with the retrieved sources: do they contain information that is **not** in the sources? What does that mean for trusting RAG answers, and how could you reduce it?

!!! question ":pencil: Written answer"
    Answer this question in `answers.md`, under the heading `## Exercise 3.4`.

## Part 4 – Interactive Q&A Interface

### Making your RAG functions importable

Both applications of this part import your functions from Parts 2–3 as a Python module called `RAG_Lab_Notebook`:

- If you worked in a **notebook** called `RAG_Lab_Notebook.ipynb`, convert it to a module (in a terminal):
    ```bash
    python -m nbconvert --to python RAG_Lab_Notebook.ipynb
    ```
    This creates `RAG_Lab_Notebook.py`. Re-export it whenever you change the notebook.

- If you worked in a **plain Python file**, it already is the module; just make sure it is called `RAG_Lab_Notebook.py`.

Keep `RAG_Lab_Notebook.py`, `rag_cli_exercise.py`, `rag_streamlit_exercise.py` and the `data/` folder in the same directory.

!!! tip "Performance"
    Importing `RAG_Lab_Notebook` re-runs every top-level statement, including building both indexes (one to a few minutes, depending on your hardware). The provided test code is skipped automatically on import (`RUN_DEMOS`); do the same with any extra test code you added. Optionally, persist the FAISS index (`faiss.write_index` / `faiss.read_index`) and the Chroma store (`persist_directory=`) and load them instead of rebuilding.

### Exercise 4.1 – Command-Line Interface

Create `rag_cli_exercise.py` from the skeleton below and complete the TODOs. It uses your **Part 2** `rag_query()` and manages the conversation history manually: before retrieval, a follow-up question such as *"Which is the largest planet in it?"* has to be rewritten into a standalone question with the help of the recent history. Run it with `python rag_cli_exercise.py`.

```python title="rag_cli_exercise.py"
"""
RAG Command-Line Interface - Exercise 4.1
=========================================
Conversational CLI on top of your Part 2 `rag_query()` with manually managed history.

Setup:
1. Export your notebook: python -m nbconvert --to python RAG_Lab_Notebook.ipynb
2. Complete the TODOs below
3. Run: python rag_cli_exercise.py

Problem to solve:
    rag_query() retrieves with the raw question, so a follow-up such as
    "Who developed it?" retrieves poorly. Before retrieval, use the LLM to rewrite
    the follow-up into a standalone question with the help of the recent history.
"""

from typing import Dict, List, Tuple

print("Loading the RAG system (building the indexes takes one to a few minutes)...")
from RAG_Lab_Notebook import query_llm, rag_query  # noqa: E402

HISTORY_WINDOW = 3  # number of previous exchanges used to rewrite a follow-up


def make_standalone_question(question: str, chat_history: List[Tuple[str, str]]) -> str:
    """
    Rewrite `question` into a self-contained question using the last HISTORY_WINDOW
    (question, answer) exchanges of `chat_history`. Use `query_llm()` for the rewrite.

    - Without history, return the question unchanged (no LLM call).
    - The LLM must only rewrite the question, not answer it; return just the rewritten question.
    """
    # TODO 1
    raise NotImplementedError


def query_rag_with_history(question: str, chat_history: List[Tuple[str, str]]) -> Dict:
    """
    Answer `question` with rag_query(), taking the conversation into account.

    Returns a dict with the keys:
        'answer'              - the generated answer
        'sources'             - the sources returned by rag_query()
        'standalone_question' - the question actually sent to rag_query()
    """
    # TODO 2
    raise NotImplementedError


# ============================================================================
# CLI
# ============================================================================

def format_sources(sources) -> str:
    """Format source documents for display."""
    if not sources:
        return ""
    lines = ["\n📚 Sources:"]
    for i, source in enumerate(sources[:3], 1):
        lines.append(f"{i}. {source.get('title', 'Unknown')}")
        if source.get("url"):
            lines.append(f"   {source['url']}")
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("RAG CLI - Conversational Q&A")
    print("=" * 70)
    print("\nAsk questions about the knowledge base. Follow-up questions are understood.")
    print("Type 'quit' or 'exit' to end.\n")

    # TODO 3: create the conversation history

    while True:
        try:
            print("-" * 70)
            question = input("\n💬 Your question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break
            if not question:
                continue

            print("\n🔍 Searching...")

            # TODO 4: answer the question with query_rag_with_history(), print the
            #         standalone question (if it differs from the original), the answer
            #         and the sources (format_sources), then update the history.

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

    print()


if __name__ == "__main__":
    main()
```

!!! note ":camera: Screenshot"
    Take a screenshot of a CLI session with at least one follow-up question, showing the rewritten standalone question, the answers and the sources, and upload it together with your solution as `f4_1.png`.

### Exercise 4.2 – Streamlit Web Interface

Create `rag_streamlit_exercise.py` from the skeleton below and complete the TODOs. It uses your **Part 3.4** `ask()` function (agent with memory); the requirements are listed in the docstring.

```python title="rag_streamlit_exercise.py"
"""
RAG Streamlit Web Interface - Exercise 4.2
==========================================
Conversational web app on top of your Part 3.4 `ask()` function (agent + checkpointer memory).

Setup:
1. Export your notebook: python -m nbconvert --to python RAG_Lab_Notebook.ipynb
2. Complete the TODOs below
3. Run: python -m streamlit run rag_streamlit_exercise.py

Requirements:
- Every browser session has its own conversation: two browser tabs must not share memory.
  (The agent remembers the conversation by thread id; Streamlit keeps per-session data in st.session_state.)
- A "New conversation" button in the sidebar starts a fresh conversation (new memory, empty chat).
- The full chat is displayed on every rerun, with the sources of each answer in an expander.
- A new question is shown immediately, answered with ask() behind a spinner, and stored in the chat.
"""

import uuid

import streamlit as st

st.set_page_config(page_title="Conversational RAG Q&A", page_icon="🤖", layout="wide")


@st.cache_resource(show_spinner="Loading the RAG system (the first start takes one to a few minutes)...")
def load_rag():
    """Import the exported notebook once per server process (provided)."""
    from RAG_Lab_Notebook import ask
    return ask


ask = load_rag()


def render_sources(sources) -> None:
    """Show LangChain Document sources in an expander (provided helper)."""
    if not sources:
        return
    with st.expander("📚 View Sources"):
        for i, doc in enumerate(sources[:3], 1):
            st.markdown(f"**{i}. {doc.metadata.get('title', 'Unknown')}**")
            if doc.metadata.get("url"):
                st.markdown(f"[🔗 Link]({doc.metadata['url']})")
            st.caption(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))


st.title("🤖 Conversational RAG Q&A")
st.markdown("Ask questions about Simple English Wikipedia. Follow-up questions are understood.")

# TODO 1: per-session state (conversation id and displayed messages)

# TODO 2: sidebar with the current conversation id and a "New conversation" button

# TODO 3: display the chat history

# TODO 4: handle a new question
```

Also create the following Streamlit configuration file next to the script. It switches off Streamlit's file watcher, which otherwise logs many harmless `ModuleNotFoundError: torchvision` messages while scanning the machine-learning libraries (if you run the app from another folder, pass `--server.fileWatcherType none` instead):

```toml title=".streamlit/config.toml"
# Streamlit's file watcher scans every imported module; with PyTorch/Transformers installed this
# logs many harmless "ModuleNotFoundError: torchvision" messages. The lab apps do not need auto-reload.
[server]
fileWatcherType = "none"
```

Run the application with:

```bash
python -m streamlit run rag_streamlit_exercise.py
```

!!! note ":camera: Screenshot"
    Take a screenshot of the web application showing a conversation with at least one follow-up question and one expanded sources panel, and upload it together with your solution as `f4_2a.png`.

!!! note ":camera: Screenshot"
    Take a screenshot of the same application opened in a second browser tab (or after clicking *New conversation*), showing that the new conversation does not remember the previous one, and upload it together with your solution as `f4_2b.png`.

## Giving feedback

Please share your opinion about this laboratory in `answers.md`, under the heading `## Feedback`. You can include:

- Your general impression of the laboratory.
- The things you liked or did not like.
- Whether you found the lab easy or difficult.
- Which exercises you enjoyed the most.
- What was necessary but not included in this laboratory.
- What was unnecessary but included in this laboratory.
