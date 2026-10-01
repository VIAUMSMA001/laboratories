---
authors: domonkosadam
---

# 03 - Integrating a Local LLM Into a Python Application

## Goal

The main goal of this laboratory is to learn how to run large language models (LLMs) locally with Ollama, how to call them from Python, and how to build LLM applications with LangChain and Streamlit.

The structure of this laboratory is as follows:

1. Installing Ollama and the required dependencies.
1. Ollama Direct API Usage [4 points in total]
    1. Simple API call [0.5 point]
    1. Streaming responses [0.5 point]
    1. Multi-turn conversation [1 point]
    1. Prompt engineering fundamentals [1 point]
    1. Parameter experimentation [1 point]
1. Choose your path [2 points in total]
    1. Option A – Local model comparison
        1. Multi-model speed comparison [1 point]
        1. Quantization impact analysis [1 point]
    1. Option B – Cloud model comparison
        1. Setting up Gemini API access [1 point]
        1. Local vs cloud comparison [1 point]
1. LangChain Integration [12 points in total]
    1. LangChain setup with Ollama (LCEL) [1 point]
    1. Conversation memory strategies [1 point]
    1. Structured output [2 points in total]
        1. Manual JSON prompting [0.5 point]
        1. List output parser [0.5 point]
        1. Pydantic output parser [0.5 point]
        1. Native structured output [0.5 point]
    1. Complex chains [6 points in total]
        1. Sequential chain [3 points]
        1. LLM-based router [3 points]
    1. Agents with custom tools [2 points]
1. Streamlit Chatbot [2 points]
1. Giving feedback [+1 point]

!!! info "Grading"
    In order to pass this laboratory, you must obtain at least 8 points out of 20.

!!! important "Screenshot requirement"
    At the end of **every** exercise (marked with a :camera: **Screenshot** note), you must take a screenshot of your working solution, including its output, and upload it together with your solution, using the exact filename given in that note (e.g. `f1_1.png`, `f2a_1.png`). Solutions missing the required screenshots, or using the wrong filename, will not be accepted.

    If an output does not fit on one screen, split it into several screenshots with a numeric suffix (e.g. `f1_4a_1.png`, `f1_4a_2.png`).

## Preparation

Don't forget to follow the assignment submission process described under [GitHub](../../information/github.md) while working on this laboratory.

!!! important "Reviewer"
    When creating the pull request for this laboratory, assign it to the `domonkosadam` GitHub user.

!!! tip "Jupyter notebook or plain Python files"
    You may solve the exercises either in a Jupyter notebook or in plain `.py` files, whichever you prefer. The code snippets below build on each other (e.g. the `ask()` helper of Exercise 1.4), so keep them in one notebook or module. Part 4 is always a separate Streamlit script (`chatbot_app.py`). Regardless of the format you choose, make sure that **everything** (code, `answers.md` and the required screenshots) is committed and pushed to your solution branch.

!!! important "Written answers"
    Some exercises contain questions (marked with :pencil: **Written answer**). Answer them in a file called `answers.md` in the root of your repository, under a heading with the exercise number (e.g. `## Exercise 1.2`). Your feedback at the end of the laboratory also goes into this file (under `## Feedback`). Make sure `answers.md` is committed and pushed together with your solution and screenshots.

## Setup

### Installing Ollama

Ollama runs large language models locally on your machine.

- **macOS:** download the app from <https://ollama.com/download> or run `brew install ollama`
- **Linux:** `curl -fsSL https://ollama.com/install.sh | sh`
- **Windows:** download the installer from <https://ollama.com/download>

Verify the installation and download the models used in this laboratory (in a terminal):

```bash
ollama --version
ollama pull llama3.2:3b
ollama pull llama3.2:1b
ollama list
```

### Python dependencies

!!! important "Mandatory"
    You need **Python 3.11 – 3.14** on Windows (x64), Linux (x64/arm64) or macOS. On *Windows on ARM*, install the x64 build of Python (it runs under emulation), because some packages have no native Windows-ARM builds.

    Create a virtual environment and install the pinned dependencies. The requirements file pins the exact version of **every** package, including all indirect dependencies, so everyone gets the same, tested environment. If `pip` reports success but installed almost nothing (check with `pip list`), your Python version / platform combination is not supported (see above).

    ```bash
    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r https://raw.githubusercontent.com/VIAUMSMA001/laboratories/main/docs/laboratories/03-local-llm-integration/requirements.txt
    ```

The main packages in the requirements file are:

```
google-genai==2.25.0
jupyterlab==4.6.4
langchain==1.4.2
langchain-core==1.6.5
langchain-ollama==1.1.0
langgraph==1.2.12
matplotlib==3.11.2
ollama==0.6.2
pandas==3.0.6
pydantic==2.13.5
streamlit==1.64.0
```

## Part 1 – Ollama Direct API Usage

In this section you use the Ollama Python library directly to generate text, stream responses, manage conversations and tune sampling parameters.

**Model for this section:** `llama3.2:3b` · **Library reference:** https://github.com/ollama/ollama-python

### Exercise 1.1 – Simple API Call

Send the prompt to the model with `ollama.chat()`, then print the prompt, the response text and the wall-clock generation time.

```python
import time
import ollama

MODEL = "llama3.2:3b"
prompt = "Explain quantum computing in one sentence."

# TODO
```

!!! note ":camera: Screenshot"
    Take a screenshot of the prompt, the response and the measured generation time, and upload it together with your solution as `f1_1.png`.

### Exercise 1.2 – Streaming Responses

Streaming displays output while it is generated, which greatly improves the user experience for long answers.

Stream a response and print the text as it arrives. When it has finished, print:

1. an **approximate** token count (words) and tokens/second based on your own wall-clock measurement,
2. the **exact** number of generated tokens and the generation speed that Ollama itself reports for the request.

Then answer the question below.

```python
prompt = "Write a short poem about artificial intelligence."

# TODO
```

!!! note ":camera: Screenshot"
    Take a screenshot of the streamed poem together with both the approximate and the exact token counts and speeds, and upload it together with your solution as `f1_2.png`.

**Question:** Why do the two token counts and the two speeds differ?

!!! question ":pencil: Written answer"
    Answer this question in `answers.md`, under the heading `## Exercise 1.2`.

### Exercise 1.3 – Multi-Turn Conversation

LLM APIs are stateless: to keep context, the conversation history has to be sent with every request.

For each test input, get a reply that takes the whole conversation into account and print each exchange. At the end, print how many messages the history contains.

```python
messages = [
    {"role": "system", "content": "You are a helpful AI assistant that specializes in explaining programming concepts."}
]

test_inputs = [
    "What is a variable in Python?",
    "Can you give me an example?",
    "What about lists?",
]

# TODO
```

!!! note ":camera: Screenshot"
    Take a screenshot of the printed conversation and the number of messages in the history, and upload it together with your solution as `f1_3.png`.

### Exercise 1.4 – Prompt Engineering Fundamentals

Prompt engineering is the craft of writing inputs that get better outputs from LLMs. The same model can produce very different results depending on how you ask.

#### Key Techniques

1. **Zero-shot vs few-shot:** ask directly, or show labeled examples before the actual task.
2. **Role prompting:** give the model a role or persona, e.g. *"You are an expert Python developer"*. Roles belong in the **system** message.
3. **Chain-of-thought (CoT):** ask the model to reason step by step before answering.
4. **Output format specification:** explicitly request the format you need (JSON, bullet points, a single word, …).
5. **Constraint setting:** specify length, tone and level of detail.

#### Tasks
1. Implement the helper `ask()`; you will use it for the rest of Part 1.
2. Only the zero-shot prompt is given. **Write the prompts** for Techniques 2–5 yourself.
3. In Technique 5 the reply must be parsed with `json.loads`. Handle the case when the model does not return valid JSON.
4. In the next code snippet, measure how much your best prompt improves over zero-shot on a small labeled set.

```python
import json

review = ("The product arrived late and the packaging was damaged, "
          "but the item itself works perfectly and exceeded my expectations.")


def ask(prompt: str, system: str | None = None, **options) -> str:
    """Send `prompt` (with an optional system message) to MODEL and return the reply text.

    Extra keyword arguments are passed to Ollama as model options, e.g. ask("Hi", temperature=0).
    """
    # TODO
    pass


print("TECHNIQUE 1: Zero-shot\n")
zero_shot_prompt = f"What is the sentiment of this review: {review}"
print(ask(zero_shot_prompt))

print("\nTECHNIQUE 2: Few-shot (at least three labeled examples: Positive / Negative / Mixed)\n")
# TODO

print("\nTECHNIQUE 3: Role prompting (role in the system message)\n")
# TODO

print("\nTECHNIQUE 4: Chain-of-thought\n")
# TODO

print("\nTECHNIQUE 5: Structured output\n")
# TODO: request JSON with the keys sentiment, confidence, positive_aspects, negative_aspects,
#       then parse it with json.loads and print the parsed dict (or a clear error message).
```

!!! note ":camera: Screenshot"
    Take a screenshot of the outputs of all five prompting techniques (including the parsed JSON of Technique 5), and upload it together with your solution as `f1_4a.png`.

#### Evaluate your prompt

1. Write `extract_label(reply)`, which returns the label (one of `LABELS`) that appears **first** in the reply (case-insensitive), or `None` if no label appears.
2. Write `optimized_prompt(review)`, combining the techniques above so that the model reliably answers with exactly one label.
3. Using `temperature=0`, evaluate the zero-shot prompt and your optimized prompt on `eval_reviews`. Print two metrics for each:
    - **accuracy**: the extracted label equals the expected label,
    - **format compliance**: the reply is *exactly* one of the labels (ignoring case, whitespace and a trailing period), so a program could use it without any parsing.

```python
eval_reviews = [
    ("Absolutely love it, works perfectly and shipping was fast.", "Positive"),
    ("Broke after two days and support never answered.", "Negative"),
    ("Great sound quality, but the battery barely lasts an hour.", "Mixed"),
    ("Oh fantastic, another charger that dies after a week. Just what I needed.", "Negative"),
    ("Setup was a pain, however once running it is excellent.", "Mixed"),
    ("I was sceptical at first, but it turned out to be exactly what I needed.", "Positive"),
    ("The screen is gorgeous. Shame it overheats within ten minutes.", "Mixed"),
    ("Five stars for the packaging, zero for the product inside.", "Negative"),
    ("Not the cheapest option, but you get what you pay for - flawless build.", "Positive"),
]
LABELS = ("Positive", "Negative", "Mixed")

# TODO
```

!!! note ":camera: Screenshot"
    Take a screenshot of the accuracy and format compliance printed for the zero-shot and for your optimized prompt, and upload it together with your solution as `f1_4b.png`.

### Exercise 1.5 – Parameter Experimentation

- **temperature:** controls randomness (0.0 = deterministic, higher = more random)
- **top_p:** nucleus sampling – only tokens within cumulative probability *p* are considered
- **top_k:** only the *k* most likely tokens are considered

For **every** value below, generate **3** completions of the same prompt (limit the length to ~60 tokens to keep it fast). For `top_p` and `top_k` use `temperature=1.0`. Collect the results in a pandas DataFrame with the columns `parameter`, `value`, `distinct_outputs` (the number of unique completions among the 3, i.e. 1–3) and `avg_words`, print it, and answer the question below.

```python
import pandas as pd

prompt = "Complete this sentence: The future of AI is..."
temperatures = [0.0, 0.5, 1.0, 2.0]
top_p_values = [0.1, 0.5, 0.9, 1.0]
top_k_values = [1, 5, 50, 100]

# TODO
```

!!! note ":camera: Screenshot"
    Take a screenshot of the resulting DataFrame of the parameter experiment, and upload it together with your solution as `f1_5.png`.

**Question:** How do temperature, top_p and top_k change the variety of the outputs? Which settings would you use for (a) code generation and (b) creative writing, and why?

!!! question ":pencil: Written answer"
    Answer this question in `answers.md`, under the heading `## Exercise 1.5`.


## Part 2 (Choose Your Path)

Complete **ONE** of the two options (or both, if you are interested):

**Option A: Local Model Comparison** (recommended with good hardware)

- **Requirements:** download 4 additional models (~19 GB in total)
- **Pros:** no API key needed, works offline, complete privacy

**Option B: Cloud Model Comparison** (recommended with limited hardware/bandwidth)

- **Requirements:** a free Google Gemini API key (no credit card)
- **Pros:** no large downloads, access to powerful models

!!! warning "Before submitting"
    Remove the code of the option you did not complete. Your solution must run from top to bottom without errors (Jupyter's *Run All* stops at the first error, so an unfinished option would stop everything after it, including Part 3).


## Part 2 – Option A: Local Model Comparison

**Models:** `qwen2.5:1.5b`, `llama3.2:3b`, `mistral:7b`, `llama3.1:8b-instruct-q4_0`, `llama3.1:8b-instruct-q8_0`

```bash
ollama pull qwen2.5:1.5b
ollama pull mistral:7b
ollama pull llama3.1:8b-instruct-q4_0
ollama pull llama3.1:8b-instruct-q8_0
```

### Exercise 2.1 – Multi-Model Speed Comparison

Run every prompt on every model and collect the results in a DataFrame `speed_df` with (at least) the columns `model`, `complexity`, `wall_seconds`, `load_seconds`, `output_tokens` and `tokens_per_sec`. Use the timing information Ollama returns with each response, not word counts.

Then draw a grouped bar chart (matplotlib) of `tokens_per_sec` per model and complexity, and answer the question.

```python
import time
import matplotlib.pyplot as plt
import ollama
import pandas as pd

test_prompts = {
    "simple": "What is 2+2?",
    "medium": "Write a Python function to reverse a string.",
    "complex": "Explain the concept of recursion with a detailed example in Python.",
}
models = ["qwen2.5:1.5b", "llama3.2:3b", "mistral:7b"]

# TODO
```

!!! note ":camera: Screenshot"
    Take a screenshot of the `speed_df` DataFrame and the bar chart, and upload it together with your solution as `f2a_1.png`.

**Question:** How does speed scale with model size? Why is the first request to a model often much slower than the following ones?

!!! question ":pencil: Written answer"
    Answer this question in `answers.md`, under the heading `## Exercise 2.1 (Option A)`.

### Exercise 2.2 – Quantization Impact Analysis

Quantization reduces model size and increases speed by storing weights with lower precision (Q4 = 4-bit, Q8 = 8-bit).

Compare `llama3.1:8b-instruct-q4_0` and `llama3.1:8b-instruct-q8_0` on the prompt below. Build a DataFrame with the size on disk (GB, from Ollama's model list), the quantization level reported by Ollama, tokens/second, and the response. Read both responses and answer the question.

```python
test_prompt = "Write a professional email requesting a meeting to discuss a project proposal."
quant_models = ["llama3.1:8b-instruct-q4_0", "llama3.1:8b-instruct-q8_0"]

# TODO
```

!!! note ":camera: Screenshot"
    Take a screenshot of the quantization DataFrame and the beginning of both responses, and upload it together with your solution as `f2a_2.png`.

**Question:** Did you notice a quality difference between Q4 and Q8? Which one would you deploy on a laptop with 16 GB RAM, and why?

!!! question ":pencil: Written answer"
    Answer this question in `answers.md`, under the heading `## Exercise 2.2 (Option A)`.


## Part 2 – Option B: Cloud Model Comparison

**Requirements:** a free Google Gemini API key and an internet connection.

In this section you call Google's Gemini API through the official **`google-genai`** SDK (the older `google-generativeai` package is no longer supported) and compare it with your local model.

### Local vs Cloud LLMs

| | Local LLMs (Ollama) | Cloud LLMs (Gemini, GPT, Claude) |
|---|---|---|
| Where the model runs | your hardware (CPU/GPU) | the provider's servers |
| Internet | not needed after download | required |
| Data privacy | data never leaves your machine | data is sent to the provider |
| Infrastructure | you manage it | the provider manages it |

### Exercise 2.1 – Setting Up Gemini API Access

1. Visit https://aistudio.google.com/app/apikey and sign in with your Google account.
2. Click **Create API key** and copy the key.
3. Make the key available as the environment variable `GEMINI_API_KEY` **before starting Jupyter or your Python script** (e.g. `export GEMINI_API_KEY=...` on macOS/Linux, `setx GEMINI_API_KEY ...` on Windows, then open a new terminal).

The free tier is rate limited; check the current limits in AI Studio. Never commit or submit your API key.

**Model:** use `gemini-3.8-flash` (the current stable Flash model, available on the free tier; checked in September 2026). Gemini models are retired regularly: older ids such as `gemini-1.5-flash` or `gemini-2.0-flash` no longer exist, and `gemini-2.5-flash` is only available to accounts that used it before. If a call fails with **404 / NOT_FOUND**, the model id is not available to your key: pick a current Flash model from the list you print in step 2 (current models: https://ai.google.dev/gemini-api/docs/models).

**Task:** Using the SDK documentation (https://googleapis.github.io/python-genai/):

1. Create a client that reads the key from the environment variable.
2. Print the names of the models available to your key that support content generation, and check that `GEMINI_MODEL` is among them.
3. Test the connection.

```python
import os
from google import genai

GEMINI_MODEL = "gemini-3.8-flash"

# TODO
client = None

response = client.models.generate_content(model=GEMINI_MODEL, contents="Say 'Gemini is connected!' and nothing else.")
print(response.text)
```

!!! note ":camera: Screenshot"
    Take a screenshot of the printed list of available models and the successful connection test, and upload it together with your solution as `f2b_1.png`. **Make sure that your API key is not visible on the screenshot!**

### Exercise 2.2 – Local vs Cloud Comparison

Run every test prompt on the local model (`llama3.2:3b`) and on your Gemini model. Collect a DataFrame with the columns `category`, `local_seconds`, `cloud_seconds`, `local_tokens`, `cloud_tokens`, `cloud_thinking_tokens` (use the token counts reported by each API; Gemini Flash models may "think" before answering, and those hidden tokens are reported separately, count towards the latency and are billed as output), `local_response`, `cloud_response`, and two columns `local_quality` / `cloud_quality` in which **you** rate each answer from 1 to 5 against the `expected` value.

Then answer the question below.

```python
import time
import ollama
import pandas as pd

MODEL = "llama3.2:3b"

test_prompts = [
    {"category": "Factual", "prompt": "What is the capital of Australia and what is its population?", "expected": "Canberra, ~460,000"},
    {"category": "Creative", "prompt": "Write a haiku about artificial intelligence.", "expected": "3 lines, 5-7-5 syllables"},
    {"category": "Code", "prompt": "Write a Python function that checks if a string is a palindrome. Include comments.", "expected": "Correct function with comments"},
    {"category": "Reasoning", "prompt": "Two trains are 800 miles apart and travel towards each other at 60 mph and 80 mph. When do they meet?", "expected": "after ~5 hours 43 minutes"},
    {"category": "Summarization", "prompt": "Summarize this in 2 sentences: Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.", "expected": "2 clear sentences"},
]

# TODO
```

!!! note ":camera: Screenshot"
    Take a screenshot of the comparison DataFrame including your quality ratings, and upload it together with your solution as `f2b_2.png`.

**Question:** For which kinds of tasks is the local model good enough? When would you pay for a cloud API instead? Consider speed, quality, privacy and cost.

!!! question ":pencil: Written answer"
    Answer this question in `answers.md`, under the heading `## Exercise 2.2 (Option B)`.

## Part 3 – LangChain Integration

LangChain is a framework for building applications with language models. This lab uses **LangChain 1.x**:

- **LCEL** (LangChain Expression Language): prompts, models and parsers are *runnables* that you compose with `|`
- **Agents** (`create_agent`): a model that can call tools, with state and memory handled by **LangGraph** checkpointers and customized with **middleware**

Many tutorials online still use pre-1.0 APIs (`LLMChain`, `ConversationChain`, `ConversationBufferMemory`, `langchain.hub`, `AgentExecutor`, …). These were removed from `langchain` 1.x and **must not be used** in this lab (neither through the legacy `langchain_classic` package).

**Model for this section:** `llama3.2:3b` · **Documentation:** https://docs.langchain.com/oss/python/langchain/overview

### Exercise 3.1 – LangChain Setup with Ollama (LCEL)

1. Create a LangChain chat model for `llama3.2:3b`.
2. Build `chain` with LCEL from a prompt template with the text `You are a {role}. {task}`, the model and an output parser, so that `chain.invoke()` returns a plain **string**.
3. Stream the output of the same chain token by token.
4. Run the chain on two different inputs **in parallel** with a single call.

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# TODO: 1-2
llm = None
chain = None

result = chain.invoke({"role": "helpful coding tutor", "task": "Explain what a Python decorator is in simple terms."})
print(f"Is a string: {isinstance(result, str)}\n{result}")

# TODO: 3 - stream {"role": "poet", "task": "Write a two-line poem about Python."}

# TODO: 4 - two inputs in a single call
```

!!! note ":camera: Screenshot"
    Take a screenshot of the type check, the invoked answer, the streamed output and the two batched answers, and upload it together with your solution as `f3_1.png`.

### Exercise 3.2 – Conversation Memory Strategies

Common strategies for conversation memory:

- **Buffer:** keep every message
- **Window:** keep only the last *k* messages
- **Summary:** replace older messages with an LLM-written summary

In LangChain 1.x, conversation memory is the **state of an agent**. A **checkpointer** stores it per `thread_id`, and **middleware** can modify the messages before each model call. Read https://docs.langchain.com/oss/python/langchain/short-term-memory.

**Tasks:**

1. Write `run_conversation(agent, thread_id)`, which sends the test messages one by one in the same thread and prints every exchange.
2. Build three agents (no tools needed), each with its own in-memory checkpointer:
    - `buffer_agent`: keeps all messages,
    - `window_agent`: your own `@before_model` middleware that keeps only the **3 most recent** messages,
    - `summary_agent`: the built-in summarization middleware, summarizing once there are more than 3 messages and keeping the most recent message.
3. Run the conversation with all three agents. Then print the message list stored in the summary agent's thread (`summary_agent.get_state(config)`) and read the summary it contains.

```python
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, SummarizationMiddleware
from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

llm = ChatOllama(model="llama3.2:3b", temperature=0)

test_messages = [
    "My name is Alice and I'm learning Python.",
    "What are some good resources for beginners? Answer briefly.",
    "Can you remind me what my name is?",
]

# TODO
```

!!! note ":camera: Screenshot"
    Take a screenshot of the three conversations (buffer, window, summary) and the messages stored in the summary agent's thread, and upload it together with your solution as `f3_2.png`.

**Question:** Which agents remembered the name, and why? What did the summary contain, and what does that tell you about summary memory with a small model? What are the trade-offs of the three strategies for long conversations (context length, cost, information loss)?

!!! question ":pencil: Written answer"
    Answer this question in `answers.md`, under the heading `## Exercise 3.2`.

### Exercise 3.3 – Structured Output

Applications often need structured data (JSON, lists, typed objects) instead of free text. Without special techniques you get inconsistent formats, extra explanatory text and parsing errors.

**Example problem:**
```
You ask:  "Extract the person's name, age and city from this text"
LLM says: "Based on the text, the person's name is John Smith, he is 35 years old and lives in New York City."
You want: {"name": "John Smith", "age": 35, "city": "New York City"}
```

Techniques you compare in this exercise:

1. **Manual prompting** for JSON and parsing it yourself
2. **Output parsers** (`langchain_core.output_parsers`): the parser adds format instructions to the prompt and parses the reply
3. **Pydantic output parser**: parses into a validated Pydantic model
4. **Native structured output** (`with_structured_output`): the schema is passed to the model/runtime itself (Ollama constrains the generation to the JSON schema)

```python
import json
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser, PydanticOutputParser
from langchain_core.exceptions import OutputParserException

sample_text = """
Customer Review:
I recently purchased the TechPro X15 laptop from your store.
The customer service representative, Sarah Johnson, was extremely helpful.
I bought it on January 15, 2024 for $1,299.99.
Overall rating: 4.5 out of 5 stars.
Main issues: Battery life could be better, keyboard is a bit loud.
Positives: Fast processor, beautiful display, lightweight design.
"""

llm = ChatOllama(model="llama3.2:3b", temperature=0)

# For the reliability experiments in 3.3.3 and 3.3.4: sampling (temperature > 0) makes repeated runs differ
RELIABILITY_MODELS = ["llama3.2:3b", "llama3.2:1b"]
RUNS = 5
```

#### Exercise 3.3.1 – Manual JSON Prompting

Write a prompt that makes the model return **only** JSON with the fields `product_name` (string), `employee_name` (string), `purchase_date` (string), `price` (number), `rating` (number), `issues` (list of strings) and `positives` (list of strings). Call the model and parse the reply with `json.loads`. Your parsing must also cope with replies wrapped in Markdown code fences. Print the raw reply and the parsed dict (or the error).

```python
# TODO
```

!!! note ":camera: Screenshot"
    Take a screenshot of the raw reply and the parsed JSON (or the parsing error), and upload it together with your solution as `f3_3_1.png`.

#### Exercise 3.3.2 – List Output Parser

Build `list_chain`: a prompt (including the parser's format instructions) → model → `CommaSeparatedListOutputParser`, which returns a Python **list** of the positive aspects of the review.

```python
# TODO
list_chain = None

parsed_list = list_chain.invoke({"review": sample_text})
print(type(parsed_list).__name__, parsed_list)
```

!!! note ":camera: Screenshot"
    Take a screenshot of the parsed Python list, and upload it together with your solution as `f3_3_2.png`.

#### Exercise 3.3.3 – Pydantic Output Parser

1. Define a Pydantic model `CustomerReview` with the fields of 3.3.1 **plus** `sentiment`, which may only be `"positive"`, `"negative"` or `"mixed"`. Give every field a description.
2. Build a chain *prompt with format instructions → model → `PydanticOutputParser`*.
3. Reliability experiment: for each model in `RELIABILITY_MODELS`, use `ChatOllama(model=..., temperature=0.8)` and run the chain `RUNS` times. Count how many runs produced a valid `CustomerReview` (failed runs raise `OutputParserException`) and store the counts in a dict `parser_successes = {model: count}`.

```python
# TODO
```

!!! note ":camera: Screenshot"
    Take a screenshot of the individual runs and the success counts per model, and upload it together with your solution as `f3_3_3.png`.

#### Exercise 3.3.4 – Native Structured Output

Repeat the reliability experiment of 3.3.3 with the chat model's `with_structured_output()` and the **same** `CustomerReview` model (same models, temperature and number of runs). Store the counts in `native_successes = {model: count}` and print both dicts.

```python
# TODO
```

!!! note ":camera: Screenshot"
    Take a screenshot of the individual runs and both success-count dictionaries, and upload it together with your solution as `f3_3_4.png`.

**Question:** Which technique was more reliable, and how did the model size change the result? What does the model actually receive in 3.3.3 compared to 3.3.4, and why does that make a difference?

!!! question ":pencil: Written answer"
    Answer this question in `answers.md`, under the heading `## Exercise 3.3`.

### Exercise 3.4 – Complex Chains

- **Sequential chains:** execute steps in order, passing the output of one step to the next
- **Routers:** send inputs to different chains depending on their content

#### Exercise 3.4.1 – Sequential Chain

Build `sequential_chain`: **outline → expand → summarize**.

- Write all three prompts yourself (3-point outline of an essay about `{topic}`; expand the outline into short paragraphs; summarize the expanded text in 2 sentences).
- `sequential_chain.invoke({"topic": ...})` must return a dict that contains the original `topic` **and every intermediate result** (`outline`, `expanded`, `summary`), all as plain strings.
- It must be a single LCEL runnable (hint: look at `RunnablePassthrough.assign`).

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOllama(model="llama3.2:3b", temperature=0.7)

# TODO
sequential_chain = None

result = sequential_chain.invoke({"topic": "The impact of AI on education"})
for key in ["outline", "expanded", "summary"]:
    print(f"===== {key} ({len(result[key].split())} words) =====\n{result[key]}\n")
```

!!! note ":camera: Screenshot"
    Take a screenshot of the outline, the expanded text and the summary with their word counts, and upload it together with your solution as `f3_4_1.png`.

#### Exercise 3.4.2 – LLM-Based Router

A keyword-based router (e.g. *"contains 'calculate' → math"*) is brittle: *"How do I compute the average of a list in Python?"* contains a math word but is a coding question.

Build a router in which **the LLM decides** the route:

1. A classifier that returns exactly one of `"math"`, `"code"`, `"general"`. Use structured output so the route is guaranteed to be valid.
2. Three specialized chains (math tutor step by step; programming expert with code; concise general answer). Write the prompts yourself.
3. `router_chain`: a single runnable that takes `{"question": ...}` and returns `{"route": ..., "answer": ...}`.
4. Run the test questions, print the route next to the **expected** route, and print the routing accuracy.

A 3B model will not route every question correctly; the last test question is deliberately hard. You are graded on the implementation and on your analysis below, not on reaching 100%.

```python
from typing import Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

test_questions = [  # (question, expected route)
    ("Calculate the area of a circle with radius 5", "math"),
    ("Write a Python function to sort a list", "code"),
    ("What is the capital of France?", "general"),
    ("How do I compute the average of a list in Python?", "code"),
    ("Why is the sky blue?", "general"),
    ("If I save 50 euros a month, how long until I have 1000 euros?", "math"),
]

# TODO
router_chain = None
```

!!! note ":camera: Screenshot"
    Take a screenshot of the route, the expected route and the (shortened) answer of every test question, and the routing accuracy, and upload it together with your solution as `f3_4_2.png`.

**Question:** Which questions were misrouted, and why do you think the model got them wrong? Name two ways to make the routing more reliable.

!!! question ":pencil: Written answer"
    Answer this question in `answers.md`, under the heading `## Exercise 3.4.2`.

### Exercise 3.5 – Agents with Custom Tools

Agents decide dynamically which tools to call. In LangChain 1.x you create one with `create_agent()` and turn Python functions into tools with the `@tool` decorator. The **docstring** and the **type hints** become the tool description the model sees. `llama3.2:3b` supports native tool calling.

**Tasks:**

1. Implement the tools:
    - `calculator(expression)`: evaluates arithmetic expressions. **Do not** pass arbitrary input to `eval()`; e.g. evaluate a parsed `ast` tree with only arithmetic operators allowed. It must also return an error message instead of hanging on inputs such as `9**9**9` (the model decides what the tool receives!).
    - `get_current_time()`: current date and time,
    - `reverse_string(text)`,
    - **one more tool of your own choice**, plus a test query for it.
2. Call your `calculator` tool directly with the expression `9**9**9` and print the result (it must be an error message, returned quickly).
3. Create `agent` with a system prompt.
4. For every query print the **names of the tools that were called** and the final answer.

```python
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool

llm = ChatOllama(model="llama3.2:3b", temperature=0)

test_queries = [
    "What is 15% of 200?",
    "What is the current time?",
    "Reverse the string 'Hello World'",
    "What is today's date, and what is 365 divided by 7?",
    # TODO: add a query for your own tool
]

# TODO
```

!!! note ":camera: Screenshot"
    Take a screenshot of the result of calling your calculator tool directly with `9**9**9`, and the tools called and the final answer for every test query (including the query for your own tool), and upload it together with your solution as `f3_5.png`.

## Part 4 – Streamlit Chatbot

**Goal:** build a working chatbot application with Streamlit on top of Ollama. Create `chatbot_app.py` from the skeleton below. It contains the page layout and the sidebar controls; the chat logic is up to you:

1. Offer the **installed** Ollama chat models in the model selector (embedding models such as `nomic-embed-text` must not appear).
2. **Stream** the assistant's answer token by token into the page, using the temperature and max-token settings from the sidebar, and show below the answer how long the generation took.
3. Keep and display the conversation history across reruns.
4. The system prompt from the sidebar must take effect even if it is changed in the middle of a conversation.

Run it with:
```bash
python -m streamlit run chatbot_app.py
```

```python title="chatbot_app.py"
"""
Local LLM Chatbot - Part 4
==========================
Run: python -m streamlit run chatbot_app.py

Requirements (see the laboratory description):
1. Offer the installed Ollama chat models in the model selector (no embedding models).
2. Stream the assistant's answer token by token, using the temperature and max-token settings,
   and show below the answer how long the generation took.
3. Keep and display the conversation history across reruns.
4. A changed system prompt must take effect even in the middle of a conversation.
"""

from datetime import datetime
import time

import ollama
import streamlit as st

# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="Local LLM Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== HELPER FUNCTIONS ====================


def list_chat_models() -> list[str]:
    """Return the names of the installed Ollama chat models (embedding models excluded)."""
    # TODO 1
    return []


def stream_response(model_name: str, messages: list, temperature: float, max_tokens: int):
    """Generator that yields the assistant's answer piece by piece."""
    # TODO 2
    yield ""


# ==================== SIDEBAR CONFIGURATION ====================

st.sidebar.header("⚙️ Configuration")

available_models = list_chat_models()
if not available_models:
    st.error("No chat models found. Is Ollama running, and did you pull a model (ollama pull llama3.2:3b)?")
    st.stop()

DEFAULT_MODEL = "llama3.2:3b"
selected_model = st.sidebar.selectbox(
    "Select Model",
    available_models,
    index=available_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in available_models else 0,
    help="Choose which local model to use",
)

temperature = st.sidebar.slider(
    "Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1,
    help="Controls randomness: 0.0 = focused, 2.0 = creative",
)

max_tokens = st.sidebar.slider(
    "Max Tokens", min_value=50, max_value=2000, value=500, step=50,
    help="Maximum length of response",
)

with st.sidebar.expander("🎭 System Prompt (Advanced)", expanded=False):
    system_prompt = st.text_area(
        "Custom System Prompt", value="You are a helpful AI assistant.", height=100,
        help="Define the chatbot's personality",
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### Current Settings")
st.sidebar.markdown(f"**Model:** `{selected_model}`")
st.sidebar.markdown(f"**Temperature:** `{temperature}`")
st.sidebar.markdown(f"**Max Tokens:** `{max_tokens}`")

# ==================== MAIN APPLICATION ====================

st.title("🤖 Local LLM Chatbot")

# TODO 3: conversation state and history display

# TODO 4: handle new user input (show it, stream the answer, show the generation time, store both messages)

# ==================== SIDEBAR CONTROLS ====================

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Controls")

if st.sidebar.button("🗑️ Clear Chat", width="stretch"):
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("💾 Export Chat", width="stretch"):
    export_text = f"Chat Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += f"Model: {selected_model}\n"
    export_text += "=" * 50 + "\n\n"
    for msg in st.session_state.get("messages", []):
        if msg["role"] != "system":
            export_text += f"{msg['role'].upper()}: {msg['content']}\n\n"

    st.sidebar.download_button(
        "📥 Download",
        export_text,
        f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        width="stretch",
    )
```

!!! note ":camera: Screenshot"
    Take a screenshot of the running chatbot in the browser, showing a conversation of at least two turns in which the model remembers something from an earlier turn, the generation time of the latest answer, and the sidebar (selected model and settings), and upload it together with your solution as `f4.png`.

## Giving feedback

Please share your opinion about this laboratory in `answers.md`, under the heading `## Feedback`. You can include:

- Your general impression of the laboratory.
- The things you liked or did not like.
- Whether you found the lab easy or difficult.
- Which exercises you enjoyed the most.
- What was necessary but not included in this laboratory.
- What was unnecessary but included in this laboratory.
