---
authors: domonkosadam
---

# Homework

## Goal

The homework is your chance to build something you care about with the tools of the course. You design and implement an end-to-end system that uses AI technologies, evaluate how well it works, and present it in a short video. The topic is up to you.

Two things are required of every project. It must be built around at least one framework or tool that we didn't use in the labs but that is closely related to what we taught (LangGraph on top of LangChain, vLLM next to Ollama, MLflow next to a plain training loop). And it must be evaluated, not only demonstrated.

!!! info "Grading"
    The homework is worth 100 points and is mandatory for the final grade: 20 points for the specification and 80 for the final submission. You pass with at least 40 points, provided the submission meets the [eligibility conditions](#eligibility-conditions).

## Deadlines

| Milestone | Deadline |
| --- | --- |
| Homework published | 2026.11.02 |
| Specification submitted (Moodle) | 2026.11.15 |
| Final submission (GitHub pull request and video) | 2026.12.13 |

## Choosing a project

Finding a good problem is part of the assignment. A hobby, something from your job or another course, a public API you've wanted to try.

### The new framework

Your system must use at least two technologies from the lectures or labs and at least one new framework. A framework is new if it isn't part of the laboratory stack:

- NumPy, Pandas, Matplotlib, scikit-learn and plain PyTorch
- Hugging Face `transformers` and `sentence-transformers` for inference and embeddings
- LangChain (LCEL, output parsers, memory, `create_agent`, checkpointers), Chroma and FAISS
- Ollama and the Gemini API
- FastAPI, Pydantic, Docker, SQLite and Streamlit

A few directions, to show what "related" can mean. Other ideas are welcome, and since tools change quickly, check that your choice is still maintained.

| Area | Possible frameworks and tools |
| --- | --- |
| Agents and orchestration | LangGraph, CrewAI, AutoGen, PydanticAI, MCP servers, Langfuse or LangSmith tracing |
| Retrieval | LlamaIndex, Haystack, Weaviate, Qdrant, LanceDB, hybrid search, rerankers, graph-based retrieval, DSPy, Ragas |
| Language models | PEFT/LoRA, TRL, Unsloth, quantization with bitsandbytes or AWQ |
| Multimodal | Whisper, CLIP, Docling, vision-language models, text-to-speech |
| Serving | vLLM, llama.cpp server, Triton, BentoML, Ray Serve, ONNX Runtime |
| MLOps | MLflow, Weights & Biases, DVC, ZenML, Kubeflow, Evidently, Optuna |
| Edge and mobile | TF Lite, CoreML, ExecuTorch, MediaPipe |
| Developer tools | Code review or repository Q&A agents with MCP tools |

## Specification [20 points]

Before you start implementing, submit a one-to-two page specification using the [Markdown template](https://raw.githubusercontent.com/VIAUMSMA001/laboratories/main/docs/homework/specification-template.md).

The specification covers the idea, the data sources, the planned pipeline with a diagram, the technologies, and the evaluation plan.

| Criterion | Points |
| --- | --- |
| Idea and objectives are clear, and the scope is realistic | 5 |
| Data sources are named, accessible and appropriate, with license and size | 4 |
| Architecture is coherent, shown in a diagram, and uses at least 2 course technologies | 5 |
| The new framework is essential and meaningful | 4 |
| Evaluation plan defines metrics, test data and a baseline | 2 |

## Final submission

### Eligibility conditions

A submission is not accepted, whatever points it would otherwise get, if any of these fails:

- [ ] The specification was approved.
- [ ] The system runs by following your README.
- [ ] It isn't a laboratory rerun with new data.
- [ ] A new framework is used meaningfully, together with at least 2 course technologies.
- [ ] The video shows the real system working.
- [ ] The work is yours, with external code, models and data attributed.
- [ ] No API keys, tokens or passwords are committed.

### The system

The system needs a documented entry point (CLI, web UI, API, mobile app or scheduled pipeline), should cope with a few error cases such as empty input or an API failure, and should read configuration and keys from environment variables, with an `.env.example` in the repository. Dependencies must be pinned so it installs on a clean machine. If it relies on a paid API, add a free path (a local model or a free tier) or cached outputs so reviewers can try it. Docker is welcome but optional.

### Data

Finding data is part of the task. Public datasets, public APIs, scraped data (respect the terms of service and `robots.txt`) and your own data are all fine, as long as nothing personal or confidential is in it. LLM-generated data may extend a dataset but can't replace the evaluation data.

Document each source in `DATA.md` with its URL, license, size and what you did with it. Don't commit anything over about 50 MB. Provide a download script and a small sample instead, so the system can be tried quickly.

Good places to start are [Google Dataset Search](https://datasetsearch.research.google.com/), [Hugging Face Datasets](https://huggingface.co/datasets), [Kaggle](https://www.kaggle.com/), [OpenML](https://www.openml.org/), the [UCI repository](https://archive.ics.uci.edu/), [data.world](https://data.world/), [Eurostat](https://ec.europa.eu/eurostat), the [London Datastore](https://data.london.gov.uk/), [OpenFlights](https://openflights.org/), [Common Crawl](https://commoncrawl.org/), [ImageNet](https://www.image-net.org/) and [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/). For APIs, see [public-apis](https://github.com/public-apis/public-apis) and [awesome-api](https://github.com/Kikobeats/awesome-api). BKK's open data for Hungarian public transport is also a good source.

### Evaluation

A demo shows that something works once. `EVALUATION.md` has to show how well it works and where it fails. It needs a test set of at least 30 items that you didn't tune on, a metric that fits the task (F1, retrieval hit rate, answer correctness, WER, task success), a comparison against a baseline or an ablation, latency (median and p95) and cost where relevant, and at least three concrete failure cases with your explanation. If an LLM or a framework like Ragas judges the answers, check at least 10 of its verdicts by hand and report how often you agree.

### Video

Record a screen capture with voice-over, 3 to 5 minutes long, in English. It should show the real system running, not slides. A structure that works is a short problem statement, the architecture and the new framework, a live demo with one edge or failure case, the main evaluation numbers, and a brief reflection on what worked and what didn't. Keep the text readable, keep keys and personal data off the screen, and make sure the instructors can open the video without registering or requesting access. Put the link in `VIDEO.md`.

### Repository

Submit through GitHub as described under [GitHub](../information/github.md): work on a `solution` branch in your own repository, open a pull request named `homework` against the main branch, assign it to the instructor listed in Moodle, and leave it open.

```text
README.md        what it is, how to install and run it, results in short
VIDEO.md         link to the demo video
DATA.md          sources, licenses, preparation steps
EVALUATION.md    method, results, baseline, failure analysis
AI_USAGE.md      AI tools used
docs/            architecture diagram
src/, tests/     code and a few smoke tests
requirements.txt pinned dependencies
.env.example     required variable names, no values
```

## Grading of the final submission [80 points]

| Criterion | Points | Full marks |
| --- | --- | --- |
| Technical depth and the new framework | 20 | The framework is central to the design and its main abstractions are used well. The architecture is non-trivial, the choices are justified, and the course technologies work together as one system. |
| Working system and engineering quality | 15 | Reproducible from the README on a clean machine, with a clear structure, separated configuration, sensible error handling and a few meaningful tests. |
| Data | 8 | The data fits the problem and took real effort to find or collect. Sources, licenses and preparation are documented, and quality issues are handled. |
| Evaluation and analysis | 15 | Separate test set, fitting metrics, an informative baseline or ablation, and a failure analysis that shows real understanding. |
| Demo video | 10 | Within the time limit, shows a real run and an edge case, and makes clear that you understand your own design. |
| Documentation and repository | 7 | README, `DATA.md`, `EVALUATION.md` and `AI_USAGE.md` are complete and match the code. The diagram is clear and the history is meaningful. |
| Ambition | 5 | Something beyond the requirements that fits the project and is documented and evaluated, such as a second new framework, a deployment with monitoring, a runtime comparison, or a mobile build. |

Partial credit is given in proportion. A system that runs but only follows a tutorial lands in the lower half of the technical depth range, and an evaluation with a handful of examples and no baseline earns a few points at most.

## FAQ

??? question "Can I use Streamlit, FastAPI, Ollama or LangChain?"
    Yes. They just don't count as the new framework.

??? question "Can I build on my lab solutions?"
    Yes, if what you add is the project. Say so in the specification.

??? question "My new framework turned out to be unsuitable."
    Tell the instructors early and send an updated specification. Switching in the last week without agreement is risky.

??? question "Can I work on something related to my thesis or job?"
    Yes, if the code and data can be shared with the instructors and the work is new for this course.
