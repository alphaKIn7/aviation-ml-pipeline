# Project Context

## About the Developer

I am a complete beginner when it comes to building production-grade projects. I am learning everything as I go — data engineering, software architecture, Python best practices, and how real-world pipelines are structured and deployed.

## How Claude Should Help Me

### When writing code
- Write clean, production-quality code, but **explain every decision** as if I have never seen it before.
- Do not assume I know why something is done a certain way. Tell me the "why" behind each choice.
- If there is a simpler way and a more production-ready way, show the production-ready way but explain what makes it better.

### When explaining code
- Explain like I am a complete beginner — no jargon without definition.
- Use analogies and real-world comparisons to make abstract concepts concrete.
- Break things down step by step. One concept at a time.
- If a line of code does something non-obvious, explain that specific line.

### Bigger picture
- When I add a new piece (a function, a module, a config file), briefly explain where it fits in the overall project and why it matters at production scale.
- Help me build mental models, not just copy-paste solutions. I want to understand enough to build the next thing myself.
- If I am doing something in a way that would cause problems in production, tell me clearly and explain why it would break.

## Goal

By the end of this project I want to be able to design and build a production-grade data pipeline on my own, with as little external help as possible. Every explanation Claude gives me should move me closer to that independence.

---

## Project: Aviation ML Pipeline

This is an end-to-end ML pipeline project focused on aviation data analytics (e.g. flight delay prediction). The folder structure and skeleton files were generated with Claude Code as a starting point. All source files currently contain only comments/docstrings — no implementation yet.

### Python background
- Comfortable with pandas data manipulation
- No prior experience building pipelines, ML systems, or production-grade Python projects
- Learning ML concepts (preprocessing, feature engineering, model training, evaluation) as part of this project

### What has been built (skeleton only)
- `src/utils/logger.py` — implemented (centralized logging)
- `src/utils/io_handler.py` — implemented (file I/O: YAML, CSV, Parquet, Pickle)
- All other `src/` files — comments only, no implementation yet
- `configs/train_config.yaml` and `configs/inference_config.yaml` — fully written
- `walkthrough.md` — full architecture overview already documented

### Implementation order (agreed roadmap)
1. `src/utils/` — logger, config, io_handler (foundation) ← in progress
2. `src/data/` — ingestion → validation → preprocessing
3. `notebooks/01_eda.py` — explore actual aviation dataset
4. `src/features/` — feature engineering based on EDA insights
5. `src/models/` — trainer → evaluator → predictor
6. `src/pipelines/` — wire everything together
7. `tests/` — write tests alongside each module
8. `scripts/`, `docker/`, CI — finalize for production
