# Contributing Guide

The Math Induction Head project is being rebuilt from scratch. Follow these rules to
keep the new codebase auditable and reproducible.

## Environment & Tooling

1. Create a fresh virtual environment and install both runtime and dev dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
2. Run formatting, linting, typing, and tests before every commit:
   ```bash
   black .
   isort .
   ruff check .
   mypy .
   pytest
   ```

## Coding Standards

- Target Python 3.11 features and typing (PEP 563 style annotations already enabled).
- Keep modules ASCII-only unless data requires otherwise.
- Prefer dataclasses for configuration/data containers; avoid long helper scripts.
- Place reusable logic in `src/` and keep notebooks or ad-hoc analysis in `notebooks/`.
- Every new script should emit a run manifest (see `src.logging_utils`) that records config,
  timestamp, and any seeds used.

## Logging & Experiment Tracking

- Use `RunLogger` for consistent log formatting. Do not create ad-hoc loggers.
- Store run artifacts under `results/<timestamp>/` or `logs/<timestamp>/`; never reuse older
  directories for new runs.
- Include configuration files (YAML/JSON) next to experiment outputs for replayability.

## Pull Request Checklist

- [ ] Code formatted + linted + typed + tested.
- [ ] Added or updated unit tests for new functionality.
- [ ] Updated documentation (README, ROADMAP, TODO) if behavior changed.
- [ ] Included run manifests/logs for any experiments referenced in the PR.

Thank you for helping rebuild the project with higher rigor.
