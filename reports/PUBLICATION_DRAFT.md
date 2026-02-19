# Induction Heads and Arithmetic – Preliminary Report

## Abstract
Placeholder abstract describing how attention/head interventions will be tested once
model runs are performed.

## Introduction
- Motivation: arithmetic circuits vs. induction heads
- Research question: can attention steering improve arithmetic?

## Methods
- Datasets: tiered suite + GSM-style prompts (hashes logged in manifests)
- Models: see `docs/multi_model_plan.md`
- Interventions: attention/neuron suppression sweeps (configured via hooks)

## Results (To Be Filled)
- Baseline stability table per tier and model
- Intervention deltas with statistical summaries (from `src/statistics.py`)

## Discussion
- Interpret potential improvements
- Limitations: placeholder until GPU experiments run

## Appendix
- Manifest schema and logging requirements
- Instructions for rerunning experiments in tmux
