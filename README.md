# NSIETE - Project 3

This repository contains materials related to Project 3 for the NSIETE course. The work is documented through notebooks, with supporting code kept in `src/router`.

- **Project name:** Text classification “router”
- **Authors:** Bc. Tomáš Horička, Bc. Juraj Budinsky
- **University:** Slovak University of Technology
- **Faculty:** Faculty of Informatics and Information Technologies STU

## Main dataset link
https://gqr-bench.github.io/

## Article link
https://arxiv.org/abs/2209.11055

## GQR paper link
https://arxiv.org/abs/2505.14524

## Project overview

The project studies a text-classification router for chatbot queries. The router
assigns in-domain questions to one of three supported domains and rejects
unsupported questions as out-of-domain:

- `0` = Law
- `1` = Finance / Fintech
- `2` = Health
- `3` = Out-of-domain (OOD)

The baseline notebook uses a SentenceTransformer embedding model and a
logistic-regression classifier over the three in-domain classes. OOD detection is
handled by a confidence threshold: if the best domain probability is below the
threshold, the router returns label `3`.

The GQR score is the harmonic mean of in-domain routing accuracy and OOD
rejection accuracy:

```text
GQR = 2 * ID_accuracy * OOD_accuracy / (ID_accuracy + OOD_accuracy)
```

## Notebook workflow

Use the notebooks in this order:

1. `notebooks/router_baseline.ipynb`

   Builds the first embedding-based guarded query router. It covers data loading,
   embedding generation, classifier training, threshold selection, validation,
   prediction inspection, saving artifacts, and optional official GQR scoring.

2. `notebooks/setfit_methodology.ipynb`

   Experiments with the methodology from
   `Efficient Few-Shot Learning Without Prompts`
   (https://arxiv.org/abs/2209.11055). It adapts SetFit to the router task by
   sampling a few labeled examples per domain, generating contrastive pairs,
   fine-tuning a SentenceTransformer, training a classifier head, and evaluating
   OOD rejection with a confidence threshold.

The notebooks import reusable helpers from `src/router`; the README focuses on
the notebook workflow used for the project write-up.

## Setup

The project uses the official `gqr` package, so use Python 3.12 or newer.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Downloaded Hugging Face datasets and embedding models are cached inside the
project by default under `data/cache/`. The notebooks also contain cells for
setting notebook-specific paths such as local train, validation, and OOD
validation files.

If the official package cannot access its hard-coded finance dataset, training
falls back to public Hugging Face datasets for the same three in-domain areas:
`dim/law_stackexchange_prompts`,
`Marina-C/question-answer-Subject-Finance-Instruct`, and
`iecjsu/lavita-ChatDoctor-HealthCareMagic-100k`.

Both notebooks report in-domain validation accuracy on the Law/Finance/Health
validation split and OOD accuracy on a separate OOD reporting split. The baseline
notebook can try the official GQR OOD data first, then public OOD datasets, and
finally a small built-in OOD sanity set if external OOD data is unavailable.

Local CSV, JSONL, or Parquet files can be used from the notebook configuration
cells. Input files need one text-like column (`text`, `query`, `question`,
`prompt`, `instruction`, or `passage`) and one label-like column (`label`,
`domain`, `category`, or `class`). Labels can be integers `0..3` or strings such
as `law`, `fintech`, `healthcare`, or `other`.

### Sources

- GQR-Bench project page: https://gqr-bench.github.io/
- Official package/code: https://github.com/williambrach/gqr
