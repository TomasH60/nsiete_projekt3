# NSIETE - Project 3

This repository contains materials related to Project 3 for the NSIETE course, including source code and supporting documents.

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

## Router baseline

This project now includes a first embedding-based guarded query router. It learns
to route in-domain chatbot questions to:

- `0` = Law
- `1` = Finance / Fintech
- `2` = Health
- `3` = Out-of-domain (OOD)

The model uses a SentenceTransformer embedding model and a logistic-regression
classifier over the three in-domain classes. OOD detection is handled by a
confidence threshold: if the best domain probability is below the threshold, the
router returns label `3`.

The GQR score is the harmonic mean of in-domain routing accuracy and OOD
rejection accuracy:

```text
GQR = 2 * ID_accuracy * OOD_accuracy / (ID_accuracy + OOD_accuracy)
```

### Setup

The project uses the official `gqr` package, so use Python 3.12 or newer.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### Train

With the official GQR package installed:

```bash
PYTHONPATH=src python -m router train --model-dir artifacts/router
```

If the official package cannot access its hard-coded finance dataset, training
falls back to public Hugging Face datasets for the same three in-domain areas:
`dim/law_stackexchange_prompts`,
`Marina-C/question-answer-Subject-Finance-Instruct`, and
`iecjsu/lavita-ChatDoctor-HealthCareMagic-100k`.

Training reports ID validation accuracy on the Law/Finance/Health validation
split and OOD accuracy on a separate OOD reporting split. The command tries the
official GQR OOD data first, then public OOD datasets, and finally a small
built-in OOD sanity set if external OOD data is unavailable.

Useful training options:

```bash
PYTHONPATH=src python -m router train \
  --model-dir artifacts/router \
  --max-ood-valid-samples 1000

PYTHONPATH=src python -m router train \
  --model-dir artifacts/router \
  --ood-valid-path data/ood_valid.csv

PYTHONPATH=src python -m router train \
  --model-dir artifacts/router \
  --skip-ood-validation
```

With a local CSV/JSONL/Parquet file:

```bash
PYTHONPATH=src python -m router train \
  --train-path data/train.csv \
  --valid-path data/valid.csv \
  --model-dir artifacts/router
```

Input files need one text-like column (`text`, `query`, `question`, `prompt`,
`instruction`, or `passage`) and one label-like column (`label`, `domain`,
`category`, or `class`). Labels can be integers `0..3` or strings such as
`law`, `fintech`, `healthcare`, or `other`.

### Predict

```bash
PYTHONPATH=src python -m router predict \
  --model-dir artifacts/router \
  "Can my landlord keep my deposit after I move out?"
```

### Evaluate

On a labeled local file:

```bash
PYTHONPATH=src python -m router evaluate data/test.csv --model-dir artifacts/router
```

On the official benchmark:

```bash
PYTHONPATH=src python -m router score-gqr --model-dir artifacts/router
```

The same trained model can be used from Python with the benchmark-compatible
scoring function:

```python
from router.scoring import scoring_function

label = scoring_function("How do I appeal a denied insurance claim?")
```

### Sources

- GQR-Bench project page: https://gqr-bench.github.io/
- Official package/code: https://github.com/williambrach/gqr
