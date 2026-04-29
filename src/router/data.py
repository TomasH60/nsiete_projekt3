"""Dataset loading and normalization for router training."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from router.labels import OOD, normalize_label

logger = logging.getLogger(__name__)

DATASET_SIZE = 15_000
TRAIN_SPLIT = 0.2
SEED = 42

TEXT_COLUMN_CANDIDATES = (
    "text",
    "query",
    "question",
    "prompt",
    "instruction",
    "passage",
)
LABEL_COLUMN_CANDIDATES = ("label", "domain", "category", "class")


@dataclass(frozen=True)
class DatasetSplit:
    """Normalized text classification split."""

    texts: list[str]
    labels: list[int]

    def limit(self, max_samples: Optional[int]) -> "DatasetSplit":
        if max_samples is None or max_samples <= 0:
            return self
        return DatasetSplit(
            texts=self.texts[:max_samples],
            labels=self.labels[:max_samples],
        )

    def extend(self, other: "DatasetSplit") -> "DatasetSplit":
        return DatasetSplit(
            texts=[*self.texts, *other.texts],
            labels=[*self.labels, *other.labels],
        )


def _pick_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str:
    columns_by_lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in columns_by_lower:
            return columns_by_lower[candidate]
    raise ValueError(
        "Could not find any of these columns: " + ", ".join(candidates)
    )


def normalize_frame(frame: Any) -> DatasetSplit:
    """Normalize a dataframe into text and integer labels."""

    text_column = _pick_column(frame.columns, TEXT_COLUMN_CANDIDATES)
    label_column = _pick_column(frame.columns, LABEL_COLUMN_CANDIDATES)

    normalized = frame[[text_column, label_column]].dropna()
    texts = [str(value).strip() for value in normalized[text_column]]
    labels = [normalize_label(value) for value in normalized[label_column]]

    clean_texts: list[str] = []
    clean_labels: list[int] = []
    for text, label in zip(texts, labels):
        if text:
            clean_texts.append(text)
            clean_labels.append(label)

    split = DatasetSplit(texts=clean_texts, labels=clean_labels)
    logger.info(
        "Normalized dataset: %d usable rows, text column=%s, label column=%s",
        len(split.texts),
        text_column,
        label_column,
    )
    return split


def load_tabular_dataset(path: Union[str, Path]) -> DatasetSplit:
    """Load CSV, JSON, JSONL, or Parquet data with text/query and label columns."""

    import pandas as pd

    data_path = Path(path)
    suffix = data_path.suffix.lower()
    logger.info("Loading local dataset from %s", data_path)

    if suffix == ".csv":
        frame = pd.read_csv(data_path)
    elif suffix in {".jsonl", ".ndjson"}:
        records = [json.loads(line) for line in data_path.read_text().splitlines()]
        frame = pd.DataFrame(records)
    elif suffix == ".json":
        frame = pd.read_json(data_path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported dataset format: {data_path}")

    return normalize_frame(frame)


def load_gqr_train_dataset() -> tuple[DatasetSplit, DatasetSplit]:
    """Load official GQR training/validation data when the package is installed."""

    try:
        import gqr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "The official gqr package is not installed. Install it in a Python "
            "3.12+ environment, or pass --train-path/--valid-path instead."
        ) from exc

    logger.info("Loading official GQR training and validation datasets")
    try:
        train_frame, valid_frame = gqr.load_train_dataset()
    except Exception as exc:
        logger.warning(
            "Official gqr.load_train_dataset() failed: %s: %s",
            type(exc).__name__,
            exc,
        )
        logger.warning(
            "Falling back to public Hugging Face dataset ids for Law/Finance/Health"
        )
        return load_public_gqr_train_dataset()

    train_split = normalize_frame(train_frame)
    valid_split = normalize_frame(valid_frame)
    logger.info(
        "Loaded GQR splits: %d train rows, %d validation rows",
        len(train_split.texts),
        len(valid_split.texts),
    )
    return train_split, valid_split


def load_public_gqr_train_dataset() -> tuple[DatasetSplit, DatasetSplit]:
    """Load public dataset ids matching the GQR in-domain domains.

    This is a resilience fallback for cases where the official gqr package points
    at a Hugging Face dataset id that is private, renamed, or unavailable.
    """

    from datasets import concatenate_datasets, load_dataset
    from sklearn.model_selection import train_test_split

    logger.info("Loading public law dataset: dim/law_stackexchange_prompts")
    law_dataset = load_dataset("dim/law_stackexchange_prompts")["train"]

    logger.info(
        "Loading public finance dataset: "
        "Marina-C/question-answer-Subject-Finance-Instruct"
    )
    finance_dataset = load_dataset(
        "Marina-C/question-answer-Subject-Finance-Instruct"
    )["train"]

    logger.info(
        "Loading public health dataset: "
        "iecjsu/lavita-ChatDoctor-HealthCareMagic-100k"
    )
    healthcare_dataset = load_dataset(
        "iecjsu/lavita-ChatDoctor-HealthCareMagic-100k"
    )["train"]

    law_filtered = law_dataset.filter(_has_nonempty_prompt)
    law_data = _take_dataset(law_filtered).map(
        lambda row: {"text": row["prompt"], "domain": "law", "label": 0},
        remove_columns=law_filtered.column_names,
    )

    finance_filtered = finance_dataset.map(
        lambda row: {
            "text": _extract_user_message(row["messages"]),
            "domain": "finance",
            "label": 1,
        },
        remove_columns=finance_dataset.column_names,
    ).filter(_has_nonempty_text)
    finance_data = _take_dataset(finance_filtered)

    healthcare_filtered = healthcare_dataset.filter(_has_nonempty_input)
    healthcare_data = _take_dataset(healthcare_filtered).map(
        lambda row: {
            "text": str(row["input"]),
            "domain": "healthcare",
            "label": 2,
        },
        remove_columns=healthcare_filtered.column_names,
    )

    logger.info(
        "Prepared public ID rows: law=%d, finance=%d, health=%d",
        len(law_data),
        len(finance_data),
        len(healthcare_data),
    )
    combined_dataset = concatenate_datasets([law_data, finance_data, healthcare_data])
    split = combined_dataset.train_test_split(test_size=TRAIN_SPLIT, seed=SEED)
    train_frame = split["train"].to_pandas()

    train_frame = train_frame.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_frame, valid_frame = train_test_split(
        train_frame,
        test_size=TRAIN_SPLIT,
        random_state=SEED,
        stratify=train_frame["domain"],
    )

    train_split = normalize_frame(train_frame)
    valid_split = normalize_frame(valid_frame)
    logger.info(
        "Loaded public fallback splits: %d train rows, %d validation rows",
        len(train_split.texts),
        len(valid_split.texts),
    )
    return train_split, valid_split


def _has_nonempty_prompt(row: dict[str, Any]) -> bool:
    value = row.get("prompt")
    return value is not None and str(value).strip() != ""


def _has_nonempty_input(row: dict[str, Any]) -> bool:
    value = row.get("input")
    return value is not None and str(value).strip() != ""


def _has_nonempty_text(row: dict[str, Any]) -> bool:
    value = row.get("text")
    return value is not None and str(value).strip() != ""


def _take_dataset(dataset: Any, size: int = DATASET_SIZE) -> Any:
    if size <= 0:
        return dataset
    return dataset.select(range(min(size, len(dataset))))


def _extract_user_message(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def load_gqr_id_test_dataset() -> DatasetSplit:
    """Load official GQR in-domain test data when available."""

    try:
        import gqr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("The official gqr package is not installed.") from exc

    logger.info("Loading official GQR ID test dataset")
    return normalize_frame(gqr.load_id_test_dataset())


def load_gqr_ood_test_dataset() -> DatasetSplit:
    """Load official GQR OOD test data when available."""

    try:
        import gqr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("The official gqr package is not installed.") from exc

    logger.info("Loading official GQR OOD test dataset")
    try:
        return normalize_frame(gqr.load_ood_test_dataset())
    except Exception as exc:
        logger.warning(
            "Official gqr.load_ood_test_dataset() failed: %s: %s",
            type(exc).__name__,
            exc,
        )
        logger.warning("Falling back to public OOD validation datasets")
        return load_public_ood_validation_dataset()


def load_public_ood_validation_dataset(max_samples: int = 1_000) -> DatasetSplit:
    """Load public non-domain questions for OOD validation reporting."""

    try:
        from datasets import concatenate_datasets, load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load public OOD data") from exc

    try:
        logger.info("Loading public OOD dataset: Stanford/web_questions")
        web_questions = load_dataset("Stanford/web_questions", split="test")
        web_questions = web_questions.map(
            lambda row: {
                "text": str(row["question"]),
                "domain": "ood",
                "label": OOD,
            },
            remove_columns=web_questions.column_names,
        ).filter(_has_nonempty_text)

        logger.info("Loading public OOD dataset: mjphayes/machine_learning_questions")
        ml_questions = load_dataset(
            "mjphayes/machine_learning_questions",
            split="test",
        )
        ml_questions = ml_questions.map(
            lambda row: {
                "text": str(row["question"]),
                "domain": "ood",
                "label": OOD,
            },
            remove_columns=ml_questions.column_names,
        ).filter(_has_nonempty_text)

        combined = concatenate_datasets([web_questions, ml_questions])
        combined = _take_dataset(combined, max_samples)
        split = normalize_frame(combined.to_pandas())
        logger.info("Loaded %d public OOD validation rows", len(split.texts))
        return split
    except Exception as exc:
        logger.warning(
            "Public OOD validation loading failed: %s: %s",
            type(exc).__name__,
            exc,
        )
        logger.warning("Falling back to built-in OOD sanity examples")
        return load_builtin_ood_validation_dataset(max_samples=max_samples)


def load_builtin_ood_validation_dataset(max_samples: int = 1_000) -> DatasetSplit:
    """Tiny built-in OOD set used only when external OOD data is unavailable."""

    examples = [
        "Who won the FIFA World Cup in 2018?",
        "How do I bake sourdough bread at home?",
        "What is the capital city of Australia?",
        "Explain how merge sort works.",
        "Write a Python function that reverses a string.",
        "What are the main causes of climate change?",
        "How far is Mars from Earth?",
        "Give me tips for growing tomatoes indoors.",
        "What is the plot of Hamlet?",
        "How do I tune an acoustic guitar?",
        "Who painted the Mona Lisa?",
        "What is the difference between HTTP and HTTPS?",
        "Plan a three day trip to Prague.",
        "How do I remove coffee stains from a shirt?",
        "What is the fastest way to learn Spanish vocabulary?",
        "Explain Newton's second law of motion.",
        "What ingredients do I need for pad thai?",
        "How do I configure SSH keys for GitHub?",
        "What is the population of Tokyo?",
        "Give me a beginner workout plan.",
        "How do chess openings work?",
        "What is Kubernetes used for?",
        "Recommend a birthday gift for a twelve year old.",
        "How do I change a bicycle tire?",
        "What is photosynthesis?",
        "Summarize the history of the Roman Empire.",
        "How can I improve my public speaking?",
        "What does a CPU cache do?",
        "How do I make cold brew coffee?",
        "What is the tallest mountain in Europe?",
    ]
    selected = examples if max_samples <= 0 else examples[:max_samples]
    logger.info("Loaded %d built-in OOD validation rows", len(selected))
    return DatasetSplit(texts=selected, labels=[OOD] * len(selected))
