"""Utilities for running richer SetFit notebook experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from router.cache import sentence_transformers_cache_dir
from router.labels import ID_LABELS, LABEL_TO_DOMAIN, OOD
from router.metrics import accuracy, confusion_counts, evaluate_router


@dataclass(frozen=True)
class SetFitExperimentConfig:
    """Configuration for a single SetFit-style experiment run."""

    run_name: str
    embedding_model: str
    n_shots: int = 8
    num_iterations: int = 20
    num_epochs: int = 1
    contrastive_batch_size: int = 16
    encode_batch_size: int = 64
    run_contrastive_fine_tuning: bool = True
    calibration_fraction: float = 0.5
    threshold_grid_size: int = 101
    seed: int = 42


def sample_few_shot_examples(
    frame: pd.DataFrame,
    n_shots: int,
    seed: int,
) -> pd.DataFrame:
    """Sample the same number of labeled examples from each in-domain class."""

    sampled_frames = []
    for label in ID_LABELS:
        class_frame = frame[frame["label"] == label]
        if len(class_frame) < n_shots:
            domain = LABEL_TO_DOMAIN[label]
            raise ValueError(
                f"Need {n_shots} examples for {domain}, found {len(class_frame)}"
            )
        sampled_frames.append(class_frame.sample(n=n_shots, random_state=seed + label))
    return pd.concat(sampled_frames).sample(frac=1, random_state=seed).reset_index(
        drop=True
    )


def build_contrastive_pairs(
    frame: pd.DataFrame,
    num_iterations: int,
    seed: int,
) -> list[Any]:
    """Create positive/negative sentence pairs from few-shot examples."""

    from sentence_transformers import InputExample

    rng = np.random.default_rng(seed)
    by_label = {
        label: frame[frame["label"] == label]["text"].tolist()
        for label in sorted(frame["label"].unique())
    }
    labels = list(by_label)
    examples: list[Any] = []

    for _ in range(num_iterations):
        for label, texts in by_label.items():
            first, second = rng.choice(texts, size=2, replace=False)
            examples.append(InputExample(texts=[first, second], label=1.0))

            other_label = int(
                rng.choice([candidate for candidate in labels if candidate != label])
            )
            negative = rng.choice(by_label[other_label])
            examples.append(InputExample(texts=[first, negative], label=0.0))

    rng.shuffle(examples)
    return examples


def split_calibration_report_frames(
    valid_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    calibration_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split validation and OOD rows into calibration and final report sets."""

    calibration_fraction = max(0.1, min(0.9, calibration_fraction))
    calibration_parts: list[pd.DataFrame] = []
    report_parts: list[pd.DataFrame] = []

    for source in (valid_df, ood_df):
        for _, group in source.groupby("label", sort=True):
            shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            split_index = int(round(len(shuffled) * calibration_fraction))
            split_index = min(max(split_index, 1), len(shuffled) - 1)
            calibration_parts.append(shuffled.iloc[:split_index])
            report_parts.append(shuffled.iloc[split_index:])

    calibration_df = (
        pd.concat(calibration_parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    report_df = (
        pd.concat(report_parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed + 1)
        .reset_index(drop=True)
    )
    return calibration_df, report_df


def encode_texts(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    """Encode texts with normalized sentence embeddings."""

    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) >= 512,
    )


def choose_best_threshold(
    true_labels: Iterable[int],
    best_labels: Iterable[int],
    confidences: np.ndarray,
    threshold_grid_size: int,
) -> tuple[float, pd.DataFrame]:
    """Search thresholds on a held-out calibration set."""

    true_label_list = list(true_labels)
    best_label_list = list(best_labels)

    threshold_rows = []
    for threshold in np.linspace(0.0, 1.0, threshold_grid_size):
        routed_predictions = [
            label if confidence >= threshold else OOD
            for label, confidence in zip(best_label_list, confidences)
        ]
        scores = evaluate_router(true_label_list, routed_predictions)
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "id_accuracy": scores.id_accuracy,
                "ood_accuracy": scores.ood_accuracy,
                "gqr_score": scores.gqr_score,
            }
        )

    threshold_results = pd.DataFrame(threshold_rows).sort_values(
        ["gqr_score", "ood_accuracy", "id_accuracy"],
        ascending=False,
    )
    best_threshold = float(threshold_results.iloc[0]["threshold"])
    return best_threshold, threshold_results.reset_index(drop=True)


def run_setfit_experiment(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    config: SetFitExperimentConfig,
) -> dict[str, Any]:
    """Run a single SetFit-style experiment and return rich outputs."""

    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import losses
    from torch.utils.data import DataLoader

    calibration_df, report_df = split_calibration_report_frames(
        valid_df=valid_df,
        ood_df=ood_df,
        calibration_fraction=config.calibration_fraction,
        seed=config.seed,
    )

    data_prep_start = time.perf_counter()
    few_shot_df = sample_few_shot_examples(train_df, config.n_shots, config.seed)
    pair_examples = build_contrastive_pairs(
        few_shot_df,
        num_iterations=config.num_iterations,
        seed=config.seed,
    )
    data_prep_seconds = time.perf_counter() - data_prep_start

    body_start = time.perf_counter()
    embedding_model = SentenceTransformer(
        config.embedding_model,
        cache_folder=str(sentence_transformers_cache_dir()),
    )
    if config.run_contrastive_fine_tuning:
        pair_loader = DataLoader(
            pair_examples,
            shuffle=True,
            batch_size=config.contrastive_batch_size,
        )
        train_loss = losses.CosineSimilarityLoss(embedding_model)
        warmup_steps = max(1, int(len(pair_loader) * config.num_epochs * 0.1))
        embedding_model.fit(
            train_objectives=[(pair_loader, train_loss)],
            epochs=config.num_epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=False,
        )
    body_training_seconds = time.perf_counter() - body_start

    head_start = time.perf_counter()
    few_shot_embeddings = encode_texts(
        embedding_model,
        few_shot_df["text"].tolist(),
        batch_size=config.encode_batch_size,
    )
    classifier = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=config.seed,
    )
    classifier.fit(few_shot_embeddings, few_shot_df["label"].tolist())
    head_training_seconds = time.perf_counter() - head_start

    calibration_embeddings = encode_texts(
        embedding_model,
        calibration_df["text"].tolist(),
        batch_size=config.encode_batch_size,
    )
    calibration_probabilities = classifier.predict_proba(calibration_embeddings)
    calibration_best_indices = np.argmax(calibration_probabilities, axis=1)
    calibration_best_labels = [
        int(classifier.classes_[index]) for index in calibration_best_indices
    ]
    calibration_confidences = calibration_probabilities.max(axis=1)
    best_threshold, threshold_results = choose_best_threshold(
        true_labels=calibration_df["label"].tolist(),
        best_labels=calibration_best_labels,
        confidences=calibration_confidences,
        threshold_grid_size=config.threshold_grid_size,
    )

    eval_start = time.perf_counter()
    report_embeddings = encode_texts(
        embedding_model,
        report_df["text"].tolist(),
        batch_size=config.encode_batch_size,
    )
    report_probabilities = classifier.predict_proba(report_embeddings)
    report_best_indices = np.argmax(report_probabilities, axis=1)
    report_best_labels = [int(classifier.classes_[index]) for index in report_best_indices]
    report_confidences = report_probabilities.max(axis=1)
    forced_predictions = report_best_labels
    routed_predictions = [
        label if confidence >= best_threshold else OOD
        for label, confidence in zip(report_best_labels, report_confidences)
    ]
    eval_seconds = time.perf_counter() - eval_start

    forced_id_mask = report_df["label"].isin(ID_LABELS)
    forced_id_accuracy = accuracy(
        report_df.loc[forced_id_mask, "label"].tolist(),
        list(np.array(forced_predictions)[forced_id_mask.to_numpy()]),
    )
    routed_scores = evaluate_router(report_df["label"].tolist(), routed_predictions)

    confusion_rows = [
        {
            "true_label": true,
            "true_domain": LABEL_TO_DOMAIN[true],
            "predicted_label": pred,
            "predicted_domain": LABEL_TO_DOMAIN[pred],
            "count": count,
            "run_name": config.run_name,
        }
        for (true, pred), count in sorted(
            confusion_counts(report_df["label"].tolist(), routed_predictions).items()
        )
    ]

    report_payload = asdict(config)
    report_payload.update(
        {
            "forced_id_accuracy": forced_id_accuracy,
            "threshold_id_accuracy": routed_scores.id_accuracy,
            "threshold_ood_accuracy": routed_scores.ood_accuracy,
            "threshold_gqr_score": routed_scores.gqr_score,
            "best_threshold": best_threshold,
            "few_shot_rows": len(few_shot_df),
            "pair_count": len(pair_examples),
            "calibration_rows": len(calibration_df),
            "report_rows": len(report_df),
            "data_prep_seconds": data_prep_seconds,
            "body_training_seconds": body_training_seconds,
            "head_training_seconds": head_training_seconds,
            "eval_seconds": eval_seconds,
            "total_seconds": (
                data_prep_seconds
                + body_training_seconds
                + head_training_seconds
                + eval_seconds
            ),
        }
    )

    return {
        "summary": report_payload,
        "threshold_results": threshold_results,
        "confusion": pd.DataFrame(confusion_rows),
        "calibration_frame": calibration_df,
        "report_frame": report_df.assign(
            forced_prediction=forced_predictions,
            routed_prediction=routed_predictions,
            confidence=report_confidences,
        ),
    }


def run_experiment_grid(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    configs: list[SetFitExperimentConfig],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Run a list of experiments and collect their outputs."""

    summaries = []
    details: dict[str, dict[str, Any]] = {}
    for config in configs:
        result = run_setfit_experiment(train_df, valid_df, ood_df, config)
        summaries.append(result["summary"])
        details[config.run_name] = result

    summary_df = pd.DataFrame(summaries).sort_values(
        "threshold_gqr_score",
        ascending=False,
    )
    return summary_df.reset_index(drop=True), details
