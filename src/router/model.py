"""Embedding-based guarded query router."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from router.cache import sentence_transformers_cache_dir
from router.labels import ID_LABELS, OOD
from router.metrics import evaluate_router, gqr_score

logger = logging.getLogger(__name__)

LOCAL_EMBEDDING_DIR = "embedding_model"


@dataclass(frozen=True)
class Prediction:
    """Single router prediction with rejection diagnostics."""

    label: int
    confidence: float
    domain_probability: float


class SentenceTransformerEmbedder:
    """Thin wrapper around sentence-transformers to keep imports lazy."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for embeddings. "
                "Install project requirements first."
            ) from exc

        self.model_name = model_name
        cache_dir = sentence_transformers_cache_dir()
        logger.info("Loading embedding model: %s", model_name)
        logger.info("Using sentence-transformers cache: %s", cache_dir)
        self.model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
        logger.info("Embedding model loaded")

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        logger.info("Encoding %d texts with batch_size=%d", len(texts), batch_size)
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) >= 512,
        )

    def save(self, output_dir: Union[str, Path]) -> None:
        logger.info("Saving embedding model to %s", output_dir)
        self.model.save(str(output_dir))


class DomainRouter:
    """Classify Law/Finance/Health queries and reject low-confidence OOD input."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.55,
        classifier: Optional[LogisticRegression] = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.classifier = classifier or LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=1000,
            n_jobs=-1,
            random_state=42,
        )
        self._embedder: Optional[SentenceTransformerEmbedder] = None

    @property
    def embedder(self) -> SentenceTransformerEmbedder:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(self.embedding_model)
        return self._embedder

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        valid_texts: Optional[Sequence[str]] = None,
        valid_labels: Optional[Sequence[int]] = None,
        target_id_recall: float = 0.95,
        batch_size: int = 64,
    ) -> "DomainRouter":
        """Fit the ID classifier and calibrate the OOD confidence threshold."""

        logger.info("Preparing training data: %d total rows", len(texts))
        train_texts, train_labels = self._only_id(texts, labels)
        if not train_texts:
            raise ValueError("Training data must contain Law/Finance/Health labels.")

        logger.info(
            "Training domain classifier on %d in-domain rows",
            len(train_texts),
        )
        train_embeddings = self.embedder.encode(train_texts, batch_size=batch_size)
        logger.info("Fitting logistic-regression classifier")
        self.classifier.fit(train_embeddings, train_labels)
        logger.info("Classifier fitted")

        if valid_texts is not None and valid_labels is not None:
            logger.info("Calibrating threshold on %d validation rows", len(valid_texts))
            self.calibrate_threshold(
                valid_texts,
                valid_labels,
                target_id_recall=target_id_recall,
                batch_size=batch_size,
            )
            logger.info("Calibrated OOD threshold: %.4f", self.threshold)

        return self

    def predict_one(self, text: str) -> int:
        """GQR-compatible scoring function: str -> label in {0, 1, 2, 3}."""

        return self.predict([text])[0]

    def predict(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
    ) -> list[int]:
        return [
            prediction.label
            for prediction in self.predict_with_scores(texts, batch_size=batch_size)
        ]

    def predict_with_scores(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
    ) -> list[Prediction]:
        embeddings = self.embedder.encode(texts, batch_size=batch_size)
        probabilities = self.classifier.predict_proba(embeddings)
        class_labels = list(self.classifier.classes_)

        predictions: list[Prediction] = []
        for row in probabilities:
            best_index = int(np.argmax(row))
            best_label = int(class_labels[best_index])
            best_probability = float(row[best_index])
            label = best_label if best_probability >= self.threshold else OOD
            predictions.append(
                Prediction(
                    label=label,
                    confidence=best_probability,
                    domain_probability=best_probability,
                )
            )
        return predictions

    def calibrate_threshold(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        target_id_recall: float = 0.95,
        batch_size: int = 64,
    ) -> float:
        """Tune the confidence threshold on validation data.

        If validation contains OOD examples, choose the threshold with the best
        validation GQR score. Otherwise, choose a low confidence quantile that
        keeps approximately target_id_recall of ID validation examples accepted.
        """

        if len(texts) == 0:
            logger.info("Skipping threshold calibration because validation is empty")
            return self.threshold

        embeddings = self.embedder.encode(texts, batch_size=batch_size)
        probabilities = self.classifier.predict_proba(embeddings)
        max_probabilities = probabilities.max(axis=1)

        if any(label == OOD for label in labels):
            logger.info("Validation contains OOD rows; searching threshold by GQR score")
            values = sorted(set(float(value) for value in max_probabilities))
            midpoints = [
                (left + right) / 2
                for left, right in zip(values, values[1:])
            ]
            thresholds = [0.0, *midpoints, 1.000001]
            best_threshold = self.threshold
            best_score = -1.0

            for threshold in thresholds:
                predictions = self._labels_from_probabilities(
                    probabilities,
                    threshold,
                )
                scores = evaluate_router(list(labels), predictions)
                if scores.gqr_score > best_score:
                    best_score = scores.gqr_score
                    best_threshold = threshold

            self.threshold = best_threshold
            logger.info(
                "Best validation threshold %.4f with GQR score %.4f",
                self.threshold,
                best_score,
            )
            return self.threshold

        quantile = max(0.0, min(1.0, 1.0 - target_id_recall))
        self.threshold = float(np.quantile(max_probabilities, quantile))
        logger.info(
            "Validation has only ID rows; threshold %.4f targets %.1f%% ID recall",
            self.threshold,
            target_id_recall * 100,
        )
        return self.threshold

    def save(self, model_dir: Union[str, Path]) -> None:
        """Persist classifier metadata and a local copy of the embedding model."""

        output_dir = Path(model_dir)
        logger.info("Saving router artifacts to %s", output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self._embedder is not None:
            self._embedder.save(output_dir / LOCAL_EMBEDDING_DIR)
        joblib.dump(
            {
                "embedding_model": self.embedding_model,
                "threshold": self.threshold,
                "classifier": self.classifier,
            },
            output_dir / "router.joblib",
        )

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "DomainRouter":
        input_dir = Path(model_dir)
        payload = joblib.load(input_dir / "router.joblib")
        local_embedding_dir = input_dir / LOCAL_EMBEDDING_DIR
        embedding_model = payload["embedding_model"]
        if local_embedding_dir.is_dir():
            embedding_model = str(local_embedding_dir)
            logger.info("Using local embedding model from %s", local_embedding_dir)
        else:
            logger.info("No local embedding model found; using %s", embedding_model)
        return cls(
            embedding_model=embedding_model,
            threshold=payload["threshold"],
            classifier=payload["classifier"],
        )

    @staticmethod
    def _only_id(
        texts: Sequence[str],
        labels: Sequence[int],
    ) -> tuple[list[str], list[int]]:
        id_texts: list[str] = []
        id_labels: list[int] = []
        for text, label in zip(texts, labels):
            if label in ID_LABELS:
                id_texts.append(text)
                id_labels.append(label)
        return id_texts, id_labels

    def _labels_from_probabilities(
        self,
        probabilities: np.ndarray,
        threshold: float,
    ) -> list[int]:
        class_labels = list(self.classifier.classes_)
        predictions: list[int] = []
        for row in probabilities:
            best_index = int(np.argmax(row))
            best_probability = float(row[best_index])
            best_label = int(class_labels[best_index])
            predictions.append(best_label if best_probability >= threshold else OOD)
        return predictions


def score_from_accuracies(id_accuracy: float, ood_accuracy: float) -> float:
    """Backward-compatible alias for notebooks."""

    return gqr_score(id_accuracy, ood_accuracy)
