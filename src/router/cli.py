"""Command line interface for training and evaluating the router."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from router.data import (
    DatasetSplit,
    load_builtin_ood_validation_dataset,
    load_gqr_train_dataset,
    load_gqr_ood_test_dataset,
    load_tabular_dataset,
)
from router.labels import LABEL_TO_DOMAIN, OOD
from router.metrics import confusion_counts, evaluate_router

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and use an embedding-based guarded query router.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="logging verbosity",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a router model")
    train_parser.add_argument("--train-path", type=Path)
    train_parser.add_argument("--valid-path", type=Path)
    train_parser.add_argument("--model-dir", type=Path, default=Path("artifacts/router"))
    train_parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    train_parser.add_argument("--threshold", type=float, default=0.55)
    train_parser.add_argument("--target-id-recall", type=float, default=0.95)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--max-train-samples", type=int)
    train_parser.add_argument("--max-valid-samples", type=int)
    train_parser.add_argument("--ood-valid-path", type=Path)
    train_parser.add_argument("--max-ood-valid-samples", type=int, default=1000)
    train_parser.add_argument(
        "--skip-ood-validation",
        action="store_true",
        help="only report in-domain validation accuracy after training",
    )

    predict_parser = subparsers.add_parser("predict", help="route one query")
    predict_parser.add_argument("text")
    predict_parser.add_argument("--model-dir", type=Path, default=Path("artifacts/router"))

    cache_parser = subparsers.add_parser(
        "cache-embedding",
        help="save the router embedding model into the model directory",
    )
    cache_parser.add_argument("--model-dir", type=Path, default=Path("artifacts/router"))

    eval_parser = subparsers.add_parser("evaluate", help="evaluate on a labeled file")
    eval_parser.add_argument("path", type=Path)
    eval_parser.add_argument("--model-dir", type=Path, default=Path("artifacts/router"))
    eval_parser.add_argument("--batch-size", type=int, default=64)

    gqr_parser = subparsers.add_parser(
        "score-gqr",
        help="run the official gqr.score_batch benchmark when gqr is installed",
    )
    gqr_parser.add_argument("--model-dir", type=Path, default=Path("artifacts/router"))
    gqr_parser.add_argument("--batch-size", type=int, default=64)

    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_training_splits(args: argparse.Namespace) -> tuple[DatasetSplit, DatasetSplit]:
    if args.train_path:
        logger.info("Using local training data")
        train = load_tabular_dataset(args.train_path)
        valid = load_tabular_dataset(args.valid_path) if args.valid_path else train
        return train, valid
    logger.info("Using official GQR training data")
    return load_gqr_train_dataset()


def load_ood_validation_split(args: argparse.Namespace) -> DatasetSplit:
    if args.skip_ood_validation:
        logger.info("Skipping OOD validation reporting")
        return DatasetSplit(texts=[], labels=[])

    if args.ood_valid_path:
        logger.info("Using local OOD validation data")
        return load_tabular_dataset(args.ood_valid_path).limit(
            args.max_ood_valid_samples
        )

    try:
        return load_gqr_ood_test_dataset().limit(args.max_ood_valid_samples)
    except Exception as exc:
        logger.warning(
            "Could not load official/public OOD validation data: %s: %s",
            type(exc).__name__,
            exc,
        )
        return load_builtin_ood_validation_dataset(
            max_samples=args.max_ood_valid_samples
        )


def train(args: argparse.Namespace) -> None:
    from router.model import DomainRouter

    logger.info("Starting router training")
    train_split, valid_split = load_training_splits(args)
    train_split = train_split.limit(args.max_train_samples)
    valid_split = valid_split.limit(args.max_valid_samples)
    logger.info(
        "Training rows: %d; validation rows: %d",
        len(train_split.texts),
        len(valid_split.texts),
    )

    router = DomainRouter(
        embedding_model=args.embedding_model,
        threshold=args.threshold,
    )
    router.fit(
        train_split.texts,
        train_split.labels,
        valid_texts=valid_split.texts,
        valid_labels=valid_split.labels,
        target_id_recall=args.target_id_recall,
        batch_size=args.batch_size,
    )

    if any(label == OOD for label in valid_split.labels):
        logger.info("Validation split already contains OOD rows")
        report_split = valid_split
    else:
        ood_valid_split = load_ood_validation_split(args)
        report_split = valid_split.extend(ood_valid_split)

    id_rows = sum(1 for label in report_split.labels if label != OOD)
    ood_rows = sum(1 for label in report_split.labels if label == OOD)
    logger.info(
        "Running validation predictions on %d ID rows and %d OOD rows",
        id_rows,
        ood_rows,
    )
    predictions = router.predict(report_split.texts, batch_size=args.batch_size)
    scores = evaluate_router(report_split.labels, predictions)

    router.save(args.model_dir)

    print(f"Saved model to {args.model_dir}")
    print(f"Threshold: {router.threshold:.4f}")
    print(f"Validation rows: {id_rows} ID, {ood_rows} OOD")
    print(f"Validation ID accuracy: {scores.id_accuracy:.4f}")
    if ood_rows:
        print(f"Validation OOD accuracy: {scores.ood_accuracy:.4f}")
        print(f"Validation GQR score: {scores.gqr_score:.4f}")
    else:
        print("Validation OOD accuracy: n/a (no OOD rows)")
        print("Validation GQR score: n/a (requires OOD rows)")


def predict(args: argparse.Namespace) -> None:
    from router.model import DomainRouter

    router = DomainRouter.load(args.model_dir)
    prediction = router.predict_with_scores([args.text])[0]
    print(
        {
            "label": prediction.label,
            "domain": LABEL_TO_DOMAIN[prediction.label],
            "confidence": round(prediction.confidence, 4),
        }
    )


def cache_embedding(args: argparse.Namespace) -> None:
    from router.model import DomainRouter

    router = DomainRouter.load(args.model_dir)
    router.embedder
    router.save(args.model_dir)
    print(f"Saved local embedding model to {args.model_dir / 'embedding_model'}")


def evaluate(args: argparse.Namespace) -> None:
    from router.model import DomainRouter

    split = load_tabular_dataset(args.path)
    router = DomainRouter.load(args.model_dir)
    predictions = router.predict(split.texts, batch_size=args.batch_size)
    scores = evaluate_router(split.labels, predictions)

    print(f"ID accuracy: {scores.id_accuracy:.4f}")
    print(f"OOD accuracy: {scores.ood_accuracy:.4f}")
    print(f"GQR score: {scores.gqr_score:.4f}")
    print("Confusion counts:")
    for (true, pred), count in sorted(confusion_counts(split.labels, predictions).items()):
        print(f"  true={true} pred={pred}: {count}")


def score_gqr(args: argparse.Namespace) -> None:
    from router.model import DomainRouter

    try:
        import gqr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "The official gqr package is required for score-gqr. "
            "Use Python 3.12+ and install gqr."
        ) from exc

    logger.info("Loading trained router from %s", args.model_dir)
    router = DomainRouter.load(args.model_dir)
    logger.info("Running official GQR benchmark scoring")
    scores = gqr.score_batch(
        lambda batch: router.predict(batch, batch_size=args.batch_size),
        batch_size=args.batch_size,
    )
    print(scores)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "cache-embedding":
        cache_embedding(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "score-gqr":
        score_gqr(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
