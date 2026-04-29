"""GQR-compatible scoring functions."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence, Union

DEFAULT_MODEL_DIR = Path("artifacts/router")


@lru_cache(maxsize=4)
def _load_router(model_dir: str) -> Any:
    from router.model import DomainRouter

    return DomainRouter.load(model_dir)


def scoring_function(
    text: str,
    model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
) -> int:
    """Return one GQR label for one text: 0=law, 1=finance, 2=health, 3=OOD."""

    router = _load_router(str(model_dir))
    return router.predict_one(text)


def scoring_function_batch(
    texts: Sequence[str],
    model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
    batch_size: int = 64,
) -> list[int]:
    """Batch variant suitable for gqr.score_batch."""

    router = _load_router(str(model_dir))
    return router.predict(list(texts), batch_size=batch_size)
