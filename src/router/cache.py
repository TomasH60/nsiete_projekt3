"""Project-local cache paths for downloaded datasets and models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
WINDOWS_CACHE_DIR = Path("C:/hf-cache/router")


def _default_cache_dir() -> Path:
    """Choose a cache root that avoids Windows path-length issues."""

    if os.name == "nt":
        return WINDOWS_CACHE_DIR
    return DEFAULT_CACHE_DIR


def configure_project_cache(
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Configure Hugging Face libraries to cache downloads inside the project."""

    root = Path(
        cache_dir
        if cache_dir is not None
        else os.environ.get("ROUTER_CACHE_DIR", _default_cache_dir())
    ).expanduser()
    root = root.resolve()

    hf_home = root / "huggingface"
    datasets_cache = hf_home / "datasets"
    hub_cache = hf_home / "hub"
    sentence_transformers_cache = hf_home / "sentence-transformers"

    for path in (root, hf_home, datasets_cache, hub_cache, sentence_transformers_cache):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(sentence_transformers_cache)

    return root


def datasets_cache_dir() -> Path:
    """Return the cache directory to pass to datasets.load_dataset."""

    return configure_project_cache() / "huggingface" / "datasets"


def sentence_transformers_cache_dir() -> Path:
    """Return the cache directory for sentence-transformers downloads."""

    return configure_project_cache() / "huggingface" / "sentence-transformers"


PROJECT_CACHE_DIR = configure_project_cache()
