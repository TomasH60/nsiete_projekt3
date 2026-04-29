"""Text classification router for guarded query routing."""

from router.metrics import gqr_score

__all__ = ["DomainRouter", "gqr_score"]


def __getattr__(name: str) -> object:
    if name == "DomainRouter":
        from router.model import DomainRouter

        return DomainRouter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
