"""Label constants used by the GQR benchmark."""

LAW = 0
FINANCE = 1
HEALTH = 2
OOD = 3

ID_LABELS = (LAW, FINANCE, HEALTH)
ALL_LABELS = (LAW, FINANCE, HEALTH, OOD)

LABEL_TO_DOMAIN = {
    LAW: "law",
    FINANCE: "finance",
    HEALTH: "health",
    OOD: "ood",
}

DOMAIN_TO_LABEL = {domain: label for label, domain in LABEL_TO_DOMAIN.items()}

DOMAIN_ALIASES = {
    "law": LAW,
    "legal": LAW,
    "finance": FINANCE,
    "fintech": FINANCE,
    "financial": FINANCE,
    "health": HEALTH,
    "healthcare": HEALTH,
    "medical": HEALTH,
    "medicine": HEALTH,
    "ood": OOD,
    "out_of_domain": OOD,
    "out-of-domain": OOD,
    "other": OOD,
}


def normalize_label(value: object) -> int:
    """Convert common label formats to the benchmark integer labels."""

    if isinstance(value, int):
        if value in ALL_LABELS:
            return value
        raise ValueError(f"Unsupported label integer: {value}")

    text = str(value).strip().lower()
    if text.isdigit():
        return normalize_label(int(text))
    if text in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[text]
    raise ValueError(f"Unsupported label value: {value!r}")
