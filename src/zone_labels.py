"""Map DICOM SEG segment labels onto gland (non-PZ) vs peripheral zone."""

from __future__ import annotations

# Class indices for softmax training / index masks.
CLASS_BG = 0
CLASS_GLAND = 1  # non-PZ prostate (TZ / CZ / TZ+CZ / …)
CLASS_PZ = 2
CLASS_NAMES = ("background", "gland_non_pz", "peripheral_zone")


def normalize_label(label: str) -> str:
    return " ".join(str(label or "").strip().lower().split())


def zone_class_for_label(label: str) -> int | None:
    """
    Return CLASS_GLAND / CLASS_PZ, or None to ignore (e.g. seminal vesicles).

    Whole-gland-only labels return None here — those SEGs are excluded from
    zone training via ``has_zone_supervision``.
    """
    text = normalize_label(label)
    if not text:
        return None
    if "seminal" in text or text in {"sv", "s.v.", "s.v"}:
        return None

    # Peripheral zone (including composites that are PZ-led).
    if (
        text == "pz"
        or text.startswith("pz+")
        or text.startswith("pz ")
        or "peripheral" in text
    ):
        return CLASS_PZ

    # Transition / central / remainder of glandular prostate.
    if (
        "tz" in text
        or "cz" in text
        or "transition" in text
        or "central" in text
    ):
        return CLASS_GLAND

    return None


def has_zone_supervision(segment_labels: list[str] | None) -> bool:
    """True if labels include both a gland (non-PZ) class and a PZ class."""
    classes = {zone_class_for_label(lab) for lab in (segment_labels or [])}
    return CLASS_GLAND in classes and CLASS_PZ in classes
