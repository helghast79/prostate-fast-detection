"""Cascade prostate MRI pipeline: localize → ROI crop → segment.

See ``src.cascade.decisions`` for locked architecture choices.
"""

from .decisions import CASCADE_DECISIONS

__all__ = ["CASCADE_DECISIONS"]
