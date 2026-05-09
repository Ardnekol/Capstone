"""Stub fallback for the optional `flash_attn` dependency.

Some Hugging Face model repositories (including Florence-2) may declare `flash_attn`
as a hard import in their custom modeling code. On many clusters this package is
not available (and building it may be difficult).

This stub exists solely to satisfy import-time checks when the model is
configured to use standard attention (e.g. `attn_implementation="eager"`).

If the model actually tries to execute FlashAttention kernels, this stub will
raise a RuntimeError.
"""

from __future__ import annotations

from .flash_attn_interface import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)

__all__ = [
    "flash_attn_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_qkvpacked_func",
]

__version__ = "0.0.0-stub"
