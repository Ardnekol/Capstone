"""Stub implementations of `flash_attn.flash_attn_interface`.

These functions deliberately fail if called; they exist to let models import.
"""

from __future__ import annotations

from typing import Any


def _not_available(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "flash_attn is not installed. This is a stub module placed under "
        "Capstone/stage_2/flash_attn to satisfy optional imports. "
        "Install the real 'flash-attn' package if you need FlashAttention."
    )


flash_attn_func = _not_available
flash_attn_qkvpacked_func = _not_available
flash_attn_varlen_qkvpacked_func = _not_available
