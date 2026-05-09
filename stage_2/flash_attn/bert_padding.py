"""Stub for `flash_attn.bert_padding`.

Some model code imports these helpers; we provide placeholders that fail if used.
"""

from __future__ import annotations

from typing import Any


def _not_available(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "flash_attn is not installed (stub in Capstone/stage_2/flash_attn). "
        "Install 'flash-attn' for bert_padding helpers."
    )


pad_input = _not_available
unpad_input = _not_available
index_first_axis = _not_available
