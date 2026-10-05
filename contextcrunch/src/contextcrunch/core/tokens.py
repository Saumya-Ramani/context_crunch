"""Token counting and text clipping.

The tokenizer is loaded once and cached. If tiktoken is unusable the module falls
back to counting characters, which is a safe over-estimate: it never claims a
message is smaller than it is.
"""

from __future__ import annotations

from functools import lru_cache

from contextcrunch.core.settings import get_settings


@lru_cache(maxsize=4)
def _encoding(name: str) -> object | None:
    """Return a tiktoken encoding, or ``None`` when it cannot be loaded."""
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception:  # noqa: BLE001 - any failure must degrade, never crash
        return None


def count_tokens(text: str) -> int:
    """Count the tokens in ``text``. Falls back to counting characters."""
    if not text:
        return 0
    encoding = _encoding(get_settings().tiktoken_encoding)
    if encoding is None:
        return len(text)
    return len(encoding.encode(text, disallowed_special=()))


def clip_to_chars(text: str, max_chars: int) -> str:
    """Clip ``text`` to ``max_chars`` characters."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def clip_to_tokens(text: str, max_tokens: int) -> str:
    """Clip ``text`` to roughly ``max_tokens`` tokens."""
    if max_tokens <= 0:
        return ""
    encoding = _encoding(get_settings().tiktoken_encoding)
    if encoding is None:
        return clip_to_chars(text, max_tokens)
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])
