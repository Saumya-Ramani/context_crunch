"""Splitting one history item into pieces that are each small enough to judge.

Laya has a limited input window, so a 14000-token filing cannot be sent whole. The
refiner splits an item into pieces, judges each one with the same five questions,
and rebuilds the item from whatever survives.

Two properties matter more than how clever the splitting is:

- **no piece may exceed ``max_chars``**. That is the whole reason this module
  exists, so the bound is enforced in one place at the end rather than trusted
  from every branch;
- **splitting is deterministic**. No model is consulted, so the same input always
  produces the same pieces, and a decision stays reproducible.

Both sizes come from Settings. This module deliberately has no defaults for them:
a silent default would mean a piece sized by a guess rather than by configuration.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

#: Separator used when a clipped item loses its middle.
CLIP_MARKER = "[... clipped ...]"

#: Paragraphs are separated by a blank line.
_PARAGRAPH = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Piece:
    """One split of an item: a short anchor naming where it came from, plus text."""

    anchor: str
    text: str


def split_text(text: str, target_chars: int, max_chars: int) -> list[Piece]:
    """Split ``text`` into pieces of at most ``max_chars`` characters.

    Paragraphs are merged until ``target_chars`` is reached, so a piece is roughly
    the size the caller asked for but never larger than ``max_chars``. A paragraph
    longer than ``max_chars`` is broken on line boundaries, and a line longer than
    that is cut at ``max_chars``.

    Text that parses as a JSON array is split one element per piece instead, which
    is what lets a 14-invoice ledger drop 12 invoices and keep 2. If parsing fails
    the text is split as ordinary text.

    Anchors are ``"part 1"``, ``"part 2"`` and so on, or ``"item 1"`` for the JSON
    case. They exist so a later decision can say which part of the original it was
    talking about.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not text.strip():
        return []

    json_pieces = _split_json_array(text, max_chars)
    if json_pieces is not None:
        return json_pieces

    return _numbered(
        _split_paragraphs(text, target_chars, max_chars), prefix="part"
    )


def clip_head_tail(text: str, head: int, tail: int) -> str:
    """Keep the start and the end of ``text``, marking what was removed.

    The head and tail are where a tool output carries its identity: the command
    that ran, and the result it produced. Short text is returned unchanged, so this
    never adds noise to something that did not need shortening.
    """
    if head < 0 or tail < 0:
        raise ValueError("head and tail must not be negative")
    if len(text) <= head + tail:
        return text
    marker = f"\n{CLIP_MARKER}\n"
    return f"{text[:head]}{marker}{text[-tail:] if tail else ''}"


def _split_json_array(text: str, max_chars: int) -> list[Piece] | None:
    """Return one piece per array element, or ``None`` when this is not an array.

    ``None`` means "not JSON", which is different from "parsed to an empty list":
    an empty array yields no pieces, and the caller is done either way.
    """
    stripped = text.strip()
    if not stripped.startswith("["):
        return None
    try:
        data = json.loads(stripped)
    except ValueError:
        return None
    if not isinstance(data, list):
        return None

    pieces: list[Piece] = []
    for number, entry in enumerate(data, start=1):
        dumped = json.dumps(entry, ensure_ascii=False)
        pieces.append(Piece(anchor=f"item {number}", text=_cap(dumped, max_chars)))
    return pieces


def _split_paragraphs(text: str, target_chars: int, max_chars: int) -> list[str]:
    """Merge paragraphs up to ``target_chars``, never exceeding ``max_chars``."""
    blocks = [block.strip() for block in _PARAGRAPH.split(text) if block.strip()]
    if not blocks:
        blocks = [text.strip()]

    pieces: list[str] = []
    buffer: list[str] = []
    used = 0

    for block in blocks:
        # An oversized paragraph arrives already broken into chunks, and each
        # chunk is treated as its own paragraph from here on.
        for chunk in _fit(block, max_chars):
            if used and used + len(chunk) > target_chars:
                pieces.append("\n\n".join(buffer))
                buffer, used = [], 0
            buffer.append(chunk)
            used += len(chunk) + 2

    if buffer:
        pieces.append("\n\n".join(buffer))

    # The merge step can still overshoot when a single chunk is large, so the
    # bound is applied once more to every finished piece.
    return [_cap(piece, max_chars) for piece in pieces]


def _fit(block: str, max_chars: int) -> list[str]:
    """Break one oversized block into chunks of at most ``max_chars``.

    Lines are kept whole where they fit, so a code file survives as readable lines.
    A single line longer than the limit is cut, because there is nothing finer to
    split it on.
    """
    if len(block) <= max_chars:
        return [block]

    chunks: list[str] = []
    buffer: list[str] = []
    used = 0
    for line in block.splitlines():
        while len(line) > max_chars:
            if buffer:
                chunks.append("\n".join(buffer))
                buffer, used = [], 0
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        if used and used + len(line) > max_chars:
            chunks.append("\n".join(buffer))
            buffer, used = [], 0
        buffer.append(line)
        used += len(line) + 1
    if buffer:
        chunks.append("\n".join(buffer))
    return chunks


def _cap(text: str, max_chars: int) -> str:
    """Return ``text`` unchanged, or cut to ``max_chars``."""
    return text if len(text) <= max_chars else text[:max_chars]


def _numbered(pieces: list[str], prefix: str) -> list[Piece]:
    """Attach ``part N`` anchors, dropping anything empty."""
    return [
        Piece(anchor=f"{prefix} {number}", text=piece)
        for number, piece in enumerate(pieces, start=1)
        if piece.strip()
    ]