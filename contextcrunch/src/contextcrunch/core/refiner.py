"""Second pass at finer granularity.

When an item is judged TRUNCATE we do not blindly clip it. We split it into
pieces, judge each piece with the **same five questions**, and rebuild the item
from whatever survives.

With ``anchors=True`` a relevant piece is kept verbatim and prefixed with its
source reference, so a memo can still cite "p. 22" after 94% of the filing is
gone. Without anchors a kept piece is kept as-is and a dropped piece becomes a
tombstone line.

The splitting itself lives in :mod:`contextcrunch.core.splitter`. It is
deterministic, so the same item always breaks the same way.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Sequence
from typing import Any

from contextcrunch.core.messages import Message
from contextcrunch.core.policy import DROP, KEEP, TRUNCATE, Profile, decide
from contextcrunch.core.settings import get_settings
from contextcrunch.core.splitter import Piece, clip_head_tail, split_text
from contextcrunch.core.state import build_state
from contextcrunch.core.tokens import count_tokens
from contextcrunch.laya.client import LayaClient
from contextcrunch.laya.types import LayaResult

#: Separator between the surviving pieces of one rebuilt item.
PIECE_SEPARATOR = "\n---\n"

_PAGE = re.compile(r"\b(?:p(?:age)?\.?\s*\.?\s*|#\s*page\s*)\d+\b", re.IGNORECASE)


def anchor_of(text: str) -> str:
    """Return a citation anchor found in ``text``, such as ``p. 22``."""
    match = _PAGE.search(text)
    return match.group(0).replace("page", "p.") if match else ""


async def refine(
    *,
    client: LayaClient,
    messages: list[Message],
    index: int,
    goal: str,
    profile: Profile,
    on_piece: Callable[[str, str, LayaResult], None] | None = None,
) -> str:
    """Rebuild one item from the pieces that survive judgement.

    ``on_piece`` is called for every piece with its action, which is how the
    side-car records what happened inside a truncated item.
    """
    settings = get_settings()
    content = messages[index].content
    pieces = split_text(content, settings.piece_target_chars, settings.piece_max_chars)
    if not pieces:
        return content

    states = [
        build_state(messages, index, goal, settings, content=piece.text) for piece in pieces
    ]
    labels = [f"piece:{index}:{piece.anchor}" for piece in pieces]
    results = await gather_answers(client, states, labels)

    kept: list[str] = []
    shortened = False
    for piece, answer in zip(pieces, results, strict=True):
        # A piece is judged with is_sub=True: its size and position inside the
        # parent item say nothing about the value of the passage itself.
        action = decide(
            {
                "role": messages[index].role,
                "tokens": count_tokens(piece.text),
                "age_in_turns": len(messages) - index,
                "tool_name": messages[index].name,
            },
            answer.answers,
            profile,
            is_sub=True,
        )
        if action != KEEP:
            shortened = True
        if on_piece is not None:
            on_piece(piece.anchor, action, answer)
        rendered = _render_piece(piece, action, profile=profile)
        if rendered:
            kept.append(rendered)

    # Splitting and rejoining is not free: the separator between pieces and the
    # per-piece clip markers can outweigh the text that was actually removed. An
    # item the model judged worth keeping must come back byte for byte, and a
    # truncation must always be shorter than what it started as, so both cases
    # return the original rather than a rejoin that costs more than it saves.
    if not shortened:
        return content

    rebuilt = PIECE_SEPARATOR.join(kept)
    return rebuilt if len(rebuilt) < len(content) else content


def _render_piece(piece: Piece, action: str, *, profile: Profile) -> str:
    """Turn one judged piece into the text that goes back into the history.

    Kept text is always original. The only thing ever added is a source comment,
    and only for the conservative profile, where a later memo may need to cite
    where a passage came from.
    """
    if action == DROP:
        return f"[{piece.anchor} removed by ContextCrunch]"

    if action == TRUNCATE:
        head = max(get_settings().piece_max_chars // 4, 100)
        tail = max(get_settings().piece_max_chars // 8, 50)
        body = clip_head_tail(piece.text, head, tail)
        return _with_anchor(body, piece, profile)

    return _with_anchor(piece.text, piece, profile)


def _with_anchor(body: str, piece: Piece, profile: Profile) -> str:
    """Prefix ``body`` with its source reference when the profile keeps anchors."""
    if not profile.anchors:
        return body
    reference = f"{piece.anchor} {anchor_of(piece.text)}".strip()
    return f"<!-- {reference} -->\n{body}"


async def gather_answers(
    client: LayaClient, states: Sequence[dict[str, Any]], labels: Sequence[str]
) -> list[LayaResult]:
    """Run Laya over ``states`` with a bounded number of concurrent calls.

    The bound comes from ``CC_LAYA_CONCURRENCY`` so a long history cannot open an
    unbounded number of model calls at once.
    """
    settings = get_settings()
    semaphore = asyncio.Semaphore(max(settings.laya_concurrency, 1))

    async def run(state: dict[str, Any], label: str) -> LayaResult:
        async with semaphore:
            return await client.predict(state, label)

    paired = zip(states, labels, strict=True)
    return list(await asyncio.gather(*(run(state, label) for state, label in paired)))