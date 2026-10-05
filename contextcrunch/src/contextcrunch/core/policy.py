"""The policy engine: Laya proposes, code disposes.

Laya only produces probabilities. This module is what actually decides, and the
three profile tables below are the only numbers allowed in code.

Dropping content is the one irreversible thing ContextCrunch does, so a DROP has
to clear every gate at once. Each gate is a veto: if any single one fails, the
item is not dropped. The model never gets the final word on anything it cannot
take back.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextcrunch.laya.types import Answer

Action = str

KEEP: Action = "KEEP"
TRUNCATE: Action = "TRUNCATE"
DROP: Action = "DROP"

#: Roles that are pinned by policy and never judged.
PINNED_ROLES = ("system", "user", "assistant")

#: Below this many tokens an item is not worth a decision, whatever it contains.
MIN_DECISION_TOKENS = 40

#: Items above this token count are truncated even when Laya says keep.
TRUNCATE_ABOVE_TOKENS = 800


@dataclass(frozen=True)
class Profile:
    """Gates for one archetype. This row is the only thing that changes per domain."""

    #: An item is droppable only if its essential probability is below this ceiling.
    ess: float
    #: An item is droppable only if its relevance score is at most this ceiling.
    rel: float
    #: A probability at or above this counts as a junk reason.
    con: float
    #: An item is droppable only if Laya's confidence in the verdict exceeds this floor.
    conf: float
    #: A relevance score at or below this counts as a junk reason on its own.
    noise: float
    #: Number of most recent turns that are never judged.
    K: int
    #: Keep relevant sub-items verbatim with their source reference.
    anchors: bool


#: Profile thresholds. The only numbers allowed in code.
PROFILES: dict[str, Profile] = {
    "aggressive": Profile(
        ess=0.50, rel=1.5, con=0.80, conf=0.70, noise=0.8, K=2, anchors=False
    ),
    "balanced": Profile(
        ess=0.40, rel=1.0, con=0.85, conf=0.80, noise=0.6, K=4, anchors=False
    ),
    "conservative": Profile(
        ess=0.30, rel=1.0, con=0.90, conf=0.90, noise=0.5, K=3, anchors=True
    ),
}

#: Fallback used when an unknown profile name is requested.
DEFAULT_PROFILE: Profile = PROFILES["conservative"]


def get_profile(name: str) -> Profile:
    """Return a profile by name, falling back to the conservative one.

    A typo in the profile name must never open the door to aggressive drops, so an
    unrecognised name resolves to the safest row rather than raising.
    """
    return PROFILES.get(name, DEFAULT_PROFILE)


def decide(
    item: dict,
    a: dict[str, Answer],
    profile: Profile,
    pinned_tools: frozenset[str] = frozenset(),
    is_sub: bool = False,
) -> str:
    """Decide what happens to one history item.

    ``item`` carries ``role``, ``tokens``, ``age_in_turns`` and ``tool_name``.

    The rules, in the order they are applied:

    1. **Pinned roles stay.** ``system``, ``user`` and ``assistant`` messages are
       the conversation itself. A human turn cannot be re-issued, so it is never
       judged at all.
    2. **Pinned tools stay.** Some tools are configured as always worth keeping,
       whatever the model thinks about their output.
    3. **Recent turns stay.** Within the last ``K`` turns the agent is most likely
       to be working with that output right now, so it is kept without asking.
    4. **Tiny items stay.** Below 40 tokens there is no meaningful saving to be had.
    5. **DROP needs every gate.** Laya must recommend ``drop``, be confident about
       it, and agree the item is not essential and not relevant, *and* give at
       least one positive reason why the item is junk. See :func:`_is_droppable`.
    6. **TRUNCATE on a recommendation or on size.** Either Laya says ``truncate``,
       or the item is simply too big to keep whole.
    7. Otherwise **KEEP**, unchanged.

    Rules 3 and 4 are skipped when ``is_sub`` is true. A sub-item is a piece of an
    item that is already on trial, so its position and size in the original history
    mean nothing: a short passage inside a 14000-token filing is still worth
    judging on its own merits.

    A missing answer is a veto. If Laya did not answer, or answered with ``None``,
    the item is kept rather than guessed at.
    """
    if item["role"] in PINNED_ROLES:
        return KEEP
    if item.get("tool_name") in pinned_tools:
        return KEEP
    if not is_sub and item["age_in_turns"] <= profile.K:
        return KEEP
    if not is_sub and item["tokens"] < MIN_DECISION_TOKENS:
        return KEEP

    verdict = a["verdict"]
    if _is_droppable(a, verdict, profile):
        return DROP
    if verdict.choice == "truncate" or (not is_sub and item["tokens"] > TRUNCATE_ABOVE_TOKENS):
        return TRUNCATE
    return KEEP


def _is_droppable(a: dict[str, Answer], verdict: Answer, profile: Profile) -> bool:
    """Return True only when every drop gate passes.

    Every gate here is a **veto**: a single failure blocks the drop, and the item
    is kept instead. That is the whole safety argument of the system. A wrong drop
    destroys evidence the agent may need, so the default under any doubt is to keep.

    The four gates are:

    - Laya recommended ``drop``;
    - Laya is confident about that recommendation, above ``conf``;
    - the item is not ``essential`` (below ``ess``) and not ``relevant``
      (at most ``rel``), meaning it cannot be re-obtained if it goes;
    - there is at least one **junk reason**, see :func:`_junk_reason`.

    A veto is not the same as a KEEP. The item may still end up truncated by the
    size rule below, which is reversible in a way that dropping is not.
    """
    if verdict.choice != "drop":
        return False
    confidence = _confidence(verdict)
    if confidence is None or not confidence > profile.conf:
        return False
    if not _below(_probability(a["essential"]), profile.ess):
        return False
    if not _at_most(_score(a["relevance"]), profile.rel):
        return False
    return _junk_reason(a, profile)


def _junk_reason(a: dict[str, Answer], profile: Profile) -> bool:
    """Return True when Laya gives at least one positive reason to call this junk.

    Three independent reasons count, because agents produce junk in three distinct
    ways and any one of them is enough:

    - **consumed**: the useful part was already extracted into a later finding, so
      the original is only keeping tokens alive;
    - **superseded**: a later item made this one obsolete, for example a newer test
      run that passed;
    - **noise**: the item is so irrelevant to the goal that it is background at best.

    Requiring only one reason keeps the gate from being so demanding that it never
    fires. The other three gates in :func:`_is_droppable` are what stop it firing
    too eagerly.
    """
    if _at_least(_probability(a["consumed"]), profile.con):
        return True
    if _at_least(_probability(a["superseded"]), profile.con):
        return True
    return _at_most(_score(a["relevance"]), profile.noise)


def _probability(answer: Answer | None) -> float | None:
    """Return the noul probability for an answer, or ``None`` when unavailable.

    Laya reports a ``noul`` answer in ``noul``. The ``probability`` alias is
    accepted too, because it is the name the reference logic uses and a caller may
    hand over an object shaped either way.
    """
    if answer is None:
        return None
    value = answer.noul
    if value is None:
        value = getattr(answer, "probability", None)
    return None if value is None else float(value)


def _score(answer: Answer | None) -> float | None:
    """Return the weighted level for a ``score`` answer, or ``None``."""
    if answer is None:
        return None
    value = answer.score
    return None if value is None else float(value)


def _confidence(verdict: Answer) -> float | None:
    """Return Laya's confidence in its chosen verdict.

    ``answer_confidence`` is the probability of the option it picked, which is what
    a gate should be measured against. It is preferred over ``confidence``, which is
    entropy-based and means something different. The hosted API omits the first, in
    which case it has already been reconstructed during parsing.
    """
    value = verdict.answer_confidence
    if value is None:
        value = verdict.confidence
    return None if value is None else float(value)


def _below(value: float | None, ceiling: float) -> bool:
    """Return True when ``value`` is known and strictly below ``ceiling``."""
    return value is not None and value < ceiling


def _at_most(value: float | None, ceiling: float) -> bool:
    """Return True when ``value`` is known and at most ``ceiling``."""
    return value is not None and value <= ceiling


def _at_least(value: float | None, floor: float) -> bool:
    """Return True when ``value`` is known and at least ``floor``."""
    return value is not None and value >= floor