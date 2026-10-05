"""Typed view of a Laya answer.

Every field here exists in a real Laya response. Nothing was invented. In
particular:

- a ``noul`` answer reports its probability in ``noul`` (there is no ``probability`` key);
- a ``score`` answer already returns the weighted expected level in ``score``;
- ``answer_confidence`` is the model's own confidence in the chosen option;
- ``action.act_probability`` says whether the agent should act on the answer.

Two backends answer, and they differ slightly. Both were verified against the
live services and neither field is invented:

===============  ==========================  ==================================
Field            local ``predict``           hosted ``/v1/systemone``
===============  ==========================  ==================================
``noul``         reported                    reported
``score``        reported                    reported
``confidence``   reported                    reported (entropy based)
``answer_conf``  reported                    **omitted**, derived
``action``       reported                    **omitted**
===============  ==========================  ==================================

:func:`derive_answer_confidence` reconstructs the one field the hosted API
leaves out, using the arithmetic the local probe output makes visible. Both
backend gaps are optional fields, so the rest of the parser is unaffected.

Unknown keys are kept as pydantic extras instead of being dropped, so a Laya
upgrade can never crash the parser.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AnswerType = Literal["choice", "score", "noul"]


class Action(BaseModel):
    """The ``action`` block of a Laya answer."""

    model_config = ConfigDict(extra="allow")

    act_probability: float | None = None


class Answer(BaseModel):
    """One question's answer, normalised across the three Laya question types."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: AnswerType
    choice: str | None = None
    noul: float | None = None
    score: float | None = None
    confidence: float | None = None
    answer_confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    legend: dict[str, str] = Field(default_factory=dict)
    action: Action | None = None

    def prob(self, option: str) -> float | None:
        """Return the probability of one option, or ``None`` when Laya did not report it."""
        return self.probabilities.get(option)

    def act_ok(self) -> bool:
        """Return True when Laya expects the agent to act on this answer."""
        if self.action is None or self.action.act_probability is None:
            return True
        return self.action.act_probability > 0.0

    def label(self) -> str:
        """Return a short human label for logs and side-car rows."""
        if self.type == "choice":
            return self.choice or "?"
        if self.type == "score":
            return f"score={self.score:.2f}" if self.score is not None else "score=?"
        return f"noul={self.noul:.2f}" if self.noul is not None else "noul=?"


class Usage(BaseModel):
    """Token accounting reported by Laya."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    output_tokens: int | None = None
    state_tokens: int | None = None
    state_tokens_dropped: int | None = None
    truncated: bool | None = None
    truncated_questions: list[str] = Field(default_factory=list)


class LayaResult(BaseModel):
    """The whole payload returned by one ``predict`` call."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    answers: dict[str, Answer] = Field(default_factory=dict)
    usage: Usage | None = None

    def answer(self, question_id: str) -> Answer | None:
        """Return one answer by id, or ``None`` when Laya skipped it."""
        return self.answers.get(question_id)

    def noul(self, question_id: str) -> float | None:
        """Return the noul probability for a question, or ``None``."""
        answer = self.answers.get(question_id)
        return None if answer is None else answer.noul


def parse_result(payload: dict[str, Any], question_ids: dict[str, str]) -> LayaResult:
    """Build a :class:`LayaResult` from a raw Laya payload.

    Laya does not echo the question id or type back in the payload, so the types
    are taken from ``question_ids`` and only guessed as a last resort.
    """
    raw_answers = payload.get("answers")
    answers: dict[str, Answer] = {}
    if isinstance(raw_answers, dict):
        for question_id, raw in raw_answers.items():
            if not isinstance(raw, dict):
                continue
            # Laya echoes "type" back, so it must not be passed twice.
            question_type = question_ids.get(question_id) or raw.get("type") or _guess_type(raw)
            answer = Answer.model_validate({"id": question_id, "type": question_type, **raw})
            if answer.answer_confidence is None:
                answer.answer_confidence = derive_answer_confidence(answer)
            answers[question_id] = answer
    raw_usage = payload.get("usage")
    return LayaResult(
        model=payload.get("model"),
        answers=answers,
        usage=Usage(**raw_usage) if isinstance(raw_usage, dict) else None,
    )


def derive_answer_confidence(answer: Answer) -> float | None:
    """Recover the probability Laya assigned to the answer it chose.

    The local model reports this as ``answer_confidence``; the hosted API omits it.
    Both mean the same thing, and the local probe output shows how it is built:
    the probability of the chosen option for a ``choice``, the largest level
    probability for a ``score``, and ``1 - noul`` for a ``noul``.

    This is arithmetic on fields the API already returned, not a guess at a new
    field. Without it the policy gate would have to fall back on ``confidence``,
    which is entropy-based and means something else entirely.
    """
    if answer.type == "noul":
        return None if answer.noul is None else 1.0 - answer.noul
    if answer.type == "choice":
        if answer.choice is None:
            return None
        return answer.probabilities.get(answer.choice)
    return max(answer.probabilities.values(), default=None)


def _guess_type(raw: dict[str, Any]) -> AnswerType:
    """Fall back to the type implied by the keys when it was not declared."""
    if "choice" in raw:
        return "choice"
    if "score" in raw:
        return "score"
    return "noul"
