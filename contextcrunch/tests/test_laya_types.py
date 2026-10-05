"""Tests for the Laya answer types and the payload parser."""

from __future__ import annotations

from contextcrunch.core.questions import QUESTION_TYPES
from contextcrunch.laya.types import LayaResult, parse_result

#: A payload shaped exactly like the probe output in docs/prompts/result.txt.
PROBE_PAYLOAD: dict = {
    "model": "laya-rl-agent",
    "answers": {
        "verdict": {
            "type": "choice",
            "choice": "drop",
            "probabilities": {"keep": 0.1, "truncate": 0.1, "drop": 0.8},
            "confidence": 0.7,
            "answer_confidence": 0.8,
            "action": {"act_probability": 1.0},
        },
        "essential": {
            "type": "noul",
            "noul": 0.12,
            "confidence": 0.88,
            "answer_confidence": 0.88,
            "action": {"act_probability": 1.0},
        },
        "consumed": {
            "type": "noul",
            "noul": 0.95,
            "confidence": 0.95,
            "answer_confidence": 0.95,
            "action": {"act_probability": 1.0},
        },
        "superseded": {
            "type": "noul",
            "noul": 0.3,
            "answer_confidence": 0.7,
            "action": {"act_probability": 1.0},
        },
        "relevance": {
            "type": "score",
            "score": 0.6,
            "legend": {"0": "noise"},
            "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1, "3": 0.0},
            "answer_confidence": 0.7,
            "action": {"act_probability": 1.0},
        },
    },
    "usage": {"input_tokens": 289, "output_tokens": 0, "state_tokens": 9},
}


def test_parses_the_probe_shape() -> None:
    """The parser must read the real field names, not the ones the spec invented."""
    result = parse_result(PROBE_PAYLOAD, dict(QUESTION_TYPES))
    assert isinstance(result, LayaResult)
    assert result.model == "laya-rl-agent"
    assert result.answer("verdict").choice == "drop"
    assert result.noul("essential") == 0.12
    assert result.answer("relevance").score == 0.6
    assert result.usage.input_tokens == 289


def test_noul_lives_in_noul_not_probability() -> None:
    """A noul answer reports its probability in 'noul'; there is no 'probability'."""
    result = parse_result(PROBE_PAYLOAD, dict(QUESTION_TYPES))
    assert result.answer("consumed").noul == 0.95
    assert not hasattr(result.answer("consumed"), "probability")


def test_score_is_already_weighted() -> None:
    """'score' is the weighted expected level, so no manual summing is needed."""
    result = parse_result(PROBE_PAYLOAD, dict(QUESTION_TYPES))
    assert result.answer("relevance").score == 0.6


def test_answer_confidence_is_available() -> None:
    """The calibrated confidence is 'answer_confidence'."""
    result = parse_result(PROBE_PAYLOAD, dict(QUESTION_TYPES))
    assert result.answer("verdict").answer_confidence == 0.8
    assert result.answer("verdict").confidence == 0.7


def test_prob_helper_and_label() -> None:
    """Helpers must report probabilities and short labels."""
    result = parse_result(PROBE_PAYLOAD, dict(QUESTION_TYPES))
    assert result.answer("verdict").prob("drop") == 0.8
    assert result.answer("verdict").prob("missing") is None
    assert result.answer("essential").label() == "noul=0.12"


def test_unknown_keys_are_kept_not_dropped() -> None:
    """A Laya upgrade must not crash the parser."""
    payload = {"answers": {"verdict": {"type": "choice", "choice": "keep", "new_field": 7}}}
    result = parse_result(payload, dict(QUESTION_TYPES))
    assert result.answer("verdict").choice == "keep"


def test_missing_and_malformed_answers_are_tolerated() -> None:
    """Missing or non-dict answers must not raise."""
    result = parse_result({"answers": {"verdict": "not-a-dict"}}, dict(QUESTION_TYPES))
    assert result.answers == {}
    assert result.noul("essential") is None
    assert result.answer("verdict") is None


def test_act_probability_gate() -> None:
    """act_probability decides whether the agent should act on the answer."""
    result = parse_result(
        {
            "answers": {
                "verdict": {
                    "type": "choice",
                    "choice": "keep",
                    "action": {"act_probability": 0.0},
                }
            }
        },
        dict(QUESTION_TYPES),
    )
    assert result.answer("verdict").act_ok() is False
