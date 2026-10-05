"""Tests for the Settings module and the no-defaults rule."""

from __future__ import annotations

import pytest

from contextcrunch.core.settings import EMERGENCY_DEFAULTS, Settings, get_settings


def test_reads_every_tunable_from_the_env() -> None:
    """Settings must pick up the CC_ prefixed variables."""
    settings = get_settings()
    assert settings.laya_mode == "fake"
    assert settings.default_profile == "conservative"
    assert settings.trigger_tokens == 200


def test_emergency_defaults_cover_every_tunable() -> None:
    """The emergency table must have an entry for each field, or fail-open breaks."""
    fields = set(Settings.model_fields)
    assert fields == {key.removeprefix("CC_").lower() for key in EMERGENCY_DEFAULTS}


def test_tunables_have_no_defaults() -> None:
    """No tunable may ship a default: everything comes from .env.

    ``laya_api_key`` is the one exemption. It is a secret, not a tuning number,
    and defaulting it would risk shipping a placeholder credential.
    """
    for name, field in Settings.model_fields.items():
        if name in ("model_config", "laya_api_key"):
            continue
        assert field.is_required(), f"{name} must be required"


def test_api_key_is_never_defaulted() -> None:
    """The secret must stay empty when absent, never fall back to a value."""
    field = Settings.model_fields["laya_api_key"]
    assert field.default == ""
    assert "laya_api_key" not in EMERGENCY_DEFAULTS


def test_api_key_is_hidden_from_repr() -> None:
    """A Settings repr must never leak the bearer token into a log line."""
    settings = get_settings()
    assert "test-key" not in repr(settings)


def test_missing_config_warns_and_degrades(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """A missing CC_ variable must warn on stderr, not crash the process.

    A real .env sits in the working tree, so it is pointed away for this test;
    otherwise it would supply the values and the warning would never fire.
    """
    for key in list(EMERGENCY_DEFAULTS):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    from contextcrunch.core.settings import reset_settings_cache

    reset_settings_cache()
    settings = get_settings()
    captured = capsys.readouterr()

    assert "incomplete configuration" in captured.err
    assert settings.laya_mode == "fake"
    # The emergency value, not the test fixture's value, since .env is off.
    assert settings.trigger_tokens == int(EMERGENCY_DEFAULTS["CC_TRIGGER_TOKENS"])


def test_unknown_extra_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrelated variable in .env must not break startup."""
    monkeypatch.setenv("SOMETHING_ELSE", "1")
    assert get_settings().laya_concurrency == 4
