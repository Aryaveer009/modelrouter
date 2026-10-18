import pytest

from modelrouter.config import ConfigError, Settings


def test_defaults_and_overrides():
    s = Settings.from_env(
        {
            "CHEAP_MODEL": "a",
            "STRONG_MODEL": "b",
            "CHEAP_PRICE_IN_PER_M": "0.3",
            "ROUTER_CONFIDENCE_THRESHOLD": "0.7",
        },
        use_dotenv=False,
    )
    assert s.cheap.name == "a" and s.cheap.price_in_per_m == 0.3
    assert s.confidence_threshold == 0.7
    assert s.fallback_enabled is True and s.min_avg_logprob is None


def test_validate_live_lists_missing_vars():
    s = Settings.from_env({}, use_dotenv=False)
    with pytest.raises(ConfigError) as e:
        s.validate_live()
    assert "CHEAP_API_KEY" in str(e.value) and "STRONG_MODEL" in str(e.value)


def test_mock_mode_needs_no_credentials():
    s = Settings.from_env({"MODELROUTER_MOCK": "1"}, use_dotenv=False)
    s.validate_live()  # does not raise
    assert s.mock and s.cheap.name == "mock-cheap"


@pytest.mark.parametrize(
    "env",
    [
        {"ROUTER_CONFIDENCE_THRESHOLD": "1.5"},
        {"ROUTER_CLASSIFIER": "magic"},
        {"CHEAP_PRICE_IN_PER_M": "abc"},
    ],
)
def test_invalid_values_rejected(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env, use_dotenv=False)
