import pytest

from conftest import CHEAP, STRONG
from modelrouter.costing import compute_cost, estimate_tokens, format_usd, total_cost
from modelrouter.types import Usage


def test_compute_cost_uses_per_million_prices():
    # 1M input at $1 + 0.5M output at $2 = $2
    assert compute_cost(Usage(1_000_000, 500_000), CHEAP) == pytest.approx(2.0)


def test_strong_costs_more_for_same_tokens():
    u = Usage(1000, 200)
    assert compute_cost(u, STRONG) > compute_cost(u, CHEAP)
    assert compute_cost(u, STRONG) == pytest.approx((1000 * 10 + 200 * 20) / 1e6)


def test_zero_usage_is_free():
    assert compute_cost(Usage(0, 0), STRONG) == 0


def test_total_cost_sums_calls():
    got = total_cost([(Usage(100, 10), CHEAP), (Usage(100, 10), STRONG)])
    assert got == pytest.approx(
        compute_cost(Usage(100, 10), CHEAP) + compute_cost(Usage(100, 10), STRONG)
    )


def test_estimate_tokens_is_positive_and_scales():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 400) == 100


def test_format_usd_handles_tiny_amounts():
    assert format_usd(0) == "$0"
    assert format_usd(0.00012) == "$0.00012"
    assert format_usd(12.5) == "$12.50"
