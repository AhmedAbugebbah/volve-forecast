import numpy as np
import pytest

from forecast_utils import (
    discount_cashflows,
    npv_from_oil,
    percentile_forecast,
    regression_metrics,
    sm3_to_bbl,
)


def test_unit_conversion():
    assert sm3_to_bbl(1) == 6.2898


def test_metrics_include_mae_rmse_and_mape():
    result = regression_metrics([10, 20], [12, 18])
    assert result["mae"] == 2
    assert result["rmse"] == 2
    assert result["mape"] == pytest.approx(15)


def test_discounting_reduces_future_cashflow():
    assert discount_cashflows([100, 100], 0.0) == 200
    assert discount_cashflows([0, 100], 0.1) < 100


def test_npv_uses_price_less_opex():
    assert npv_from_oil([1], 70, 10, 0) == 60


def test_bootstrap_intervals_are_ordered_and_nonnegative():
    intervals = percentile_forecast([100, 100], [-10, 0, 10], samples=200)
    assert np.all(intervals["p10"] <= intervals["p50"])
    assert np.all(intervals["p50"] <= intervals["p90"])
    assert np.all(intervals["p10"] >= 0)
