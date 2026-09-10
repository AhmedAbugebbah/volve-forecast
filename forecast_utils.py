"""Shared, small building blocks for the Volve forecasting prototype."""

from __future__ import annotations

import numpy as np

SM3_TO_BBL = 6.2898


def sm3_to_bbl(value):
    """Convert standard cubic metres of oil to stock-tank barrels."""
    return np.asarray(value) * SM3_TO_BBL


def regression_metrics(actual, predicted):
    """Return scale-dependent errors and MAPE (with zero actuals excluded)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    nonzero = actual != 0
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100)
        if nonzero.any()
        else float("nan"),
    }


def discount_cashflows(cashflows, annual_rate):
    """Discount monthly cashflows at an effective annual discount rate."""
    cashflows = np.asarray(cashflows, dtype=float)
    periods = np.arange(1, len(cashflows) + 1) / 12.0
    return float(np.sum(cashflows / (1 + annual_rate) ** periods))


def npv_from_oil(oil_bbl, price, opex, annual_discount):
    """Simplified pre-tax operating NPV from monthly oil volumes."""
    return discount_cashflows(np.asarray(oil_bbl) * (price - opex), annual_discount)


def percentile_forecast(forecast, residuals, rng=None, samples=1000):
    """Residual bootstrap intervals as P10/P50/P90 arrays."""
    rng = np.random.default_rng(42) if rng is None else rng
    forecast = np.asarray(forecast, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size == 0:
        residuals = np.array([0.0])
    draws = rng.choice(residuals, size=(samples, len(forecast)), replace=True)
    draws = np.clip(forecast[None, :] + draws, 0, None)
    return {
        "p10": np.percentile(draws, 10, axis=0),
        "p50": np.percentile(draws, 50, axis=0),
        "p90": np.percentile(draws, 90, axis=0),
    }
