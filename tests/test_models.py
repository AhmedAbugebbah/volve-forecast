import numpy as np
import pandas as pd
import importlib.util

spec = importlib.util.spec_from_file_location("arps_module", "02_arps.py")
arps_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arps_module)
arps = arps_module.arps


def test_arps_curve_declines_for_positive_parameters():
    values = arps(np.array([1.0, 2.0, 3.0]), 1000, 0.05, 0.7)
    assert values[0] > values[1] > values[2] > 0


def test_monthly_dates_aggregate_without_changing_month():
    dates = pd.to_datetime(["2014-04-01", "2014-04-15", "2014-05-02"])
    grouped = pd.DataFrame({"date": dates, "oil": [1, 2, 3]})
    result = grouped.groupby(grouped["date"].dt.to_period("M"))["oil"].sum()
    assert result.loc["2014-04"] == 3
    assert result.loc["2014-05"] == 3
