"""
Step 3 - Machine learning forecast: per-well gradient boosting with
walk-forward (one-step-ahead) validation on the SAME split as Arps.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from forecast_utils import percentile_forecast, regression_metrics

DATA = "data/monthly_production.csv"
ARPS = "output/arps_params.csv"
OUT = "output/comparison.csv"
FIG = "figures/ml_vs_arps.png"
HORIZON = 60  # months for the NPV forecast


def make_features(g):
    g = g.copy()
    g["oil_lag1"] = g["bbl"].shift(1)
    g["oil_lag2"] = g["bbl"].shift(2)
    g["oil_lag3"] = g["bbl"].shift(3)
    g["cum_bbl"] = g["bbl"].cumsum() - g["bbl"]
    g["onstream_lag1"] = g["on_stream_h"].shift(1)
    g["press_lag1"] = g["avg_pressure"].shift(1)
    g["choke_lag1"] = g["avg_choke"].shift(1)
    return g


FEATURES = ["t", "oil_lag1", "oil_lag2", "oil_lag3", "cum_bbl",
            "onstream_lag1", "press_lag1", "choke_lag1"]


def new_model():
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_depth=3,
        min_samples_leaf=5, l2_regularization=1.0, random_state=42)


def walk_forward(g, split):
    """One-step-ahead predictions for the validation window."""
    g = make_features(g)
    preds = np.full(len(g), np.nan)
    for i in range(split, len(g)):
        train = g.iloc[:i]
        model = new_model()
        model.fit(train[FEATURES], train["bbl"])
        preds[i] = model.predict(g.iloc[[i]][FEATURES])[0]
    return g, preds


def forecast_future(g, model, months=HORIZON):
    """Iterative multi-step forecast feeding predictions back as lags."""
    g = make_features(g).iloc[-1:].copy()
    future, rows = [], g.copy()
    last_onstream = g["on_stream_h"].iloc[-3:].mean()
    last_press = g["press_lag1"].iloc[-1] if pd.notna(g["press_lag1"].iloc[-1]) else 0
    last_choke = g["choke_lag1"].iloc[-1] if pd.notna(g["choke_lag1"].iloc[-1]) else 0
    t0 = int(g["t"].iloc[-1])
    for step in range(1, months + 1):
        row = g.iloc[-1].copy()
        row["t"] = t0 + step
        row["oil_lag1"], row["oil_lag2"], row["oil_lag3"] = (
            future[-1] if future else g["bbl"].iloc[-1],
            future[-2] if len(future) >= 2 else g["bbl"].iloc[-2:-1].sum(),
            future[-3] if len(future) >= 3 else g["bbl"].iloc[-3:-1].sum())
        row["cum_bbl"] = g["cum_bbl"].iloc[-1] + sum(future)
        row["onstream_lag1"], row["press_lag1"], row["choke_lag1"] = (
            last_onstream, last_press, last_choke)
        yhat = float(model.predict(pd.DataFrame([row[FEATURES]]))[0])
        yhat = max(yhat, 0.0)
        future.append(yhat)
    return np.array(future)


def main():
    df = pd.read_csv(DATA)
    arps = pd.read_csv(ARPS).set_index("well")

    rows, fc = [], {}
    for well, g in df.groupby("well"):
        g = g[g["active"]].reset_index(drop=True)
        selection_end = max(5, int(len(g) * 0.6))
        test_start = max(selection_end + 1, int(len(g) * 0.8))
        g_feat, preds = walk_forward(g, selection_end)
        actual = g_feat["bbl"].values
        selection_metrics = regression_metrics(actual[selection_end:test_start],
                                                preds[selection_end:test_start])
        # The final test is evaluated once, after model choice is fixed.
        test_model = new_model()
        test_model.fit(g_feat.iloc[:test_start][FEATURES],
                       g_feat.iloc[:test_start]["bbl"])
        test_pred = test_model.predict(g_feat.iloc[test_start:][FEATURES])
        test_metrics = regression_metrics(actual[test_start:], test_pred)
        naive_pred = actual[test_start - 1:-1]
        moving_pred = np.array([np.mean(actual[max(0, i - 3):i])
                                for i in range(test_start, len(actual))])
        naive_metrics = regression_metrics(actual[test_start:], naive_pred)
        moving_metrics = regression_metrics(actual[test_start:], moving_pred)

        # Final forecast model can use all observed history (not test labels).
        model = new_model()
        model.fit(g_feat[FEATURES], g_feat["bbl"])
        future = forecast_future(g_feat, model)
        interval = percentile_forecast(future, actual[test_start:] - test_pred)

        rows.append(dict(well=well, ml_selection_mae=selection_metrics["mae"],
                         ml_selection_rmse=selection_metrics["rmse"],
                         ml_selection_mape=selection_metrics["mape"],
                         ml_mae=test_metrics["mae"], ml_rmse=test_metrics["rmse"],
                         ml_mape=test_metrics["mape"],
                         arps_selection_mape=arps.loc[well, "arps_selection_mape"],
                         arps_selection_mae=arps.loc[well, "arps_selection_mae"],
                         arps_selection_rmse=arps.loc[well, "arps_selection_rmse"],
                         naive_mae=naive_metrics["mae"], naive_rmse=naive_metrics["rmse"],
                         naive_mape=naive_metrics["mape"],
                         moving_avg_mae=moving_metrics["mae"],
                         moving_avg_rmse=moving_metrics["rmse"],
                         moving_avg_mape=moving_metrics["mape"],
                         arps_mape=arps.loc[well, "arps_mape"],
                         arps_mae=arps.loc[well, "arps_mae"],
                         arps_rmse=arps.loc[well, "arps_rmse"],
                         ml_eur_5y_mbbl=future.sum() / 1e3,
                         arps_eur_5y_mbbl=arps.loc[well, "eur_5y_mbbl"]))
        fc[well] = dict(g=g_feat, preds=preds, test_pred=test_pred,
                        split=test_start, selection_end=selection_end, future=future,
                        p10=interval["p10"], p50=interval["p50"], p90=interval["p90"],
                        residuals=actual[test_start:] - test_pred)
        print(f"{well:12s} selection ML MAPE {selection_metrics['mape']:6.1f}% | "
              f"untouched test ML MAPE {test_metrics['mape']:6.1f}%")

    comp = pd.DataFrame(rows)
    # Winner is selected only on the model-selection window.
    comp["winner"] = np.where(comp["ml_selection_mape"] < comp["arps_selection_mape"],
                              "ML", "Arps")
    comp.to_csv(OUT, index=False)

    # save the actual monthly ML forecasts (needed for NPV cashflows)
    fc_df = pd.DataFrame({
        **{f"{w}_{band}": r[band] for w, r in fc.items()
           for band in ("p10", "p50", "p90")},
        **{w: r["future"] for w, r in fc.items()},
    })
    fc_df.to_csv("output/ml_forecasts_monthly.csv", index=False)
    print("\n", comp.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    # figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Gradient Boosting vs Arps - selection period and untouched final test",
                 fontsize=13, weight="bold")
    for ax, (well, r) in zip(axes.flat, fc.items()):
        g, split = r["g"], r["split"]
        t, q = g["t"].values, g["bbl"].values
        ax.plot(t[:r["selection_end"]], q[:r["selection_end"]], "o-", ms=3, color="#1f77b4", label="fit")
        ax.plot(t[r["selection_end"]:split], q[r["selection_end"]:split], "o-", ms=3, color="#ffbf69", label="selection")
        ax.plot(t[split:], q[split:], "o-", ms=3, color="#ff7f0e", label="untouched test")
        ax.plot(t[split:], r["test_pred"], "-", color="green", lw=1.8, label="ML test pred")
        ax.axvspan(t[split], t[-1], color="orange", alpha=0.08)
        m = comp.set_index("well").loc[well]
        ax.set_title(f"{well}  ML {m['ml_mape']:.0f}% vs Arps {m['arps_mape']:.0f}%",
                     fontsize=10)
        ax.set_xlabel("months online"); ax.set_ylabel("oil, bbl/month")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    print(f"\nsaved -> {OUT}, {FIG}")


if __name__ == "__main__":
    main()
