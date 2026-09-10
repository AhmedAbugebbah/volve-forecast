"""
Step 2 - Arps decline curve analysis.
Hyperbolic model: q(t) = qi / (1 + b*Di*t)^(1/b)
Fit on first 70% of each well's history, validate on the last 30%.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from forecast_utils import regression_metrics

DATA = "data/monthly_production.csv"
OUT_PARAMS = "output/arps_params.csv"
FIG = "figures/arps_fits.png"


def arps(t, qi, Di, b):
    return qi / np.power(1.0 + b * Di * t, 1.0 / b)


def fit_curve(g, end):
    train = g.iloc[:end]
    p0 = [max(train["bbl"].iloc[0], 1.0), 0.05, 0.5]
    bounds = ([1e2, 1e-4, 0.05], [5e6, 0.9, 1.2])
    return curve_fit(arps, train["t"].values.astype(float),
                     train["bbl"].values, p0=p0, bounds=bounds, maxfev=20000)[0]


def fit_well(g):
    g = g[g["active"]].reset_index(drop=True)
    n = len(g)
    selection_end = max(5, int(n * 0.6))
    test_start = max(selection_end + 1, int(n * 0.8))
    t, q = g["t"].values.astype(float), g["bbl"].values
    try:
        selection_params = fit_curve(g, selection_end)
        final_params = fit_curve(g, test_start)
    except (ValueError, RuntimeError):
        return None

    selection_pred = arps(t[selection_end:test_start], *selection_params)
    test_pred = arps(t[test_start:], *final_params)
    selection_metrics = regression_metrics(q[selection_end:test_start], selection_pred)
    test_metrics = regression_metrics(q[test_start:], test_pred)

    # The final curve is refit only through the end of the selection period.
    t_future = np.arange(n + 1, n + 61)
    eur_5y = np.trapezoid(arps(t_future, *final_params), t_future) / 1e3
    pred = arps(t, *final_params)

    return dict(well=g["well"].iloc[0], qi=final_params[0], Di=final_params[1],
                b=final_params[2], train_months=test_start,
                selection_months=test_start - selection_end, test_months=n - test_start,
                arps_selection_mae=selection_metrics["mae"],
                arps_selection_rmse=selection_metrics["rmse"],
                arps_selection_mape=selection_metrics["mape"],
                arps_mae=test_metrics["mae"], arps_rmse=test_metrics["rmse"],
                arps_mape=test_metrics["mape"], eur_5y_mbbl=eur_5y,
                pred=pred, t=t, q=q, split=test_start, selection_end=selection_end,
                residuals=q[test_start:] - test_pred, well_df=g)


def main():
    df = pd.read_csv(DATA)
    results, curves = [], {}

    for well, g in df.groupby("well"):
        r = fit_well(g)
        if r:
            results.append({k: v for k, v in r.items()
                            if not isinstance(v, (np.ndarray, pd.DataFrame))})
            curves[well] = r

    res = pd.DataFrame(results).sort_values("arps_mape")
    res.to_csv(OUT_PARAMS, index=False)
    print(res[["well", "qi", "Di", "b", "arps_mape", "arps_rmse", "eur_5y_mbbl"]]
          .to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # figure: 2x3 grid of fits
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Arps Hyperbolic Decline Fits - selection and untouched test periods",
                 fontsize=13, weight="bold")
    for ax, (well, r) in zip(axes.flat, curves.items()):
        t, q, pred, split = r["t"], r["q"], r["pred"], r["split"]
        ax.plot(t[:r["selection_end"]], q[:r["selection_end"]], "o-", ms=3, color="#1f77b4", label="fit")
        ax.plot(t[r["selection_end"]:split], q[r["selection_end"]:split], "o-", ms=3, color="#ffbf69", label="selection")
        ax.plot(t[split:], q[split:], "o-", ms=3, color="#ff7f0e", label="untouched test")
        ax.plot(t, pred, "-", color="crimson", lw=1.8, label="Arps fit")
        ax.axvspan(t[split], t[-1], color="orange", alpha=0.08)
        ax.set_title(f"{well}  (MAPE {r['arps_mape']:.1f}%)", fontsize=10)
        ax.set_xlabel("months online"); ax.set_ylabel("oil, bbl/month")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    print(f"\nsaved -> {OUT_PARAMS}, {FIG}")


if __name__ == "__main__":
    main()
