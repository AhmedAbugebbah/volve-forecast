"""
Step 4 - Economics: 5-year NPV per well using the winning forecast method.
Assumptions: $70/bbl oil, $12/bbl variable opex, 10% annual discount rate.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from forecast_utils import discount_cashflows, npv_from_oil, percentile_forecast

PRICE, OPEX, DISC = 70.0, 12.0, 0.10

def main():
    comp = pd.read_csv("output/comparison.csv").set_index("well")
    forecasts = pd.read_csv("output/ml_forecasts_monthly.csv")
    arps_params = pd.read_csv("output/arps_params.csv").set_index("well")

    rows = []
    for well, m in comp.iterrows():
        if m["winner"] == "ML":
            monthly_bbl = forecasts[f"{well}_p50"].values
            p10 = forecasts[f"{well}_p10"].values
            p90 = forecasts[f"{well}_p90"].values
        else:
            # rebuild the Arps monthly profile from its fitted parameters
            a = arps_params.loc[well]
            n_hist = int((pd.read_csv("data/monthly_production.csv")
                          .query("well == @well and active")["t"]).max())
            t = np.arange(n_hist + 1, n_hist + 61, dtype=float)
            monthly_bbl = a["qi"] / np.power(1 + a["b"] * a["Di"] * t, 1 / a["b"])
            residual = max(float(a.get("arps_rmse", 0)), 0)
            p10 = np.clip(monthly_bbl - 1.28 * residual, 0, None)
            p90 = monthly_bbl + 1.28 * residual

        eur = monthly_bbl.sum() / 1e3
        npv = npv_from_oil(monthly_bbl, PRICE, OPEX, DISC)
        rows.append(dict(well=well, method=m["winner"],
                         eur_5y_mbbl=eur, npv_musd=npv / 1e6,
                         npv_p10_musd=npv_from_oil(p10, PRICE, OPEX, DISC) / 1e6,
                         npv_p90_musd=npv_from_oil(p90, PRICE, OPEX, DISC) / 1e6))

    npv_df = pd.DataFrame(rows).sort_values("npv_musd", ascending=False)
    npv_df.to_csv("output/npv.csv", index=False)
    sensitivity = []
    for price in (50, 70, 90):
        for opex in (8, 12, 18):
            for well, m in comp.iterrows():
                q = forecasts[f"{well}_p50"].values if m["winner"] == "ML" else (
                    arps_params.loc[well, "qi"] /
                    np.power(1 + arps_params.loc[well, "b"] * arps_params.loc[well, "Di"] *
                             np.arange(1, 61), 1 / arps_params.loc[well, "b"]))
                sensitivity.append({"well": well, "price_usd_bbl": price,
                                    "opex_usd_bbl": opex,
                                    "npv_musd": npv_from_oil(q, price, opex, DISC) / 1e6})
    pd.DataFrame(sensitivity).to_csv("output/npv_sensitivity.csv", index=False)
    print(npv_df.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    total = npv_df["npv_musd"].sum()
    print(f"\nPortfolio simplified pre-tax 5y operating NPV: ${total:,.0f} MM")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#3d5afe" if w == "ML" else "#ff8f00" for w in npv_df["method"]]
    bars = ax.bar(npv_df["well"], npv_df["npv_musd"], color=colors, edgecolor="white")
    for b, v in zip(bars, npv_df["npv_musd"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"${v:,.0f}M",
                ha="center", fontsize=10, weight="bold")
    ax.set_title(f"5-Year NPV per well - Volve field "
                 f"($70/bbl, $12/bbl opex, 10% discount) | portfolio: ${total:,.0f} MM",
                 fontsize=11, weight="bold")
    ax.set_ylabel("NPV, million USD")
    ax.grid(axis="y", alpha=0.3)
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color="#3d5afe", label="ML forecast"),
                       mpatches.Patch(color="#ff8f00", label="Arps forecast")])
    fig.tight_layout()
    fig.savefig("figures/npv.png", dpi=130)
    print("saved -> output/npv.csv, figures/npv.png")

if __name__ == "__main__":
    main()
