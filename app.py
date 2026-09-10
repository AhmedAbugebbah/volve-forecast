"""
Streamlit dashboard: Arps vs ML production forecasting with NPV economics.
Visual system: petroleum report / well-log strip-chart aesthetic.
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Volve Field Lab", page_icon="🛢️", layout="wide")

HORIZON = 60
PRICE_DEF, OPEX_DEF, DISC_DEF = 70.0, 12.0, 10.0

# palette
SHALE, CORE, SAND = "#1A1C1F", "#222529", "#E8E4DA"
AMBER, TEAL, RED = "#D9822B", "#3A9B9B", "#C4453C"
STEEL, GRID, RULE = "#8A939E", "#2A2E33", "#3A4046"
MONO = "'Consolas','JetBrains Mono',ui-monospace,monospace"


@st.cache_data
def load():
    monthly = pd.read_csv("data/monthly_production.csv", parse_dates=["date"])
    arps = pd.read_csv("output/arps_params.csv")
    comp = pd.read_csv("output/comparison.csv")
    npv_base = pd.read_csv("output/npv.csv")
    forecasts = pd.read_csv("output/ml_forecasts_monthly.csv")
    sensitivity = pd.read_csv("output/npv_sensitivity.csv")
    return monthly, arps, comp, npv_base, forecasts, sensitivity


monthly, arps_params, comp, npv_base, forecasts, sensitivity = load()
WELLS = sorted(comp["well"].unique())

# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown(
        f'<div class="mh-title" style="font-size:1rem">VOLVE FIELD LAB</div>'
        f'<div style="font-family:{MONO};font-size:.7rem;color:{STEEL};">'
        f'FORECAST CONSOLE</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**Economic assumptions**")
    st.caption("Drag them — every NPV number on this page updates live.")
    price = st.slider("Oil price, $/bbl", 30.0, 130.0, PRICE_DEF, step=1.0)
    opex = st.slider("Opex, $/bbl", 4.0, 30.0, OPEX_DEF, step=0.5)
    disc_pct = st.slider("Discount rate, %/yr", 4.0, 18.0, DISC_DEF, step=0.5)
    st.divider()
    st.markdown(f'<span style="font-family:{MONO};font-size:.72rem;color:{STEEL};">'
                f'DATA · Equinor Volve open dataset<br>'
                f'FIELD · Norwegian North Sea, 2008-2016<br>'
                f'STACK · Python · pandas · scikit-learn · SciPy</span>',
                unsafe_allow_html=True)


# ---------------- economics engine ----------------
def well_forecast(well):
    m = comp.set_index("well").loc[well]
    g = monthly[(monthly["well"] == well) & (monthly["active"])]
    n_hist = int(g["t"].max())
    t = np.arange(n_hist + 1, n_hist + HORIZON + 1, dtype=float)
    if m["winner"] == "ML":
        q = forecasts[well].values.astype(float)
    else:
        a = arps_params.set_index("well").loc[well]
        q = a["qi"] / np.power(1 + a["b"] * a["Di"] * t, 1 / a["b"])
    return g, n_hist, t, q, m


def npv_of(q):
    net = q * (price - opex)
    disc = (1 + disc_pct / 100.0) ** (np.arange(1, HORIZON + 1) / 12.0)
    return float((net / disc).sum())


npv_live = pd.DataFrame(
    [{"well": w, "npv_musd": npv_of(well_forecast(w)[3]) / 1e6} for w in WELLS]
)
portfolio_npv = npv_live["npv_musd"].sum()
avg_ml, avg_arps = comp["ml_mape"].mean(), comp["arps_mape"].mean()

# ---------------- CSS ----------------
st.markdown(
    f"""
    <style>
      .block-container {{padding-top: 1.2rem; max-width: 1240px;}}
      #MainMenu, footer {{visibility: hidden;}}
      * {{transition: none !important; animation: none !important;}}
      :focus-visible {{outline: 2px solid {AMBER}; outline-offset: 2px;}}

      .masthead {{border-bottom: 3px double {AMBER}; padding: 12px 4px 10px;
                  display: flex; justify-content: space-between; align-items: flex-end;
                  gap: 16px; flex-wrap: wrap; margin-bottom: 14px;}}
      .mh-title {{font-size: 1.22rem; letter-spacing: .16em; font-weight: 700;
                  color: {SAND};}}
      .mh-meta {{font-family: {MONO}; font-size: .72rem; color: {STEEL};
                 text-align: right; line-height: 1.6;}}

      .metric {{border-left: 2px solid {RULE}; padding: 4px 16px 2px;}}
      .m-label {{font-size: .66rem; letter-spacing: .12em; text-transform: uppercase;
                 color: {STEEL};}}
      .m-value {{font-family: {MONO}; font-size: 1.5rem; color: {SAND};
                 font-variant-numeric: tabular-nums; margin-top: 2px;}}
      .m-sub {{font-family: {MONO}; font-size: .72rem; color: {STEEL}; margin-top: 2px;}}

      div[data-baseweb="tab-list"] {{gap: 26px; border-bottom: 1px solid {GRID};}}
      [data-baseweb="tab"] {{background: none;}}
      [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{display: none;}}
      [data-baseweb="tab"] p {{font-size: .76rem; letter-spacing: .14em;
          text-transform: uppercase; color: {STEEL}; font-weight: 600;}}
      [data-baseweb="tab"][aria-selected="true"] p {{color: {AMBER};}}
      [data-baseweb="tab"][aria-selected="true"] {{border-bottom: 2px solid {AMBER};}}

      .report-table {{width: 100%; border-collapse: collapse;
                      font-family: {MONO}; font-size: .8rem;}}
      .report-table th {{text-transform: uppercase; letter-spacing: .08em;
          font-size: .66rem; font-weight: 600; color: {STEEL}; text-align: left;
          border-bottom: 2px solid {AMBER}; padding: 7px 10px;}}
      .report-table td {{padding: 6px 10px; border-bottom: 1px solid {GRID};
          color: {SAND}; font-variant-numeric: tabular-nums;}}
      .report-table tr:hover td {{background: {CORE};}}
      .win-ml {{color: {AMBER}; font-weight: 700;}}
      .win-arps {{color: {TEAL}; font-weight: 700;}}

      .note {{border-left: 2px solid {AMBER}; background: {CORE};
              padding: 10px 14px; color: #C9CED6; font-size: .88rem;}}
      div[data-testid="stDataFrame"] {{border: none;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def metric(label, value, sub=""):
    subh = f'<div class="m-sub">{sub}</div>' if sub else ""
    return (f'<div class="metric"><div class="m-label">{label}</div>'
            f'<div class="m-value">{value}</div>{subh}</div>')


def report_table(df, formatters):
    return st.markdown(
        df.to_html(index=False, formatters=formatters, classes="report-table",
                   border=0, escape=False),
        unsafe_allow_html=True)


AX = dict(labelFont="Consolas", labelColor=STEEL, labelFontSize=11,
          titleFont="Segoe UI", titleColor="#C9CED6", titleFontSize=11,
          gridColor=GRID, gridDash=[1, 3], domain=False, tickColor=RULE)


def theme(chart):
    return chart.configure_axis(**AX).configure_view(stroke=None)


# ---------------- masthead ----------------
st.markdown(
    f"""
    <div class="masthead">
      <div>
        <div class="mh-title">VOLVE FIELD — PRODUCTION FORECAST REPORT</div>
        <div class="mh-meta" style="text-align:left">
          ARPS DECLINE ANALYSIS vs GRADIENT BOOSTING · NPV ECONOMICS</div>
      </div>
      <div class="mh-meta">
        5 PRODUCING WELLS · EQUINOR OPEN DATA · NORWEGIAN NORTH SEA 2008–2016<br>
        SELECTION 60% · FINAL TEST 20% · TEST LABELS NEVER SELECT THE WINNER
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- metric strip ----------------
c1, c2, c3, c4 = st.columns(4)
c1.markdown(metric("Wells analysed", f"{len(comp)}"), unsafe_allow_html=True)
c2.markdown(metric("ML wins head-to-head",
                   f"{(comp['winner'] == 'ML').sum()} / {len(comp)}",
                   "long-complex wells favour ML"), unsafe_allow_html=True)
c3.markdown(metric("Avg validation MAPE",
                   f"{avg_ml:.1f}%", f"Arps {avg_arps:.1f}% · Δ {avg_ml - avg_arps:+.1f} pts"),
            unsafe_allow_html=True)
c4.markdown(metric("Portfolio 5y NPV", f"${portfolio_npv:,.0f} MM",
                   f"pre-tax operating · @ ${price:,.0f}/bbl"),
            unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3 = st.tabs(["Head-to-head", "Well explorer", "Economics"])

# ---------------- tab 1 ----------------
with tab1:
    st.markdown("**Forecast accuracy — final test after model selection**")
    st.caption("The winner is chosen using the middle selection period. The last 20% of "
               "each well's history is untouched until the final accuracy report. "
               "MAPE is scale-free but overreacts to small production values, so MAE and "
               "RMSE are shown alongside it.")

    tbl = comp.copy()
    report_table(tbl, {
        "well": lambda w: f"<b>{w}</b>",
        "ml_mape": lambda v: f"{v:.1f}%",
        "arps_mape": lambda v: f"{v:.1f}%",
        "ml_mae": lambda v: f"{v:,.0f}",
        "arps_mae": lambda v: f"{v:,.0f}",
        "ml_rmse": lambda v: f"{v:,.0f}",
        "arps_rmse": lambda v: f"{v:,.0f}",
        "ml_eur_5y_mbbl": lambda v: f"{v:,.0f}",
        "arps_eur_5y_mbbl": lambda v: f"{v:,.0f}",
        "winner": lambda w: (f'<span class="win-ml">ML</span>' if w == "ML"
                             else f'<span class="win-arps">ARPS</span>'),
    })

    long = comp.melt(id_vars="well", value_vars=["ml_mape", "arps_mape"],
                     var_name="method", value_name="mape")
    long["method"] = long["method"].map({"ml_mape": "ML", "arps_mape": "Arps"})
    bars = (alt.Chart(long).mark_bar().encode(
        x=alt.X("well:N", sort=WELLS, title="well"),
        xOffset=alt.X("method:N", sort=["ML", "Arps"]),
        y=alt.Y("mape:Q", title="validation MAPE, %"),
        color=alt.Color("method:N", scale=alt.Scale(domain=["ML", "Arps"],
                                                    range=[AMBER, TEAL]), title=None),
        tooltip=[alt.Tooltip("well", title="well"),
                 alt.Tooltip("method", title="method"),
                 alt.Tooltip("mape", format=".1f", title="MAPE %")])
        .properties(height=300))
    st.altair_chart(theme(bars), use_container_width=True)

    st.markdown(
        '<div class="note">ML cuts forecast error roughly in half on long, '
        'operationally complex wells. Arps stays competitive on short-history wells. '
        'Best practice: run both, select per well.</div>',
        unsafe_allow_html=True)

# ---------------- tab 2 ----------------
with tab2:
    well = st.selectbox("Well", WELLS, label_visibility="collapsed")
    g_hist, n_hist, t_fut, q_fut, m = well_forecast(well)
    a = arps_params.set_index("well").loc[well]

    m1, m2, m3 = st.columns(3)
    m1.markdown(metric(f"ML MAPE — {well}", f"{m['ml_mape']:.1f}%",
                       f"MAE {m['ml_mae']:,.0f} · RMSE {m['ml_rmse']:,.0f}"),
                unsafe_allow_html=True)
    m2.markdown(metric("Arps MAPE", f"{m['arps_mape']:.1f}%"),
                unsafe_allow_html=True)
    m3.markdown(metric(f"5y NPV — winner {m['winner']}",
                       f"${npv_of(q_fut) / 1e6:,.0f} MM",
                       f"EUR {q_fut.sum() / 1e3:,.0f} Mbbl · P10-P90 below"),
                unsafe_allow_html=True)

    hist = g_hist[g_hist["active"]]
    arps_t = np.arange(1, n_hist + HORIZON + 1, dtype=float)
    arps_q = (a["qi"] / np.power(1 + a["b"] * a["Di"] * arps_t, 1 / a["b"]))

    df_hist = pd.DataFrame({"t": hist["t"].values,
                            "q": np.clip(hist["bbl"].values, 1, None),
                            "series": "history"})
    df_arps = pd.DataFrame({"t": arps_t,
                            "q": np.clip(arps_q, 1, None),
                            "series": "Arps curve"})
    df_fut = pd.DataFrame({"t": t_fut,
                           "q": np.clip(q_fut, 1, None),
                           "series": "ML forecast"})
    if m["winner"] == "ML" and f"{well}_p10" in forecasts:
        interval = pd.DataFrame({
            "t": t_fut,
            "p10": forecasts[f"{well}_p10"].values,
            "p90": forecasts[f"{well}_p90"].values,
        })
        st.area_chart(interval.set_index("t"), height=120)
        st.caption("Forecast interval: residual bootstrap P10-P90 band; it is an "
                   "uncertainty range, not a guarantee.")
    df_rule = pd.DataFrame({"split": [n_hist + 0.5]})

    base = alt.Chart(pd.concat([df_hist, df_arps, df_fut])).mark_line().encode(
        x=alt.X("t:Q", title="months online"),
        y=alt.Y("q:Q", scale=alt.Scale(type="log"),
                title="oil rate, bbl/month — log scale"),
        color=alt.Color("series:N", scale=alt.Scale(
            domain=["history", "Arps curve", "ML forecast"],
            range=[SAND, TEAL, AMBER]), title=None),
        tooltip=["series", alt.Tooltip("t", title="month"),
                 alt.Tooltip("q", format=",", title="bbl/mo")])
    pts = (alt.Chart(df_hist).mark_circle(size=16, color=SAND, opacity=.85)
           .encode(x="t:Q", y="q:Q"))
    rule = (alt.Chart(df_rule).mark_rule(strokeDash=[6, 4], color=RED)
            .encode(x="split:Q"))
    hero = theme((base + pts + rule).properties(height=400)
                 .configure_scale(bandPaddingInner=0.3))
    st.altair_chart(hero, use_container_width=True)
    st.caption("Semi-log decline plot — the standard Arps reading convention. "
               "Red rule = last observed month; left: history + fitted Arps curve; "
               "right: 5-year forward view. Log scale floors at 1 bbl/mo.")

    with st.expander("Fitted Arps parameters"):
        st.markdown(f'<span style="font-family:{MONO};font-size:.8rem;color:{SAND}">'
                    f'qi = {a["qi"]:,.0f} bbl/mo &nbsp;·&nbsp; Di = {a["Di"]:.3f} /mo'
                    f' &nbsp;·&nbsp; b = {a["b"]:.2f} &nbsp;·&nbsp; '
                    f'trained on first 70% of history</span>', unsafe_allow_html=True)

# ---------------- tab 3 ----------------
with tab3:
    st.markdown("**5-Year simplified pre-tax operating NPV — live assumptions**")
    st.caption(f"${price:,.0f}/bbl oil · ${opex:,.1f}/bbl opex · "
               f"{disc_pct:.1f}% annual discount · winning forecast per well")

    npv_sorted = npv_live.sort_values("npv_musd", ascending=False)
    hbars = (alt.Chart(npv_sorted).mark_bar(color=AMBER).encode(
        x=alt.X("npv_musd:Q", title="NPV, million USD"),
        y=alt.Y("well:N", sort=alt.EncodingSortField(
            field="npv_musd", order="descending"), title=None),
        tooltip=[alt.Tooltip("well", title="well"),
                 alt.Tooltip("npv_musd", format="$.1f", title="NPV, MM")])
        .properties(height=260))
    st.altair_chart(theme(hbars), use_container_width=True)

    tbl = npv_sorted.copy()
    tbl["eur_5y_mbbl"] = [well_forecast(w)[3].sum() / 1e3 for w in tbl["well"]]
    tbl["method"] = [comp.set_index("well").loc[w, "winner"] for w in tbl["well"]]
    tbl = tbl.rename(columns={"well": "Well", "method": "Method",
                              "eur_5y_mbbl": "EUR 5y, Mbbl", "npv_musd": "NPV, MM USD"})
    report_table(tbl, {
        "Well": lambda w: f"<b>{w}</b>",
        "Method": lambda w: (f'<span class="win-ml">ML</span>' if w == "ML"
                             else f'<span class="win-arps">ARPS</span>'),
        "EUR 5y, Mbbl": lambda v: f"{v:,.0f}",
        "NPV, MM USD": lambda v: f"${v:,.1f}",
    })
    st.markdown(metric("Portfolio total", f"${portfolio_npv:,.0f} MM",
                       f"5-year discounted · {len(WELLS)} wells · @ ${price:,.0f}/bbl"),
                unsafe_allow_html=True)
    st.markdown("**Price / opex sensitivity**")
    sens = sensitivity.groupby(["price_usd_bbl", "opex_usd_bbl"])["npv_musd"].sum().reset_index()
    st.dataframe(sens.pivot(index="opex_usd_bbl", columns="price_usd_bbl",
                            values="npv_musd").style.format("${:,.0f}"),
                 use_container_width=True)

with st.expander("Methodology and economic scope"):
    st.markdown(
        "Production is aggregated from daily standard cubic metres (Sm3) to monthly "
        "stock-tank barrels using 1 Sm3 = 6.2898 bbl. Models use a temporal 60% fit, "
        "20% selection period, and untouched 20% final test period. Arps and gradient "
        "boosting are compared with last-value and moving-average baselines in the "
        "pipeline outputs. NPV is a simplified **pre-tax operating** calculation: "
        "oil price minus variable opex, discounted monthly for five years. It excludes "
        "taxes, royalties, capex, abandonment, downtime/facility constraints, "
        "transport, water/gas economics, and working capital. Treat it as a prototype "
        "scenario tool, not investment advice."
    )

st.divider()
st.caption("AhmedAbugebbah · petroleum engineering × data science · "
           "data: Equinor Volve open dataset · methodological demonstration, "
           "not investment advice")
