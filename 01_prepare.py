"""
Step 1 - Prepare: raw Volve daily production -> clean monthly series per well.
Source: Equinor Volve field dataset (via public mirror), 2008-2016, Norwegian North Sea.
"""
import pandas as pd
import numpy as np
from forecast_utils import SM3_TO_BBL, sm3_to_bbl

RAW = "data/volve_daily_production.csv"
OUT = "data/monthly_production.csv"

def main():
    df = pd.read_csv(RAW)
    df["date"] = pd.to_datetime(df["DATEPRD"], format="%d-%b-%y", errors="coerce")
    if df["date"].isna().any():
        raise ValueError("DATEPRD contains values that could not be parsed")
    df = df[df["FLOW_KIND"] == "production"].copy()

    # daily volumes are Sm3; keep only days the well actually produced
    df["oil_sm3"] = pd.to_numeric(df["BORE_OIL_VOL"], errors="coerce").fillna(0.0)
    df["on_stream"] = pd.to_numeric(df["ON_STREAM_HRS"], errors="coerce").fillna(0.0)

    # monthly aggregation per well
    agg = df.groupby(["NPD_WELL_BORE_NAME", pd.Grouper(key="date", freq="MS")]).agg(
        oil_sm3=("oil_sm3", "sum"),
        on_stream_h=("on_stream", "sum"),
        avg_pressure=("AVG_DOWNHOLE_PRESSURE", "mean"),
        avg_whp=("AVG_WHP_P", "mean"),
        avg_choke=("AVG_CHOKE_SIZE_P", "mean"),
    ).reset_index().rename(columns={"NPD_WELL_BORE_NAME": "well"})

    agg = agg.sort_values(["well", "date"]).reset_index(drop=True)

    # a "producing month" = well online at least 100 hours
    agg["active"] = agg["on_stream_h"] >= 100
    agg["bbl"] = sm3_to_bbl(agg["oil_sm3"])  # Sm3 -> stock tank barrels

    # keep wells with a meaningful production history
    stats = agg.groupby("well").agg(months=("date", "count"),
                                    active_months=("active", "sum"),
                                    total_mbbl=("bbl", lambda s: s.sum() / 1e3))
    keep = stats[(stats["active_months"] >= 18) & (stats["total_mbbl"] > 500)].index
    agg = agg[agg["well"].isin(keep)].copy()

    # continuous month index per well
    agg["t"] = agg.groupby("well").cumcount() + 1

    agg.to_csv(OUT, index=False)

    print(f"wells kept: {len(keep)}")
    for w in keep:
        s = stats.loc[w]
        print(f"  {w:12s} months={int(s.months):3d} active={int(s.active_months):3d} "
              f"total={s.total_mbbl:8.0f} Mbbl")
    print(f"\nsaved -> {OUT} ({len(agg)} rows)")

if __name__ == "__main__":
    main()
