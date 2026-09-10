# Data dictionary and source notes

The raw file is `data/volve_daily_production.csv`, from Equinor's Volve Data
Sharing release. The project uses the daily production stream only and keeps
wells with at least 18 active months and more than 500 Mbbl after aggregation.

| Field | Meaning | Unit / treatment |
|---|---|---|
| `DATEPRD` | Production date | Parsed as `%d-%b-%y` |
| `NPD_WELL_BORE_NAME` | Well-bore identifier | Retained as `well` |
| `BORE_OIL_VOL` | Daily oil volume | Sm3; converted with 6.2898 bbl/Sm3 |
| `ON_STREAM_HRS` | Daily online time | Hours; active month is at least 100 hours |
| `AVG_DOWNHOLE_PRESSURE` | Daily average downhole pressure | Source unit, monthly mean |
| `AVG_WHP_P` | Daily wellhead pressure | Source unit, monthly mean |
| `AVG_CHOKE_SIZE_P` | Daily choke size | Source unit, monthly mean |

The monthly file contains `oil_sm3`, `bbl`, `on_stream_h`, `active`, and a
within-well month index `t`. Missing sensor values are left as missing in
monthly means; production volume and online hours use zero for missing daily
values.

The NPV scenario is nominal USD per barrel, monthly production in bbl/month,
and an effective annual discount rate. Taxes, royalties, capex, abandonment,
facility constraints, downtime beyond the active-month filter, transport,
water/gas economics, and working capital are intentionally excluded.
