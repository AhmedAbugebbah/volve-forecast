# LinkedIn post draft

---

I forecast oil production from a real North Sea field two ways — 60-year-old
engineering math vs machine learning — then asked the question recruiters
actually care about: how much money is left in these wells? 🛢️

The dataset: Volve — Equinor released the full production history of this real
Norwegian field to the public. 15,634 well-days, 5 producing wells, up to
104 months of history.

The duel:
📉 Arps hyperbolic decline — the industry-standard decline curve analysis
🤖 Gradient boosting with lagged production + operational features

Validated honestly: temporal 70/30 split, walk-forward one-step-ahead,
no lookahead. Then I converted the winning forecasts into 5-year NPV
at $70/bbl.

What I found:
→ ML cut forecast error roughly in HALF on wells with long, complex
  histories (MAPE 126% → 46% on well F-12)
→ But classical Arps still wins on short-history wells — 60-year-old
  math is hard to beat when data is scarce
→ Best answer isn't "ML replaces DCA". It's: run both, pick per well.

The portfolio answer: $893 MM of discounted 5-year value across 5 wells —
and the ranking of which wells deserve the capital.

This project is where petroleum engineering meets data science, and I built
it to work on real problems like this at scale.

Full code + interactive dashboard on GitHub: [link]
#PetroleumEngineering #DataScience #MachineLearning #EnergyIndustry
#ProductionForecasting #OilAndGas #Python

---

## Tips before posting
- Attach: figures/arps_fits.png, figures/ml_vs_arps.png, figures/npv.png
- Post 8–10 AM Tue–Thu (Cairo time catches EU morning + Gulf afternoon)
- First comment: link the GitHub repo + dashboard screenshot
- Reply to every comment within the first hour — the algorithm rewards it
