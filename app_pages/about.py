import streamlit as st

import app_data
import config as cfg

res = app_data.artifacts().results
th = res["thresholds"]
sens = app_data.json_file(str(cfg.MODELS_DIR / "sensitivity_guide_default" / "results.json"))

st.markdown(f"""
The copilot forecasts hazardous working conditions at eight North Sea sites for the next
{res['horizon_hours']} hours and finds UK safety guidance that matches the hazard. It is coursework for
CS5079 Applied AI at the University of Aberdeen.

### How it works
- **Data:** Hourly wind, pressure and temperature from the Open-Meteo archive, and wave height, period and
  direction from the Open-Meteo ERA5-Ocean model, 2020 to November 2026.
- **Target:** The worst class over the next {res['horizon_hours']} hours. An hour is **Critical** when
  {th['wind_col'].replace('_', ' ')} reaches {th['wind_crit']} m/s or wave height reaches {th['wave_crit']} m,
  and **Warning** from {th['wind_warn']} m/s or {th['wave_warn']} m.
- **Split by time:** Train up to 2024, choose the model on 2025, test once on 2026. The hours just before each
  boundary are dropped so no target looks into the next period.
- **Baseline:** Persistence - the next {res['horizon_hours']} hours are assumed to look like the current hour.
- **Guidance:** A corpus of MAIB investigation reports and HSE offshore notices,
  bulletins and information sheets, split into 200-word passages. Search uses BM25, MiniLM embeddings, or a
  reciprocal rank fusion of the two.

### Limitations
- The thresholds are assumptions taken from published operating limits, not limits for a specific vessel or
  operation.
""" + (f"""- With the looser '{sens['threshold_set']}' thresholds the chosen model changes to {sens['winner']},
  so results depend on this choice.
""" if sens else "") + """- The model learns from reanalysis data, not live forecasts, so a live forecast would carry extra error.
- Site coordinates are approximate, and nearby points share one coarse wave grid cell.
- The app replays a sample of 400 hours per site from 2026; it is not connected to live data.
- Retrieved passages are pointers to the source documents and do not replace them.

**Decision support for coursework only. Not for operational use.**

### Data and licences
Weather and sea-state data: [Open-Meteo](https://open-meteo.com), used under their free non-commercial terms
with attribution. Safety documents: [MAIB reports on GOV.UK](https://www.gov.uk/maib-reports) and the
[HSE offshore pages](https://www.hse.gov.uk/offshore/), Open Government Licence.
""")
