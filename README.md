# heliodata.ai — MAPS Dashboard

**Live space weather dashboard for C/2026 A1 (MAPS) perihelion watch**  
Enneagrid Research Consortium · April 4, 2026 14:24 UTC

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Dashboard](https://img.shields.io/badge/Live-heliodata.ai-cyan)](https://heliodata.ai)
[![Science Log](https://img.shields.io/badge/Science%20Log-public%20JSONL-green)](https://heliodata.ai/data/science_log.jsonl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19295676.svg)](https://doi.org/10.5281/zenodo.19295676)
---

## What this is

heliodata.ai is a live space weather dashboard built for the solar encounter of 
comet C/2026 A1 (MAPS), a Kreutz sungrazer with perihelion on April 4, 2026 at 
14:24 UTC — passing 160,000 km from the solar surface.

On February 18, 2026, a four-AI consortium issued **16 falsifiable, quantified 
predictions** about this event. The dashboard monitors conditions against those 
predictions in real time. After April 4, every prediction is scored publicly.

**If we are wrong, we say so.**

---

## The Consortium

| Member | Role |
|--------|------|
| Claude (Anthropic) | Coordinator · implementation · integration |
| Grok (xAI) | Wire format · CI pipeline · scoring schema |
| DeepSeek | Mathematical formalization · CW-complex topology |
| Wolfram Language CAG | F10.7 baseline · orbital mechanics · Tier 2 classifiers |

Coordinated by **G.L. Eukene (Eugharaz)** — professional geodesist, 
Basque Country, Spain.

---

## The 16 Predictions — Locked February 18, 2026

| # | Prediction | Value |
|---|-----------|-------|
| 1 | Fragmentation probability | 55% ± 10% |
| 2 | Peak apparent magnitude | −7 ± 2 |
| 3 | Peak brightness time | Apr 4 ~14:20 UTC ± 3h |
| 4 | Maximum tail length | 30° ± 10° |
| 5 | Post-perihelion fragment count | 3–5 |
| 6 | Naked-eye visibility window | Apr 5–21 |
| 7 | SOHO coronal detection | Yes — 75% |
| 8 | Ion tail disconnection events | 3 ± 2 |
| 9 | PanSTARRS C/2025 R3 peak magnitude | 3.5 ± 1.5 |
| 10 | In-situ spacecraft detection | No — 85% |
| 11 | CME within ±12h of perihelion | Yes — 80% |
| 12 | Earth-directed CME | No — 90% |
| 13 | Max Kp index Apr 1–7 | 5 ± 1.5 |
| 14 | G1+ geomagnetic storm | 58% |
| 15 | NOAA official advisory | Yes — 70% |
| 16 | Satellite fleet survival to May | >99% |

Full scoring window: **May 1–15, 2026**

---

## Dashboard Features (v12)

- **Solar Intelligence Panel** — GOES X-ray class · active regions (NOAA SRS) · 
  solar wind Bz (DSCOVR L1) · CAG flare probability
- **F10.7 Wolfram CAG Baseline** — 81-day smoothed solar radio flux · 
  SC25 decline modelling
- **Comet Intelligence Panel** — CAG orbital state · DC index · tail PA · 
  observer vocabulary
- **Tier 2 CAG Classifiers** — fragmentation morphology · tail length estimator · 
  ion disconnection counter · solar wind shock detector
- **Orbital Track** — Barker's equation ecliptic diagram · live planet positions
- **Science Log** — public JSONL · hourly git-committed · UUID provenance

---

## Data Sources

- NOAA SWPC — F10.7 · Kp · GOES X-ray · DSCOVR solar wind · SRS · DONKI CME
- NASA SDO AIA 304Å — chromosphere imagery
- SOHO LASCO C2/C3 — coronagraph
- Wolfram Language CAG — orbital computation · F10.7 baseline · image classifiers
- SMOS solar flux dataset — Serco Red Lab / ESRIN  
  (Zenodo: [10.5281/zenodo.15275693](https://doi.org/10.5281/zenodo.15275693))
- JPL SBDB / Horizons — orbital elements (solution #9 · 705 observations)

---

## Evidence Chain

The science log starts at corona ingress (March 28, 2026) and runs hourly 
through the May scoring window. Every entry is git-committed. Every Wolfram CAG 
computation carries a UUID for reproducibility.

Public log: [heliodata.ai/data/science_log.jsonl](https://heliodata.ai/data/science_log.jsonl)

---

## Orbital Elements (JPL Solution #9)
```
q  = 0.005732 AU
e  = 0.9999627756792491
i  = 144.49°
T  = 2026-Apr-04 14:24 UTC
```

---

## License

Copyright 2026 Eugenio García López — Enneagrid Research Consortium

Licensed under the Apache License, Version 2.0.  
See [LICENSE](LICENSE) for full terms.

---

## Season 2

PanSTARRS C/2025 R3 — perihelion April 20, 2026  
SMOS dataset integration · EU magnetometer network proposal

---

*heliodata.ai · Enneagrid Research Consortium · Basque Country, Spain*
```

---

Paste that directly into the README editor on GitHub. It renders cleanly with the badges, table, and all sections. Hit **Commit changes** with message:
```
Initial README — MAPS dashboard v12, 16 predictions, corona ingress day
