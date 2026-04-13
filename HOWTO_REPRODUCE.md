# HOWTO: Reproduce and Independently Verify Any Enneagrid Consortium Season

**Document purpose:** This guide explains exactly how any external researcher, auditor,
or interested reader can fully reproduce and verify the consortium's predictions, scoring,
and evidence chain using only publicly available materials. No private access or trust in
the consortium members is required.

**Last updated:** April 2026 · aligned with PROTOCOLS_v2  
**Applies to:** Season 1 (MAPS — C/2026 A1) and all subsequent seasons

---

## 1. Obtain the Public Materials

All necessary files are available in two permanent locations:

- **GitHub** — `github.com/heliodata-ai/maps-dashboard` — full version history with GPG
  signatures on all commits from March 31, 2026 onward (key `B60047DAF4DBADB4`)
- **Zenodo** — DOI `10.5281/zenodo.19295676` — permanent archive including the final
  GPG-signed science log as an attachment

Required files for a full audit:

| File | Location | Purpose |
|------|----------|---------|
| `science_log.jsonl` | GitHub + Zenodo | Complete evidence chain — all 675 entries |
| `PROTOCOLS_v2.md` | GitHub | Full rulebook: scoring, Wolfram roles, directives |
| `scorecard.json` | GitHub + Zenodo | Final scored outcomes with Brier scores |
| Season report (PDF/MD) | Zenodo | Narrative account with methodology notes |
| `CITATION.cff` | GitHub | Canonical academic citation |

---

## 2. Verify the Integrity of the Evidence Chain

Clone the repository and verify GPG signatures:

```bash
git clone https://github.com/heliodata-ai/maps-dashboard.git
cd maps-dashboard
git log --oneline --show-signature
```

Or verify a specific commit:

```bash
git verify-commit <commit-hash>
```

All commits from March 31, 2026 onward must be signed with key `B60047DAF4DBADB4`.
The season-closing commit (`12223de` for Season 1) is explicitly marked "FINAL" and
"sealed" — confirm it has not been amended.

Cross-check the `science_log.jsonl` in the repository against the file attached to the
Zenodo deposit. If signatures validate and history is intact, the log is tamper-evident.

**Note on schema:** Science log entries 1–32 (through 2026-03-26T21:00 UTC) use the
legacy schema (`ts` timestamp field, reduced field set). Entry 33 onward conforms to
`comm11-v1` (`utc` field, full schema). Both versions are preserved unmodified as part
of the evidence chain.

---

## 3. Reproduce the Scoring Step by Step

For each prediction in the science log:

**Step 1 — Extract the prediction parameters**

Search by `prediction_id` and extract:

- `locked_value` — the predicted value including uncertainty bounds
- `scoring_source` — the authoritative data source for verification
- `scoring_window` — the time window within which the outcome is evaluated
- `null_condition` — any pre-registered condition that voids the prediction (if present)
- `independence_class` — the structural independence tier of this prediction

**Step 2 — Obtain independent observational data**

Download from the public sources named in each prediction's `scoring_source` field:

| Data type | Source |
|-----------|--------|
| Comet magnitudes, visibility, tail length, nucleus reports | COBS / MPC |
| Conjunction photometry | SOHO LASCO C3 / CCOR-1 |
| Solar activity (F10.7, Kp, X-ray flares) | NOAA SWPC (`planetary-k-index.json`, `10cm-flux.json`, `xray-flares`) |
| CME catalog | NASA DONKI |
| Spectra, fragmentation reports | ATel, CBET, refereed papers |

**Step 3 — Apply the scoring rules from PROTOCOLS Section 4**

- **Rule 1 (interval containment):** `center − unc ≤ observed ≤ center + unc` → CONFIRMED
- **Rule 2 (binary outcome):** compare event occurrence against the pre-registered >50% /
  ≤50% threshold
- Apply any null condition exactly as written — do not interpret or adjust

**Step 4 — Check the Override Register**

Any deliberate deviation from a Wolfram-generated bound has a corresponding Override
Register entry with a documented justification and consortium vote. Confirm each deviation
is pre-registered; deviations without an entry are protocol violations.

**Step 5 — Compute calibration metrics**

Brier scores and other calibration metrics use the formulas in PROTOCOLS Section 3.
Wolfram Language equivalents are given there for reference. Python implementations
are also acceptable and will produce identical results.

Because all scoring sources are public and all rules are deterministic, independent
auditors should arrive at identical scores within stated calibration uncertainty.

---

## 4. Verify Wolfram CAG Computations

Every critical pre-lock calibration and post-season mathematical check produces a unique
UUID via the internal CAG endpoint. These UUIDs appear in the science log and final
scorecard JSON. The canonical Wolfram expressions are given in PROTOCOLS Section 3.

To reproduce any computation independently:

```wolfram
(* Paste the expression into any of: *)
(* - Wolfram Engine (free, community edition) *)
(* - Wolfram Cloud (free tier) *)
(* - Mathematica *)
```

Confirm that each consortium lock either matched the Wolfram-generated bound or has a
corresponding Override Register entry explaining the deviation.

### Important limitation — CAG Tier-2 classifier (Season 1)

Auditors should be aware of a documented finding from Season 1, reported in full in the
season report (Section 6.2) and in the Zenodo README:

**The CAG Tier-2 fragmentation classifier returned identical static values
(`frag=intact`, confidence 0.98, `tail=2.33°`, `ion=0`) on every run from March 27
through April 5, 2026.** It never functioned as a dynamic detector. All evidence chain
UUIDs for this classifier represent the same static computation, not independent
observations.

Additionally, the CAG key returned HTTP 401 Unauthorized from approximately March 26
onward. All calls during this period failed silently (null values logged). The key was
rotated on April 4 at 15:59 UTC; three UUIDs entered the evidence chain after recovery.

These are documented limitations, not concealed failures. The static classifier finding
is the primary reason the Wolfram CAG role has been redesigned in Season 2
(HALEAKALĀ — C/2025 R3). In Season 2 the CAG functions as a symbolic computation
engine for pre-lock calibration only; it is not used as a live state classifier.

You cannot re-execute the exact internal CAG calls (they run on a private VPS endpoint),
but the symbolic results are fully reproducible with the published expressions and any
Wolfram Language environment.

---

## 5. Perform a Full Audit

Follow the audit protocol in PROTOCOLS Section 6:

1. Read the science log **forward chronologically** — never start from the outcome and
   work backward. Begin at T−24h before the event of interest.
2. Verify that anchor selection followed Section 7 — complete reference-class
   construction before analog selection.
3. Confirm all null preconditions were pre-registered **before** lock (Section 5).
4. Check that the Season 1 control prediction used a Wolfram-generated random bit
   (`RandomInteger[]` at lock time) — earlier JD-parity method was replaced in
   PROTOCOLS_v2 due to determinability before lock.

---

## 6. Lookahead — Season 2 (HALEAKALĀ)

Season 2 covers C/2025 R3 PanSTARRS (perihelion April 19–20, 2026, q=0.499 AU).
The evidence chain, scoring methodology, and PROTOCOLS_v2 are identical to Season 1
with the following changes:

- CAG role is limited to pre-lock symbolic calibration (not live classification)
- Control prediction uses `RandomInteger[]` at lock time (not JD parity)
- Telegram alerting architecture active — CAG authentication failures caught within
  minutes
- SMOS L-band solar flux (1.4 GHz, ESA Earth Explorer) used as independent proxy
  alongside NOAA F10.7 for Tier-2 solar activity classification and space weather
  prediction calibration

The Season 2 repository and Zenodo deposit will follow the same structure as Season 1.
The same verification steps in this document apply without modification.

---

## 7. Full Reproduction Package (Published at Season Close)

At the close of each season, the following are published on both Zenodo and GitHub:

- Final report (PDF + Markdown)
- `scorecard.json`
- GPG-signed `science_log.jsonl` (full history)
- `PROTOCOLS_v2.md` (current operative version)
- `CITATION.cff`

With these materials, anyone can:

```bash
git clone https://github.com/heliodata-ai/maps-dashboard.git
git log --show-signature          # verify integrity
cat science_log.jsonl | python3 -m json.tool | less   # inspect entries
# re-score every prediction against public data
# re-run the Wolfram mathematics in any WL environment
```

This design meets high standards of computational reproducibility and open science.

---

## 8. Bottom Line — The Evidence Chain Is the Product

The combination of `science_log.jsonl` + `PROTOCOLS_v2.md` turns every Enneagrid season
into a fully auditable scientific experiment. You do not need to trust G.L. Eukene,
Claude, Grok, DeepSeek, or Wolfram CAG. You can verify every step yourself — from the
moment a prediction was proposed to its final scoring, including the documented failures.

**Intended audience:** Citizen scientists, astronomers, forecasters, reproducibility
researchers, and anyone who wants to treat the consortium's work as a transparent dataset
rather than an opaque claim.

---

*Enneagrid Research Consortium · heliodata.ai · Apache 2.0*
