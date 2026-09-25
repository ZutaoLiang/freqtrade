# Full rerun under SKILL scheme C (2026-09-25)

Scheme C: TRAIN 2025-01-01..2025-10-01 (unchanged) · EXCLUDED 2025-10-01..2025-11-30 (10/10 exchange event) ·
VALID 2025-12-01..2026-03-01 · HOLDOUT 2026-03-01..latest local (funding ends 2026-08-31, so unchanged for engine runs).
Snapshot of r3 copied to r3c (r3 is also written by another agent, process `agy`); split constants and paths patched here only.

## TRAIN (unchanged by construction — verified)
- R01–R49 (54 scripts, both sessions): every TRAIN CSV identical to the original (max diff ≤ 5e-12).
- R51–R750 (10 batch scripts, 700 rounds): TRAIN and VALID CSVs identical to the other agent's 08:19–08:21 run (which already used scheme C).
→ every round rejected on TRAIN stays rejected.

## Candidates that had reached VALID — re-judged on VALID-C
| candidate | old VALID (2025-10..2026-02) | VALID-C (2025-12..2026-02) | verdict |
|---|---|---|---|
| R13 settlement receiver (freqtrade) | PF 1.04 | 174 tr, PF 0.80, t −1.59 | still REJECT |
| R20 OI build-up short | −33 bp, PF 0.83 | 18 tr (0.12/day), +44 bp, t 0.7 | still REJECT (frequency) |
| R21 EMA-offset dip buy | −40 bp, PF 0.78 | 976 tr, PF 1.007, t 0.75 | still REJECT |
| R30 BTC momentum high-beta | PF 1.02 | −47 bp, PF 0.63 | still REJECT |
| R34 funding z-score | −58 bp, PF 0.81 | +31 bp, PF 1.16, t 0.66 | still REJECT |
| **R24 funding exhaustion short (freqtrade)** | 230 tr, PF 1.41, t 2.05 | **99 tr, PF 2.42, t 2.30, 3/3 months** | improves; relaxed-pass |
| R51–R750 (700 rounds) | — | 0 of 51 TRAIN passers reach PF ≥ 1.2 and t ≥ 2 | no survivor |

Closest R51–R750 misses: DualSq squeeze variants (e.g. R274/R266/R286): VALID-C PF 1.57–1.88, +66…+95 bp, 6–8 trades/day, but day t 0.8–0.96.

## R24 under scheme C (freqtrade, real funding)
- VALID-C: 99 trades, 1.10/day, +203 bp, PF 2.42, t 2.30; fee ×1.5 PF 2.31; plateau PF 2.10 / 1.81 (F 0.024 / 0.036 %), 2.01 / 2.39 (hold 384 / 576).
- HOLDOUT (unchanged, already read once): 97 trades, PF 1.60, t 1.01.
- VALID-C + HOLDOUT: 196 trades, 0.72/day, +157 bp, PF 1.96, **t 2.36**, 7/9 months positive, ARC 60 %, top day 21 %; fee ×1.5 +289 USDT.
- Ex-ARC: 100 trades, PF 1.75, t 1.35.
- Standard option: still FAILS (B: 99 < 300 VALID trades; E: ARC 60 %). Relaxed option: PASSES (stronger than under the old split).
- Dropped Oct–Nov contained 131 R24 trades netting −14.6 USDT — the improvement comes from excluding them.

Caveat: VALID-C is a subset of the old VALID, and the split was redefined after Oct–Nov was known to hurt several candidates;
VALID-C evidence is therefore post-hoc. Only fresh data (after 2026-08-31 with funding) is a clean test now.
