# Method — comparables-weighted log-ratio forecaster

## What it does

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads one case JSON and writes five quantiles (5/25/50/75/95%) of Twitch
concurrent viewers for hours 2-24 of the launch day.

1. **Pool.** The case's comparables (38 in practice, 42 in the final). A
   launch enters only if its release-day curve has an hour-1 print and at
   least 20 of 24 hours. The GTA VI category's own pre-release history
   (a comparable in every case) is excluded by this filter.
2. **Unlock alignment.** A launch-day curve jumps when the US East Coast
   unlocks at midnight local. Each comparable's jump hour is found from its
   own curve (largest log increase in hours 2-10). The target's jump hour
   comes from its unlock schedule (GTA VI: US East 05:00Z -> hour 6;
   practice launches: parsed from the case notes, else the US
   daylight-saving rule). Curves are compared in time relative to the jump.
3. **Two per-hour local-linear regressions** of log(viewers_h / viewers_1),
   weighted by similarity to the target (size kernel
   exp(-|log p1 difference|) x platform factor 2x same class / 0.5x
   different):
   - one feature: log hour-1 viewers;
   - three features: log hour-1 viewers, log hour-1 channels, and
     log(hour-1 viewers / pre-release median viewers).
   Spread = 0.8 x the weighted residual SD, turned into the five quantiles
   with standard-normal z-scores. The two components are blended 50/50
   (geometric mean of the quantiles). Hours with too few pool rows fall
   back to the persistence forecast.

## Why this shape

- Across 37 past launches, log(peak / hour-1) correlates **-0.94** with log
  hour-1 viewers: big hour-1 prints mean a flatter day, small ones mean a
  much bigger peak later. Hour-1 size is by far the strongest signal.
- The launch-jump (hour-1 vs pre-release median) does **not** predict the
  peak ratio (r ~ 0.16), so it is not used.
- Peak-hour sharpening was tested and rejected: it made both practice and
  leave-one-out scores worse.
- The spread multiplier 0.8 was tuned on practice + leave-one-out (0.8
  beats 1.0 and 0.9; 0.7 is a near-tie on practice).

## Validation (practice answers extracted from SullyGnome's public charts;
the local scorer reproduces the house baseline's stated ~0.387 exactly)

| model | practice loss | typical miss | cov90 | leave-one-out (27 big launches) |
|---|---:|---:|---:|---:|
| house baseline (persistence starter) | 0.3871 | 82.8% | 0.783 | — |
| one-feature local-linear, size+platform | 0.1802 | 33.7% | 0.967 | 0.2416 |
| three-feature regression, size+platform | 0.2004 | 40.9% | 0.946 | 0.2057 |
| **blend 50/50, spread x0.8 (shipped)** | **0.1857** | **37.0%** | **0.924** | **0.2163** |

Per-case practice loss of the shipped blend: p00 Wolverine 0.2723,
p01 Red Dead Redemption 2 0.2046, p02 God of War Ragnarok 0.1103,
p03 Spider-Man 2 0.1557. Live rank-1 target (leaderboard, 17 Nov): loss
< 0.122, typical miss ~19%.

## Shortcomings and lessons

- **Still ~1.5x away from rank 1 on practice (0.186 vs 0.122).** The
  residual SD of log(peak/hour-1) after regressing on log hour-1 size is
  0.43 for big launches: most of the remaining error is genuine
  cross-launch variance (which streamers show up, competing events,
  day-of-week), not a missing feature we could find. The pre-release
  trend (last-24h vs prior-7d mean) does not predict the peak ratio
  (r = -0.34 overall, +0.11 for big launches).
- **Wolverine (p00) is the hardest case (0.27):** hour-1 is small (23k)
  but the day peaks at 212k (9x). Small hour-1 launches split into
  "flat small" and "explosive small" and no feature in the case separates
  them reliably.
- **Practice vs leave-one-out disagree** on the best single model (the
  one-feature model wins practice, the three-feature wins
  leave-one-out); the blend is the hedge, and it is the best compromise
  on the average of the two metrics.
- **The final's pool is stronger than the practice pools:** it has 42
  comparables including the four practice launches themselves, which the
  practice cases exclude from each other.
- **Model panel:** GLM and Nemotron (Crowllm) answered; both pointed at
  the pre-release trajectory / day-of-week as the richest unused signal,
  but the data refutes the pre-release trend as a peak predictor (above).
  DeepSeek (TokenLab) answered once on retry but its content was lost to
  output truncation; Qwen timed out twice and never answered.
- **Cost:** $0.00048 (one DeepSeek call); everything else used free models.
