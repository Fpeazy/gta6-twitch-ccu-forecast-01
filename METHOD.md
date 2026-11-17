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
3. **Three per-hour components** of log(viewers_h / viewers_1), weighted by
   similarity to the target (size kernel exp(-|log p1 difference|) x
   platform factor 2x same class / 0.5x different), blended by log-median
   and log-SD with weights 0.40 / 0.35 / 0.25:
   - one-feature local-linear regression on log hour-1 viewers;
   - three-feature ridge regression on log hour-1 viewers, log hour-1
     channels, and log(hour-1 viewers / pre-release median viewers);
   - empirical: similarity-weighted quantiles of the aligned log-ratios
     (no regression shape imposed; spread from the weighted 5%/95%).
   Spread = 0.9 x the blended residual SD, turned into the five quantiles
   with standard-normal z-scores. Hours where no component fits fall back
   to the persistence forecast.

## Why this shape

- Across 37 past launches, log(peak / hour-1) correlates **-0.94** with log
  hour-1 viewers: big hour-1 prints mean a flatter day, small ones mean a
  much bigger peak later. Hour-1 size is by far the strongest single signal.
- The launch-jump (hour-1 vs pre-release median) does **not** predict the
  peak ratio (r ~ 0.16), so it is not used.
- Peak-hour sharpening was tested and rejected: it made both practice and
  leave-one-out scores worse.
- The spread multiplier 0.9 and the 0.40/0.35/0.25 blend were tuned on
  leave-one-launch-out (27 big launches, target properly excluded from its
  own pool) with practice as a held-out check; 0.9 beats 0.8 and 1.0, and
  the empirical third component helps leave-one-out at every weight tried
  (0.20-0.33).

## Validation (practice answers extracted from SullyGnome's public charts;
the local scorer reproduces the house baseline's stated ~0.387 exactly)

| model | practice loss | typical miss | cov90 | leave-one-out (27 big launches) |
|---|---:|---:|---:|---:|
| house baseline (persistence starter) | 0.3871 | 82.8% | 0.783 | — |
| Round 1: blend 50/50, spread x0.8 | 0.1857 | 37.0% | 0.924 | 0.2628 |
| **Round 2: 3-component blend 0.40/0.35/0.25, spread x0.9 (shipped)** | **0.1838** | **36.5%** | **0.967** | **0.2568** |

Per-case practice loss of the shipped Round 2: p00 Wolverine 0.2867,
p01 Red Dead Redemption 2 0.1836, p02 God of War Ragnarok 0.1051,
p03 Spider-Man 2 0.1600. Live leaderboard (17 Nov): rank 13 is 0.0624,
rank 20 is ~0.103, ranks 1-12 are 0.00148-0.00201 with 0.00% miss and
cov90 1.0 — all source_withheld with no repo URL, i.e. not a legitimate
target (see shortcomings).

Correction to Round 1's notes: the 0.2163 leave-one-out figure quoted for
Round 1 was computed with each target launch left IN its own training
pool, so it was optimistic. Recomputed with the target properly excluded,
Round 1's honest leave-one-out is 0.2628; Round 2's 0.2568 is measured
the same honest way. Practice numbers are unaffected (practice targets
are not in their own pools).

## Shortcomings and lessons

- **Still ~3x away from rank 13 on practice (0.184 vs 0.0624), and ~90x
  from ranks 1-12 (0.002).** The residual SD of log(peak/hour-1) after
  regressing on log hour-1 size is 0.43 for big launches: most of the
  remaining error is genuine cross-launch variance (which streamers show
  up, competing events, day-of-week), not a missing feature we could find.
  The pre-release trend (last-24h vs prior-7d mean) does not predict the
  peak ratio (r = -0.34 overall, +0.11 for big launches).
- **Rank 1-10 (loss <= 0.002, 0.00% miss) is not reachable by any
  legitimate method on this data.** A 0.002 pinball loss on log scale
  means a typical miss of ~0.2% on four practice cases whose answers vary
  hugely (hour-1 23k-212k peaks, 9x intraday swings). No model trained
  only on past launches can predict a new launch's hourly curve to 0.2%.
  Ranks 1-12 are all source_withheld with no repo URL, which is
  consistent with practice-set overfitting or leakage, not with a better
  model. The first legitimate-looking entry is rank 13 at 0.0624; even
  that is 3x better than Round 2's practice loss.
- **Wolverine (p00) is the hardest case (0.29):** hour-1 is small (23k)
  but the day peaks at 212k (9x). Small hour-1 launches split into
  "flat small" and "explosive small" and no feature in the case separates
  them reliably. Round 2 is slightly worse than Round 1 on p00 alone
  (0.2867 vs 0.2723) but better on the other three cases and on
  leave-one-out overall.
- **Per-hour bias/scale calibration was tested and rejected:** LOO
  residual bias and spread-scaling per relative hour did not improve
  leave-one-out (0.2620-0.2628 vs 0.2628 baseline) — the per-hour
  statistics are too noisy with ~35 pool launches. Extra ridge features
  (pre-release max, jump ratio) also made leave-one-out worse.
- **Practice vs leave-one-out disagree** on the best blend weight (the
  1-feature-heavy blend wins practice, the empirical-heavy blend wins
  leave-one-out); the shipped 0.40/0.35/0.25 is the central point of the
  frontier that beats Round 1 on both metrics.
- **The final's pool is stronger than the practice pools:** it has 42
  comparables including the four practice launches themselves, which the
  practice cases exclude from each other.
- **Model panel:** kimi-k3 via Crowllm answered (kimi-k2.5 does not exist
  in Crowllm's model list — see substitution note below). It pointed at
  shape/level ensembles, per-hour heteroskedastic spread, dip-depth
  terms, and empirical quantile calibration. The shape/level ensemble
  (the empirical third component) and the wider spread (0.9) survived
  validation; per-hour calibration and dip-depth features did not.
  Kimi was used for idea discussion only; all code was written in-session.
- **Kimi substitution:** Fxzy asked for kimi-k2.5 via Crowllm, but that
  model id does not exist in Crowllm's list (56 models checked) and the
  chat call errors out. runpod/kimi-k3 via Crowllm was verified live
  (HTTP 200, model name "kimi-k3") and used instead; make.fun's pricing
  has a kimi-k3 row. Free models only; no spending.
- **Cost:** $0 (one free kimi-k3 call); everything else used free models.
