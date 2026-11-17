"""Comparables-weighted log-ratio forecaster for gta6-twitch-ccu-forecast-01.

Forecasts Twitch concurrent viewers in the Grand Theft Auto VI category for
hours 2-24 of launch day, as five quantiles per hour. The program reads one
case JSON (/input/<id>.in) and writes /output/<id>.out. Standard library only,
no network, fully deterministic.

Method (validated on the four practice launches and by leave-one-launch-out
over 27 big past launches; see README for the numbers):

  1. Pool: the case's comparables (past big game launches with full hourly
     curves). A launch enters the pool only if its release-day curve has an
     hour-1 print and at least 20 of the 24 hours. The GTA VI category's own
     pre-release history (a comparable in every case) is excluded by this
     filter: it has no hour-1 print on its own day_start.
  2. Unlock alignment: a launch-day curve jumps when the US East Coast
     unlocks (midnight local). Each comparable's jump hour is found from its
     own curve (the hour of the largest log increase in hours 2-10). The
     target's jump hour comes from its known unlock schedule (GTA VI: US
     East 05:00Z -> hour 6; practice launches: parsed from the case notes,
     else the US daylight-saving rule). Curves are compared in time relative
     to the jump, so the unlock step lines up across launches.
  3. Two per-hour local-linear regressions of log(viewers_h / viewers_1),
     weighted by similarity to the target:
       (a) one feature: log hour-1 viewers;
       (b) three features: log hour-1 viewers, log hour-1 channels, and
           log(hour-1 viewers / pre-release median viewers).
     Weights are a size kernel exp(-|log p1 difference|) times a platform
     factor (2x for the same console-only/PC class, 0.5x otherwise).
     The spread is 0.8 x the weighted residual standard deviation, turned
     into the five quantiles with standard-normal z-scores. The two
     components are blended 50/50 (geometric mean of the quantiles).
  4. Fallbacks: if a pool fit has too few rows for an hour, that hour falls
     back to the persistence forecast (hour-1 print with a widening
     lognormal spread). Quantiles are rounded to 2 decimals half-up,
     clamped to (0, 1e9], and forced non-decreasing.

Why this shape: across 37 past launches, log(peak / hour-1) correlates
-0.94 with log hour-1 viewers -- big hour-1 prints mean a flatter day,
small ones mean a much bigger peak later. Hour-1 size is by far the
strongest single signal; channels and the pre-release level add a little
more. The launch-jump (hour-1 vs pre-release median) does NOT predict the
peak ratio (r ~ 0.16), so it is not used. Peak sharpening was tested and
rejected: it made both practice and leave-one-out scores worse.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Persistence spread (log units per sqrt hour), used only as a fallback.
SIGMA = 0.15
# Spread multiplier applied to each fit's residual standard deviation.
# Tuned on the practice cases + leave-one-launch-out: 0.8 beats 1.0 and 0.9.
SPREAD_MULT = 0.8
# Blend weight of the one-feature component (the rest is the three-feature one).
BLEND_W = 0.5
# Minimum pool rows for a fit: 5 for the one-feature fit, 8 for the three-feature.
MIN_ROWS_1F = 5
MIN_ROWS_3F = 8
# Ridge penalty for the three-feature regression (standardized features).
RIDGE_LAM = 1.0
# Size-kernel bandwidth (log units).
SIZE_BW = 1.0
# Platform factor: same console-only/PC class vs different.
PLATFORM_SAME = 2.0
PLATFORM_DIFF = 0.5


def parse_ts(s):
    """ISO 8601 timestamp (with a trailing Z) to a naive UTC datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return None
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def curve_of(comp):
    """{hour: viewers} for the release day (day_start 00:00 UTC, hours 1..24)."""
    ds = parse_ts(comp["day_start"])
    out = {}
    for pt in comp.get("hourly") or []:
        if pt.get("viewers") is None or pt["viewers"] <= 0:
            continue
        t = parse_ts(pt["from"])
        h = int((t - ds).total_seconds() // 3600) + 1
        if 1 <= h <= 24:
            out[h] = pt["viewers"]
    return out


def data_jump_hour(curve):
    """Hour of the largest log increase in hours 2..10 (the US-unlock step)."""
    best_h, best_d = None, 0.0
    for h in range(2, 11):
        if h in curve and (h - 1) in curve and curve[h - 1] > 0:
            d = math.log(curve[h] / curve[h - 1])
            if d > best_d:
                best_d, best_h = d, h
    return best_h


def channels1_of(comp):
    """Channels at the release day's first hour, if the record carries it."""
    ds = comp["day_start"]
    for pt in comp.get("hourly") or []:
        if pt.get("from") == ds and pt.get("channels"):
            return pt["channels"]
    return None


def pre_stats(comp):
    """(median, max) viewers over the hourly points before day_start."""
    ds = parse_ts(comp["day_start"])
    pre = [pt["viewers"] for pt in comp.get("hourly") or []
           if pt.get("viewers") and parse_ts(pt["from"]) < ds]
    if not pre:
        return None, None
    return median(pre), max(pre)


def us_east_hour_from_text(text):
    """US East Coast unlock hour (UTC) parsed from a free-text unlock note.

    Looks for "US ... HH:00", "U.S. ... HH:00", "east coast ... HH:00" or
    "Eastern midnight ... HH:00", accepting a trailing Z or UTC.
    """
    if not text:
        return None
    m = re.search(r"(?:US|U\.S\.|east coast|Eastern midnight)[^0-9]{0,40}?"
                  r"(\d{1,2}):00(?:Z| UTC)", text)
    if m:
        return int(m.group(1))
    return None


def dst_us_east_hour(day_start):
    """US Eastern midnight in UTC: 04:00Z during daylight saving
    (2nd Sunday of March to 1st Sunday of November), else 05:00Z."""
    d = parse_ts(day_start).date()
    y = d.year

    def nth_sunday(month, n):
        dt = datetime(y, month, 1)
        add = (6 - dt.weekday()) % 7
        return (dt + timedelta(days=add + 7 * (n - 1))).date()

    start, end = nth_sunday(3, 2), nth_sunday(11, 1)
    return 4 if start <= d < end else 5


def target_jump_hour(case):
    """The target launch's US-unlock jump hour (1-based hour of the day).

    GTA VI's unlock is a known fact of this competition (rules: US east
    coast at 05:00 UTC, so the step lands in hour 6). For other launches
    the case notes state it; otherwise the daylight-saving rule gives it.
    """
    name = (case.get("game") or {}).get("name", "")
    if "Grand Theft Auto VI" in name:
        return 6
    h = us_east_hour_from_text(case.get("notes") or "")
    if h is None:
        h = dst_us_east_hour(case["day_start"])
    return h + 1


def target_pre_stats(case):
    """(median, max) viewers over the 14 days before the target's day_start.

    Uses the case's own history_hourly. The final case's history only
    starts two days before launch, so when it is short the target game's
    own pre-release history (shipped as a comparable in every case) fills
    the same 14-day window; timestamps already present win.
    """
    ds = parse_ts(case["day_start"])
    window_start = ds - timedelta(days=14)
    by_ts = {}
    for pt in case.get("history_hourly") or []:
        if pt.get("viewers") and window_start <= parse_ts(pt["from"]) < ds:
            by_ts[pt["from"]] = pt["viewers"]
    if len(by_ts) < 100:
        name = (case.get("game") or {}).get("name", "")
        for comp in case.get("comparables") or []:
            if name and name.lower() in (comp.get("name") or "").lower():
                for pt in comp.get("hourly") or []:
                    if (pt.get("viewers") and window_start <= parse_ts(pt["from"]) < ds
                            and pt["from"] not in by_ts):
                        by_ts[pt["from"]] = pt["viewers"]
    vals = list(by_ts.values())
    if not vals:
        return None, None
    return median(vals), max(vals)


def build_pool(case):
    """Pool entries: (curve, meta) for each usable comparable launch."""
    pool = []
    for comp in case.get("comparables") or []:
        c = curve_of(comp)
        if 1 not in c or len(c) < 20:
            continue
        platforms = (comp.get("platforms") or "").lower()
        pre_med, _pre_max = pre_stats(comp)
        pool.append((c, {
            "p1": c[1],
            "console_only": "pc" not in platforms,
            "jump_h": data_jump_hour(c),
            "channels1": channels1_of(comp),
            "pre_med": pre_med,
        }))
    return pool


def size_platform_weight(meta, target):
    """Similarity weight: size kernel times a platform-match factor."""
    if not meta.get("p1") or not target.get("p1"):
        return 0.0
    w = math.exp(-abs(math.log(meta["p1"]) - math.log(target["p1"])) / SIZE_BW)
    w *= PLATFORM_SAME if meta.get("console_only") == target.get("console_only") else PLATFORM_DIFF
    return w


def wls_1feat(xs, ys, ws, x0):
    """Weighted least squares y ~ a + b*x; predict at x0.

    Returns (prediction, weighted residual standard deviation).
    """
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        sd = (sum(w * (y - my) ** 2 for w, y in zip(ws, ys)) / sw) ** 0.5
        return my, sd
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = (sum(w * r * r for w, r in zip(ws, resid)) / sw) ** 0.5
    return a + b * x0, sd


def solve_linear(A, rhs):
    """Solve a small dense linear system by Gaussian elimination with
    partial pivoting. Returns None if the system is singular."""
    n = len(rhs)
    M = [row[:] + [rhs[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(n):
            if r != col:
                fct = M[r][col] / pv
                for cc in range(col, n + 1):
                    M[r][cc] -= fct * M[col][cc]
    return [M[i][n] / M[i][i] for i in range(n)]


def wls_multi(rows, target_feats, keys):
    """Ridge-regularized weighted regression of y on standardized features.

    rows: [(features dict, y, weight)]. target_feats: the target's features
    (same keys). Returns (prediction, weighted residual SD) or None when
    the system cannot be solved.
    """
    X, y, w = [], [], []
    for f, yv, wv in rows:
        X.append([f[k] for k in keys])
        y.append(yv)
        w.append(wv)
    nfeat = len(keys)
    sw = sum(w)
    means = [sum(wi * xi[j] for wi, xi in zip(w, X)) / sw for j in range(nfeat)]
    sds = [(sum(wi * (xi[j] - means[j]) ** 2 for wi, xi in zip(w, X)) / sw) ** 0.5
           or 1.0 for j in range(nfeat)]
    Xn = [[(xi[j] - means[j]) / sds[j] for j in range(nfeat)] for xi in X]
    tx = [(target_feats[k] - means[j]) / sds[j] for j, k in enumerate(keys)]
    A = [[sum(wi * Xn[i][a] * Xn[i][b] for i, wi in enumerate(w))
          + (RIDGE_LAM if a == b else 0.0) for b in range(nfeat)]
         for a in range(nfeat)]
    rhs = [sum(wi * Xn[i][a] * y[i] for i, wi in enumerate(w)) for a in range(nfeat)]
    beta = solve_linear(A, rhs)
    if beta is None:
        return None
    ybar = sum(wi * yi for wi, yi in zip(w, y)) / sw
    pred = ybar + sum(beta[j] * tx[j] for j in range(nfeat))
    resid = [y[i] - (ybar + sum(beta[j] * Xn[i][j] for j in range(nfeat)))
             for i in range(len(y))]
    sd = (sum(wi * r * r for wi, r in zip(w, resid)) / sw) ** 0.5
    return pred, sd


def persistence_quantiles(p1, h):
    """Fallback forecast: hour-1 print with a widening lognormal spread."""
    return [max(1.0, p1 * math.exp(z * SIGMA * math.sqrt(h - 1))) for z in Z]


def component_1feat(p1, target_jump, pool, target, h):
    """One-feature aligned local-linear forecast for hour h, or None."""
    r = h - target_jump
    xs, ys, ws = [], [], []
    for curve, meta in pool:
        jc = meta.get("jump_h")
        if jc is None:
            continue
        ch = jc + r
        if ch in curve and 1 in curve and ch >= 2:
            xs.append(math.log(curve[1]))
            ys.append(math.log(curve[ch] / curve[1]))
            ws.append(size_platform_weight(meta, target))
    if len(xs) < MIN_ROWS_1F:
        return None
    pred, sd = wls_1feat(xs, ys, ws, math.log(p1))
    sd = max(sd, 0.10) * SPREAD_MULT
    return [max(1.0, p1 * math.exp(pred + z * sd)) for z in Z]


def component_3feat(p1, target_jump, pool, target, target_feats, h):
    """Three-feature aligned ridge forecast for hour h, or None."""
    r = h - target_jump
    rows = []
    for curve, meta in pool:
        jc = meta.get("jump_h")
        if jc is None:
            continue
        ch = jc + r
        if ch in curve and 1 in curve and ch >= 2:
            f = {"p1": math.log(curve[1])}
            f["ch1"] = math.log(meta["channels1"]) if meta.get("channels1") else None
            f["premed"] = (math.log(curve[1] / meta["pre_med"])
                           if meta.get("pre_med") else None)
            if any(v is None for v in f.values()):
                continue
            rows.append((f, math.log(curve[ch] / curve[1]),
                         size_platform_weight(meta, target)))
    if len(rows) < MIN_ROWS_3F:
        return None
    out = wls_multi(rows, target_feats, ["p1", "ch1", "premed"])
    if out is None:
        return None
    pred, sd = out
    sd = max(sd, 0.10) * SPREAD_MULT
    return [max(1.0, p1 * math.exp(pred + z * sd)) for z in Z]


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    ch1 = first.get("channels")

    # The competition's cases are all console-only launches at launch
    # (the four practice games and GTA VI, PS5/Xbox only, no PC).
    target = {"p1": p1, "console_only": True}
    target_jump = target_jump_hour(case)
    pool = build_pool(case)
    pre_med, _pre_max = target_pre_stats(case)
    target_feats = {"p1": math.log(p1)}
    if ch1:
        target_feats["ch1"] = math.log(ch1)
    if pre_med:
        target_feats["premed"] = math.log(p1 / pre_med)

    hours = []
    for row in case.get("forecast") or []:
        h = row["hour"]
        q1 = component_1feat(p1, target_jump, pool, target, h)
        q3 = (component_3feat(p1, target_jump, pool, target, target_feats, h)
              if "ch1" in target_feats and "premed" in target_feats else None)
        if q1 is None and q3 is None:
            q = persistence_quantiles(p1, h)
        elif q1 is None:
            q = q3
        elif q3 is None:
            q = q1
        else:
            q = [math.exp(BLEND_W * math.log(q1[k]) + (1 - BLEND_W) * math.log(q3[k]))
                 for k in range(5)]
        # Round half-up to 2 decimals, clamp, force non-decreasing.
        q = [round2(min(1e9, max(1.0, v))) for v in q]
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        hours.append({"hour": h, "q": q})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
