import numpy as np
import pandas as pd

try:
    from entropy_weighting import INDICATORS, make_indicators, normalize, entropy_weights
except ImportError:
    from src.entropy_weighting import INDICATORS, make_indicators, normalize, entropy_weights

# combined weighting: half from entropy, half shared equally
# (so one changing indicator cannot take all the weight)
ENTROPY_SHARE = 0.5

# fixed ranges for the score of each indicator (1 = good, 0 = bad)
TEMP_RISE_BAD = 20.0    # C above a healthy panel
DUST_BAD = 30.0         # % (DSM501B)

# no grading in the dark (power is 0 at night, that is not a fault)
MIN_GRADING_LUX = 1000.0


def trap(x, a, b, c, d):
    # trapezoid membership: 0 before a, rises a->b, 1 from b to c, falls c->d
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    y = np.where((x >= b) & (x <= c), 1.0, y)
    if b > a:
        y = np.where((x >= a) & (x < b), (x - a) / (b - a), y)
    if d > c:
        y = np.where((x > c) & (x <= d), (d - x) / (d - c), y)
    return y


def tri(x, a, b, c):
    # triangle membership: peak at b
    return trap(x, a, b, b, c)


def indicator_scores(ind):
    # every indicator on the same scale: 1 = good, 0 = bad
    scores = pd.DataFrame(index=ind.index)
    scores["power_ratio"] = ind["power_ratio"]
    scores["voltage_ratio"] = ind["voltage_ratio"]
    scores["current_ratio"] = ind["current_ratio"]
    scores["temp_rise"] = ((TEMP_RISE_BAD - ind["temp_rise"]) / TEMP_RISE_BAD).clip(0, 1)
    scores["dust_level"] = ((DUST_BAD - ind["dust_level"]) / DUST_BAD).clip(0, 1)
    return scores


def combined_weights(ind):
    # entropy weights mixed with equal weights
    w = entropy_weights(normalize(ind))
    return ENTROPY_SHARE * w + (1 - ENTROPY_SHARE) / len(INDICATORS)


def fuzzy_grade(S, E, T, D):
    # S = weighted score (0-100), E = electrical, T = thermal, D = dust (1 = good)

    # ---- fuzzify ----
    s_low, s_med, s_high = trap(S, 0, 0, 30, 50), tri(S, 30, 55, 80), trap(S, 65, 85, 100, 100)
    e_bad, e_ok, e_good = trap(E, 0, 0, 0.3, 0.5), tri(E, 0.3, 0.6, 0.85), trap(E, 0.7, 0.9, 1, 1)
    t_hot, t_normal = trap(T, 0, 0, 0.25, 0.5), trap(T, 0.25, 0.5, 1, 1)
    d_dirty, d_clean = trap(D, 0, 0, 0.4, 0.6), trap(D, 0.4, 0.6, 1, 1)

    # ---- rules (AND = min) ----
    rules = np.vstack([
        e_bad,                                                   # R1 -> FAULTY
        t_hot,                                                   # R2 -> FAULTY
        np.minimum(s_low, 1 - e_good),                           # R3 -> FAULTY
        np.minimum(s_med, 1 - e_bad),                            # R4 -> DEGRADED
        np.minimum(d_dirty, 1 - e_bad),                          # R5 -> DEGRADED
        e_ok,                                                    # R6 -> DEGRADED
        np.minimum.reduce([s_high, e_good, t_normal, d_clean]),  # R7 -> HEALTHY
    ])

    # ---- combine rules of the same output (OR = max) ----
    faulty = rules[0:3].max(axis=0)
    degraded = rules[3:6].max(axis=0)
    healthy = rules[6]

    # ---- defuzzify (centroid over 0..100) ----
    y = np.arange(101)[:, None]
    mu = np.maximum.reduce([
        np.minimum(faulty, trap(y, 0, 0, 20, 40)),
        np.minimum(degraded, tri(y, 30, 50, 70)),
        np.minimum(healthy, trap(y, 60, 80, 100, 100)),
    ])
    total = mu.sum(axis=0)
    health = np.where(total > 1e-6, (mu * y).sum(axis=0) / np.maximum(total, 1e-9), S)

    return health


def health_status(score):
    # NaN score = not graded (too dark)
    score = np.asarray(score, dtype=float)
    return np.where(np.isnan(score), "NOT_GRADED",
           np.where(score >= 70, "HEALTHY",
           np.where(score >= 40, "DEGRADED", "FAULTY")))


def run_fuzzy_stage(df):
    # input: output of run_kalman_stage(), output: the same rows with the grade
    df = df.copy()
    ind = make_indicators(df)
    scores = indicator_scores(ind)
    weights = combined_weights(ind)

    S = (scores[list(INDICATORS)] * weights[list(INDICATORS)]).sum(axis=1) * 100
    E = scores[["voltage_ratio", "current_ratio"]].min(axis=1)
    health = fuzzy_grade(S.to_numpy(), E.to_numpy(),
                         scores["temp_rise"].to_numpy(), scores["dust_level"].to_numpy())

    # too dark to grade: no score (left out when sent to the backend)
    dark = (df["irradiance_lux"] < MIN_GRADING_LUX).to_numpy()
    health = np.where(dark, np.nan, health)

    df["weighted_score"] = S.where(~dark).round(2)
    df["health_score"] = np.round(health, 2)
    df["health_status"] = health_status(health)
    return df