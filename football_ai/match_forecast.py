"""Transparent, uncalibrated pre-match scenarios; not betting advice or shot xG."""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import poisson, skellam

LABELS = {"goals": "Goles", "corners": "Córners", "yellow_cards": "Amarillas",
          "shots": "Tiros", "shots_on_target": "Tiros al arco", "fouls": "Faltas"}


def count_scenarios(estimates: dict) -> dict:
    """Independent Poisson marginals, 10–90% predictive quantiles.

    The range contains >=80% mass under the assumed distribution, not an
    empirically demonstrated 80% coverage. Parameter uncertainty is omitted.
    """
    output = {}
    for key, label in LABELS.items():
        home, away = estimates.get(f"home_{key}"), estimates.get(f"away_{key}")
        if any(v is None or not math.isfinite(v) or not 0 <= v <= 200 for v in (home, away)):
            continue
        total = home + away
        output[key] = {"label": label, "home": home, "away": away, "total": total,
                       "low": int(poisson.ppf(.1, total)), "high": int(poisson.ppf(.9, total)),
                       "home_low": int(poisson.ppf(.1, home)), "home_high": int(poisson.ppf(.9, home)),
                       "away_low": int(poisson.ppf(.1, away)), "away_high": int(poisson.ppf(.9, away))}
    return output


def score_scenarios(home: float | None, away: float | None) -> dict | None:
    if any(v is None or not math.isfinite(v) or not 0 <= v <= 50 for v in (home, away)):
        return None
    # scipy.skellam requires strictly positive means, so handle real zeros.
    if home == 0:
        p = [0.0, math.exp(-away), -math.expm1(-away)]
    elif away == 0:
        p = [-math.expm1(-home), math.exp(-home), 0.0]
    else:
        p = [float(skellam.sf(0, home, away)), float(skellam.pmf(0, home, away)),
             float(skellam.cdf(-1, home, away))]
    max_goal = max(6, int(poisson.ppf(.99999, max(home, away))))
    axis = np.arange(max_goal + 1)
    joint = np.outer(poisson.pmf(axis, home), poisson.pmf(axis, away))
    scores = [{"home": int(i), "away": int(j), "probability": float(joint[i,j])}
              for i,j in np.ndindex(joint.shape)]
    scores.sort(key=lambda item: item["probability"], reverse=True)
    return {"home_win": p[0], "draw": p[1], "away_win": p[2], "scores": scores[:6],
            "both_score": float((-math.expm1(-home))*(-math.expm1(-away))),
            "at_least_three": float(poisson.sf(2, home + away)), "joint_mass": float(joint.sum())}
