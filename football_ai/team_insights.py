"""Simple, explainable team summaries. No model training required."""
from __future__ import annotations
import unicodedata
import numpy as np
import pandas as pd

STAT_NAMES = {"goals_for": "Goles a favor", "goals_against": "Goles recibidos",
    "corners": "Córners a favor", "corners_against": "Córners concedidos",
    "yellow_cards": "Amarillas", "red_cards": "Rojas", "shots": "Tiros",
    "shots_on_target": "Tiros al arco", "shots_against": "Tiros concedidos",
    "shots_on_target_against": "Tiros al arco concedidos", "fouls": "Faltas", "possession": "Posesión (%)",
    "xg": "xG del proveedor", "xga": "xGA del proveedor"}


def search_text(text: str) -> str:
    return " ".join("".join(c for c in unicodedata.normalize("NFKD", text.casefold())
                            if not unicodedata.combining(c)).split())


def find_teams(history: pd.DataFrame, query: str) -> list[tuple[str, str]]:
    query = search_text(query)
    query = {"barca": "barcelona", "barça": "barcelona", "man city": "manchester city",
             "man united": "manchester united"}.get(query, query)
    if len(query) < 2 or history.empty:
        return []
    pairs = set(zip(history.competition, history.home_team)) | set(zip(history.competition, history.away_team))
    return sorted((league, team) for league, team in pairs if query in search_text(team))


def recent_matches(history: pd.DataFrame, team: str, competition: str | None, cutoff,
                   count: int = 10, venue: str | None = None) -> pd.DataFrame:
    if count not in {5, 10} or venue not in {None, "home", "away"}:
        raise ValueError("Ventana o localía inválida")
    if history.empty:
        return pd.DataFrame(columns=["date", "opponent", "venue", "result", *STAT_NAMES])
    dates = pd.to_datetime(history.date, utc=True)
    # Exclude the entire reference day conservatively, including live fixtures.
    boundary = pd.Timestamp(cutoff)
    boundary = (boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")).normalize()
    selected = history[(dates < boundary) & ((history.competition == competition) if competition else True)
                       & ((history.home_team == team) | (history.away_team == team))].copy()
    selected["_date"] = dates[selected.index]
    selected = selected.drop_duplicates(subset=["date", "home_team", "away_team"])
    if "is_simulated" in selected and selected.is_simulated.nunique() > 1:
        raise ValueError("No se mezclan datos reales y simulados")
    if venue:
        selected = selected[selected[f"{venue}_team"] == team]
    selected = selected.sort_values("_date").tail(count)
    rows = []
    for r in selected.to_dict("records"):
        side, other = ("home", "away") if r["home_team"] == team else ("away", "home")
        gf, ga = r[f"{side}_goals"], r[f"{other}_goals"]
        if pd.isna(gf) or pd.isna(ga):
            continue
        row = {"date": pd.Timestamp(r["_date"]).date(), "opponent": r[f"{other}_team"],
            "venue": "Local" if side == "home" else "Visitante", "goals_for": gf, "goals_against": ga,
            "result": "V" if gf > ga else "E" if gf == ga else "D"}
        for field in ["corners", "yellow_cards", "red_cards", "shots", "shots_on_target", "possession", "xg", "fouls"]:
            row[field] = r.get(f"{side}_{field}")
        row["shots_against"] = r.get(f"{other}_shots")
        row["shots_on_target_against"] = r.get(f"{other}_shots_on_target")
        row["corners_against"], row["xga"] = r.get(f"{other}_corners"), r.get(f"{other}_xg")
        rows.append(row)
    return pd.DataFrame(rows, columns=["date", "opponent", "venue", "result", *STAT_NAMES])


def summarize(recent: pd.DataFrame) -> dict:
    summary = {"n": len(recent), "metrics": {}, "form": " ".join(recent.result.astype(str)) if len(recent) else ""}
    for name in STAT_NAMES:
        values = pd.to_numeric(recent.get(name, pd.Series(dtype=float)), errors="coerce")
        values = values[np.isfinite(values) & (values >= 0)]
        summary["metrics"][name] = {"mean": float(values.mean()) if len(values) else None,
                                    "available": len(values), "total": len(recent)}
    return summary


def matchup_estimates(home: dict, away: dict, *, stale: bool = False) -> dict:
    """Descriptive heuristic, not xG or a calibrated prediction model."""
    def mean(side, key):
        stat = side["metrics"][key]
        return stat["mean"] if stat["available"] >= 3 and not stale else None
    def blend(a, b):
        return (a + b) / 2 if a is not None and b is not None else None
    return {"home_goals": blend(mean(home, "goals_for"), mean(away, "goals_against")),
            "away_goals": blend(mean(away, "goals_for"), mean(home, "goals_against")),
            "home_corners": blend(mean(home, "corners"), mean(away, "corners_against")),
            "away_corners": blend(mean(away, "corners"), mean(home, "corners_against")),
            "home_yellow_cards": mean(home, "yellow_cards"), "away_yellow_cards": mean(away, "yellow_cards"),
            "home_shots": blend(mean(home, "shots"), mean(away, "shots_against")),
            "away_shots": blend(mean(away, "shots"), mean(home, "shots_against")),
            "home_shots_on_target": blend(mean(home, "shots_on_target"), mean(away, "shots_on_target_against")),
            "away_shots_on_target": blend(mean(away, "shots_on_target"), mean(home, "shots_on_target_against")),
            "home_fouls": mean(home, "fouls"), "away_fouls": mean(away, "fouls")}
