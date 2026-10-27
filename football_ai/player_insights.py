"""Player-level recent form and explicitly conditional, uncalibrated scenarios."""
from __future__ import annotations
from collections import Counter
import math
import pandas as pd
from scipy.stats import beta

PLAYER_STATS = {"totalGoals": "goals", "goalAssists": "assists", "totalShots": "shots",
                "shotsOnTarget": "shots_on_target", "yellowCards": "yellow_cards",
                "foulsCommitted": "fouls", "saves": "saves"}
PLAYER_EVENTS = {
    "goal": ("Marcar al menos un gol", "goals", 1),
    "assist": ("Dar al menos una asistencia", "assists", 1),
    "shot_on_target": ("Hacer al menos un tiro al arco", "shots_on_target", 1),
    "two_on_target": ("Hacer dos o más tiros al arco", "shots_on_target", 2),
    "two_shots": ("Hacer dos o más tiros", "shots", 2),
    "yellow_card": ("Recibir al menos una amarilla", "yellow_cards", 1),
    "two_saves": ("Realizar dos o más atajadas", "saves", 2),
}
PLAYER_COLUMNS = ["match_id", "date", "competition", "team_id", "team", "player_id", "player",
                  "position", "played", "starter", "minutes", *PLAYER_STATS.values()]


def number(value, maximum=200):
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
        return v if math.isfinite(v) and 0 <= v <= maximum and v.is_integer() else None
    except (TypeError, ValueError):
        return None


def player_rows(summary: dict, fixture: dict) -> list[dict]:
    """Only explicit statistics; bench membership is not an appearance."""
    if fixture.get("status") != "STATUS_FULL_TIME":
        raise ValueError("Sólo jugadores de partidos finalizados FT")
    if str((summary.get("header") or {}).get("id")) != str(fixture["id"]):
        raise ValueError("Resumen de otro partido")
    rows = []
    teams = {str(fixture[f"{s}_id"]): fixture[f"{s}_team"] for s in ("home", "away")}
    for roster in summary.get("rosters") or []:
        team_id = str((roster.get("team") or {}).get("id", ""))
        if team_id not in teams:
            continue
        members = roster.get("roster") or []
        ids = Counter(str((r.get("athlete") or {}).get("id", "")) for r in members)
        for r in members:
            athlete = r.get("athlete") or {}
            ident = str(athlete.get("id", ""))
            name = athlete.get("displayName") or athlete.get("fullName")
            if not ident or not name or ids[ident] != 1:
                continue
            fields = {s.get("name"): s.get("value", s.get("displayValue")) for s in r.get("stats") or []}
            appearances = number(fields.get("appearances"), 1)
            # Explicit appearance count has priority. Active means roster availability, not minutes played.
            played = appearances == 1 if appearances is not None else (r.get("starter") is True or r.get("subbedIn") is True)
            row = {"match_id": f"community-{fixture['id']}", "date": fixture["date"],
                   "competition": fixture["competition"], "team_id": int(team_id), "team": teams[team_id],
                   "player_id": ident, "player": str(name), "position": (r.get("position") or {}).get("abbreviation", ""),
                   "played": played, "starter": r.get("starter") is True,
                   "minutes": number(fields.get("minutesPlayed"), 130)}
            for source, target in PLAYER_STATS.items():
                row[target] = number(fields.get(source))
            if row["shots"] is not None and row["shots_on_target"] is not None and row["shots_on_target"] > row["shots"]:
                row["shots_on_target"] = None
            rows.append(row)
    return rows


def rank_players(players: pd.DataFrame, history: pd.DataFrame, fixture: dict,
                 cutoff, event: str = "goal", min_appearances: int = 3) -> pd.DataFrame:
    """Jeffreys-smoothed frequency conditional on another comparable appearance.

    p = (games_with_event + .5)/(games_with_valid_stat + 1).
    10th/90th Beta quantiles express parameter uncertainty under iid assumptions,
    NOT validated prediction coverage. No minutes or lineup probabilities invented.
    """
    if event not in PLAYER_EVENTS or min_appearances < 3:
        raise ValueError("Evento o muestra mínima inválidos")
    if players.empty or history.empty:
        return pd.DataFrame()
    boundary = pd.Timestamp(cutoff)
    boundary = (boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")).normalize()
    data = players.copy()
    data["_date"] = pd.to_datetime(data.date, utc=True, errors="coerce")
    data = data[(data._date < boundary) & (data._date >= boundary-pd.Timedelta(days=180))]
    data = data.drop_duplicates(["match_id", "team_id", "player_id"])
    match_dates = pd.to_datetime(history.date, utc=True, errors="coerce")
    previous = history[match_dates < boundary].copy()
    previous["_date"] = match_dates[previous.index]
    _, stat, threshold = PLAYER_EVENTS[event]
    output = []
    for side in ("home", "away"):
        tid = int(fixture[f"{side}_id"])
        if "home_id" not in previous or "away_id" not in previous:
            continue
        matches = previous[(previous.home_id == tid) | (previous.away_id == tid)]
        if matches.empty:
            continue
        latest_date = matches._date.max().normalize()
        if (boundary-latest_date).days > 90:
            continue
        current_ids = set(data[(data.team_id == tid) & (data._date.dt.normalize() == latest_date)].player_id)
        for ident, group in data[(data.team_id == tid) & data.player_id.isin(current_ids)].groupby("player_id"):
            group = group.sort_values("_date")
            appearances = group[group.played.eq(True)]
            known_positions = appearances[~appearances.position.isin(["", "SUB"])].position
            position = known_positions.iloc[-1] if not known_positions.empty else group.iloc[-1].position
            if event == "two_saves" and str(position).upper() not in {"G", "GK"}:
                continue
            values = pd.to_numeric(appearances[stat], errors="coerce")
            values = values[values.notna() & values.ge(0) & values.le(200) & values.mod(1).eq(0)]
            n = len(values)
            if n < min_appearances:
                continue
            hits = int(values.ge(threshold).sum())
            alpha, bet = hits + .5, n-hits + .5
            last = group.iloc[-1]
            output.append({"player_id": str(ident), "player": last.player, "team_id": tid,
                "team": fixture[f"{side}_team"], "side": side, "event": event, "metric": stat,
                "threshold": threshold, "probability": alpha/(alpha+bet),
                "lower": float(beta.ppf(.1, alpha, bet)), "upper": float(beta.ppf(.9, alpha, bet)),
                "hits": hits, "sample": n, "mean": float(values.mean()), "total": float(values.sum()),
                "appearances": len(appearances), "starts": int(appearances.starter.eq(True).sum()),
                "last_seen": last._date.date().isoformat(), "position": position,
                "minutes_available": int(appearances.minutes.notna().sum())})
    if not output:
        return pd.DataFrame()
    return pd.DataFrame(output).sort_values(["probability", "sample", "mean", "player"],
                                             ascending=[False,False,False,True]).reset_index(drop=True)
