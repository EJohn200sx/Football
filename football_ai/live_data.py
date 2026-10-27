"""API-Football adapter. User credentials stay server-side, in memory/.env."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import time
import pandas as pd
import requests

API_BASE = "https://v3.football.api-sports.io"
MAPPING = {"Corner Kicks": "corners", "Yellow Cards": "yellow_cards", "Red Cards": "red_cards",
    "Total Shots": "shots", "Shots on Goal": "shots_on_target", "Ball Possession": "possession",
    "expected_goals": "xg"}


class ProviderError(RuntimeError):
    pass


def statistic(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).rstrip("%"))
        return number if math.isfinite(number) and number >= 0 else None
    except (ValueError, TypeError):
        return None


def utc(value) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True)


def normalize_fixture(item: dict) -> dict:
    return {"id": int(item["fixture"]["id"]), "date": item["fixture"]["date"],
            "status": item["fixture"]["status"]["short"],
            "competition": item["league"]["name"], "league_id": int(item["league"]["id"]),
            "season": str(item["league"]["season"]),
            "home_team": item["teams"]["home"]["name"], "away_team": item["teams"]["away"]["name"],
            "home_id": int(item["teams"]["home"]["id"]), "away_id": int(item["teams"]["away"]["id"])}


def completed_row(item: dict, stats: list[dict]) -> dict:
    fixture = normalize_fixture(item)
    if fixture["status"] != "FT":
        raise ValueError("Sólo se usan partidos finalizados de 90 minutos (FT)")
    fulltime = item.get("score", {}).get("fulltime", {})
    goals = fulltime if fulltime.get("home") is not None and fulltime.get("away") is not None else item.get("goals", {})
    hg, ag = statistic(goals.get("home")), statistic(goals.get("away"))
    if hg is None or ag is None or not hg.is_integer() or not ag.is_integer():
        raise ValueError("Marcador final ausente o inválido")
    row = {**fixture, "match_id": f"api-football-{fixture['id']}", "date": utc(fixture["date"]).date().isoformat(),
           "home_goals": int(hg), "away_goals": int(ag), "source": "API-Football", "is_simulated": False}
    by_team = {int(x["team"]["id"]): x.get("statistics", []) for x in stats}
    for side in ["home", "away"]:
        fields = {x["type"]: x.get("value") for x in by_team.get(fixture[f"{side}_id"], [])}
        for original, name in MAPPING.items():
            row[f"{side}_{name}"] = statistic(fields.get(original))
    return row


@dataclass
class FootballAPI:
    key: str = field(repr=False)
    session: object = field(default_factory=requests.Session, repr=False)
    cache: dict = field(default_factory=dict, repr=False)
    min_interval: float = 6.1  # Conservative spacing; no concurrent bursts.
    last_call: float = 0

    def get(self, endpoint: str, params: dict, ttl: int = 900) -> list:
        if not self.key.strip():
            raise ProviderError("Configura tu clave API-Football en Conectar datos actuales.")
        if endpoint not in {"teams", "fixtures", "fixtures/statistics"}:
            raise ProviderError("Endpoint no admitido")
        cache_key = (endpoint, tuple(sorted(params.items())))
        cached = self.cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < ttl:
            return cached[1]
        pause = self.min_interval - (time.monotonic() - self.last_call)
        if pause > 0:
            time.sleep(pause)
        try:
            self.last_call = time.monotonic()
            response = self.session.get(f"{API_BASE}/{endpoint}", params=params,
                headers={"x-apisports-key": self.key}, timeout=(10, 30), allow_redirects=False)
            if response.status_code in {401, 403}:
                raise ProviderError("La clave o el plan no permite esta consulta. Revisa tu cuenta API-Football.")
            if response.status_code == 429:
                raise ProviderError("Límite del proveedor alcanzado. Conserva lo cargado e inténtalo más tarde.")
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            raise ProviderError("No se pudo obtener una respuesta válida del proveedor. No se generaron datos de reemplazo.") from None
        if not isinstance(payload, dict):
            raise ProviderError("Respuesta del proveedor no reconocida")
        if payload.get("errors"):
            # Never echo raw provider errors: they can include account/request details.
            raise ProviderError("El proveedor rechazó la consulta: revisa clave, cuota y acceso a la temporada actual.")
        records = payload.get("response")
        if not isinstance(records, list):
            raise ProviderError("Respuesta sin lista de resultados")
        if payload.get("paging", {}).get("total", 1) > 1:
            raise ProviderError("Consulta con más de una página; precisa el nombre del equipo.")
        self.cache[cache_key] = (time.monotonic(), records)
        return records

    def search(self, query: str) -> list[dict]:
        if not 3 <= len(query.strip()) <= 80:
            raise ValueError("Escribe de 3 a 80 caracteres")
        return [{"id": int(r["team"]["id"]), "name": r["team"]["name"],
                 "country": r["team"].get("country", "")}
                for r in self.get("teams", {"search": query.strip()}, ttl=3600)]

    def report(self, team_id: int, count: int = 5, now=None, progress=None) -> dict:
        if count not in {5, 10}:
            raise ValueError("Ventana no válida")
        now = utc(now if now is not None else datetime.now(timezone.utc))
        upcoming = [normalize_fixture(r) for r in self.get("fixtures", {"team": int(team_id), "next": 5})]
        upcoming = sorted((r for r in upcoming if r["status"] == "NS" and utc(r["date"]) > now), key=lambda r: utc(r["date"]))
        if not upcoming:
            return {"fixture": None, "history": pd.DataFrame(), "warnings": ["No hay un próximo partido con fecha confirmada accesible en tu plan."], "fetched_at": now.isoformat()}
        fixture = upcoming[0]
        seen, completed = set(), []
        warnings = []
        for ident in [fixture["home_id"], fixture["away_id"]]:
            for r in self.get("fixtures", {"team": ident, "league": fixture["league_id"], "last": count, "status": "FT"}):
                fid = r["fixture"]["id"]
                if fid in seen or r["fixture"]["status"]["short"] != "FT" or utc(r["fixture"]["date"]) >= now.normalize():
                    continue
                seen.add(fid); completed.append(r)
        rows, stats_unavailable = [], False
        for i, match in enumerate(completed):
            statistics = []
            if not stats_unavailable:
                try:
                    statistics = self.get("fixtures/statistics", {"fixture": int(match["fixture"]["id"])}, ttl=21600)
                except ProviderError as exc:
                    warnings.append(str(exc)); stats_unavailable = True
            try:
                rows.append(completed_row(match, statistics))
            except ValueError:
                warnings.append("Se omitió un partido sin marcador final válido.")
            if progress:
                progress((i + 1) / len(completed))
        return {"fixture": fixture, "history": pd.DataFrame(rows), "warnings": warnings,
                "fetched_at": now.isoformat(), "count": count}
