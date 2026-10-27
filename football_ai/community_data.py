"""Read-only client for the documented, no-key worldcup26.ir community API.

The public service is not an official league feed and has no availability SLA.
Use only its documented API; never scrape or call its upstream providers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import time
import re
from threading import RLock

import pandas as pd
import requests

from football_ai.live_data import ProviderError, statistic, utc
from football_ai.team_insights import search_text
from football_ai.player_insights import player_rows, PLAYER_COLUMNS

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://worldcup26.ir/get/soccer"
LEAGUES = {"LaLiga": "esp.1", "Premier League": "eng.1", "Serie A": "ita.1",
           "Bundesliga": "ger.1", "Ligue 1": "fra.1"}
CUPS = {"esp.1": ["esp.copa_del_rey", "esp.super_cup"],
        "eng.1": ["eng.fa", "eng.league_cup", "eng.charity"],
        "ger.1": ["ger.dfb_pokal"], "ita.1": ["ita.coppa_italia"], "fra.1": ["fra.coupe_de_france"]}
STAT_MAP = {"wonCorners": "corners", "yellowCards": "yellow_cards", "redCards": "red_cards",
            "totalShots": "shots", "shotsOnTarget": "shots_on_target", "possessionPct": "possession",
            "expectedGoals": "xg", "foulsCommitted": "fouls"}


def stats_dict(items: list) -> dict:
    return {r.get("name"): statistic(r.get("value", r.get("displayValue")))
            for r in items if isinstance(r, dict)}


def roster_cards(rosters: list, team_id: int) -> dict:
    """Aggregate explicit player totals only for a complete, unique match roster.

    A missing event is NOT a zero card. Staff cards are not player cards.
    """
    for side in rosters:
        if str(side.get("team", {}).get("id")) != str(team_id):
            continue
        roster = side.get("roster") or []
        ids = [str(r.get("athlete", {}).get("id", "")) for r in roster]
        if len(roster) < 11 or not all(ids) or len(ids) != len(set(ids)):
            return {}
        if sum(r.get("starter") is True for r in roster) != 11:
            return {}
        result = {}
        for key in ("yellowCards", "redCards"):
            values = [stats_dict(r.get("stats", [])).get(key) for r in roster]
            if all(v is not None and float(v).is_integer() for v in values):
                result[key] = sum(values)
        return result
    return {}


def parse_event(event: dict, league: str, competition: str) -> dict:
    comp = event["competitions"][0]
    teams = {c["homeAway"]: c for c in comp["competitors"]}
    status = (comp.get("status") or event.get("status") or {}).get("type", {})
    row = {"id": int(event["id"]), "date": utc(event["date"]).isoformat(),
           "league_id": league, "competition": competition,
           "season": str((event.get("season") or {}).get("year", "")),
           "status": status.get("name", ""),
           "venue": (comp.get("venue") or {}).get("fullName", ""),
           "time_confirmed": not bool(event.get("timeValid") is False or comp.get("timeValid") is False)}
    for side in ("home", "away"):
        team = teams[side]["team"]
        row[f"{side}_id"] = int(team["id"])
        row[f"{side}_team"] = team.get("displayName") or team["name"]
        row[f"{side}_logo"] = team.get("logo", "")
    if row["home_id"] == row["away_id"]:
        raise ValueError("Equipos idénticos en el encuentro")
    return row


def completed_event(event: dict, league: str, competition: str, summary: dict | None = None) -> dict:
    fixture = parse_event(event, league, competition)
    if fixture["status"] != "STATUS_FULL_TIME":
        raise ValueError("Se requieren resultados FT de 90 minutos")
    summary = summary or {}
    box = {str(s.get("team", {}).get("id")): s for s in (summary.get("boxscore") or {}).get("teams", [])}
    competitors = {c["homeAway"]: c for c in event["competitions"][0]["competitors"]}
    row = {**fixture, "match_id": f"community-{fixture['id']}",
           "date": utc(fixture["date"]).date().isoformat(),
           "source": "worldcup26.ir", "is_simulated": False}
    for side in ("home", "away"):
        competitor = competitors[side]
        goals = statistic(competitor.get("score"))
        if goals is None or not goals.is_integer() or goals > 50:
            raise ValueError("Marcador final inválido")
        row[f"{side}_goals"] = int(goals)
        fields = stats_dict(competitor.get("statistics", []))
        for k, v in stats_dict(box.get(str(fixture[f"{side}_id"]), {}).get("statistics", [])).items():
            if v is not None:
                fields[k] = v
        cards = roster_cards((summary.get("rosters") or []), fixture[f"{side}_id"])
        for k, v in cards.items():
            if fields.get(k) is None:
                fields[k] = v
        for original, target in STAT_MAP.items():
            value = fields.get(original)
            if value is not None and (value > (100 if target == "possession" else 200)
                                      or (target not in {"possession", "xg"} and not float(value).is_integer())):
                value = None
            row[f"{side}_{target}"] = value
        row[f"{side}_cards_source"] = "player_totals" if cards else "team_statistics"
    row["source_synced_at"] = (summary.get("meta") or {}).get("lastSyncedAt")
    return row


@dataclass
class CommunityFootball:
    session: object = field(default_factory=requests.Session, repr=False)
    cache: dict = field(default_factory=dict, repr=False)
    min_interval: float = 0.4
    last_call: float = 0.0
    _lock: object = field(default_factory=RLock, repr=False)

    def get(self, path: str, params: dict | None = None, ttl: int = 900) -> dict:
        with self._lock:
            return self._get(path, params, ttl)

    def _get(self, path: str, params: dict | None = None, ttl: int = 900) -> dict:
        if not re.fullmatch(r"(?:leagues|[a-z0-9_.-]+/(?:clubs|fixtures|summary))", path):
            raise ValueError("Ruta no admitida")
        params = params or {}
        key = (path, tuple(sorted(params.items())))
        cached = self.cache.get(key)
        if cached and time.monotonic() - cached[0] < ttl:
            return cached[1]
        time.sleep(max(0.0, self.min_interval - (time.monotonic() - self.last_call)))
        try:
            self.last_call = time.monotonic()
            response = self.session.get(f"{BASE_URL}/{path}", params=params, timeout=(8, 25),
                                        allow_redirects=False, headers={"Accept": "application/json"})
            if response.status_code == 429:
                raise ProviderError("La fuente gratuita alcanzó su límite. Espera unos minutos antes de actualizar.")
            if response.status_code >= 400 or response.status_code < 200 or response.status_code >= 300:
                raise ProviderError("La fuente comunitaria no está disponible para esta consulta. Inténtalo más tarde.")
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError):
            raise ProviderError("No se pudo conectar con la fuente comunitaria. Revisa tu conexión o prueba API-Football.") from None
        if not isinstance(data, dict) or data.get("error"):
            raise ProviderError("Respuesta incompleta de la fuente comunitaria")
        self.cache[key] = (time.monotonic(), data)
        LOGGER.info("Community API: %s", path)
        return data

    def search(self, query: str, league: str = "esp.1") -> list[dict]:
        if league not in LEAGUES.values():
            raise ValueError("Selecciona una liga disponible")
        text = search_text(query)
        text = {"barca": "barcelona", "man city": "manchester city", "man united": "manchester united",
                "atletico madrid": "atletico", "psg": "paris"}.get(text, text)
        if not 2 <= len(text) <= 80:
            raise ValueError("Escribe entre 2 y 80 caracteres")
        data = self.get(f"{league}/clubs", ttl=21600)
        if not isinstance(data.get("clubs"), list):
            raise ProviderError("La fuente no devolvió un catálogo de equipos")
        return [{"id": int(t["id"]), "name": t.get("displayName") or t["name"], "league": league,
                 "country": data.get("league", {}).get("country", ""), "logo": t.get("logo", "")}
                for t in data["clubs"] if text in search_text(t.get("displayName") or t.get("name", ""))]

    def fixtures(self, league: str, start, end) -> tuple[list[dict], dict]:
        records, seen, first = [], set(), None
        page = 1
        while True:
            payload = self.get(f"{league}/fixtures", {"from": utc(start).strftime("%Y%m%d"),
                "to": utc(end).strftime("%Y%m%d"), "status": "all", "limit": 200, "page": page})
            if not isinstance(payload.get("events"), list):
                raise ProviderError("Calendario sin lista de partidos")
            if first is None:
                first = payload.get("league", {})
            pages = int(payload.get("pageCount", 1))
            if pages > 15:
                raise ProviderError("Calendario demasiado amplio; no se mostrará una selección parcial como completa.")
            for event in payload["events"]:
                ident = str(event.get("id"))
                if ident not in seen:
                    records.append(event)
                    seen.add(ident)
            if page >= pages:
                break
            if not payload["events"]:
                raise ProviderError("El calendario contiene una página vacía inesperada")
            page += 1
        return records, first or {}

    def report(self, team_id: int, count: int = 5, league: str = "esp.1", now=None, progress=None) -> dict:
        if count not in {5, 10} or league not in LEAGUES.values() or int(team_id) <= 0:
            raise ValueError("Equipo, ventana o liga inválidos")
        now = utc(now if now is not None else datetime.now(timezone.utc))
        warnings, coverage, batches, latest = [], [], {}, []
        # Next fixture includes available national cups and European competitions.
        catalogue = self.get("leagues", ttl=21600).get("leagues", [])
        available = {x["slug"] for x in catalogue if x.get("coverage", {}).get("hasData")}
        extra = CUPS.get(league, []) + ["uefa.champions", "uefa.europa", "uefa.europa.conf",
                                      "eur.uefa.champions", "eur.uefa.europa", "eur.uefa.europa.conf"]
        slugs = [league] + [s for s in extra if s in available]
        for i, slug in enumerate(slugs):
            try:
                events, meta = self.fixtures(slug, now-pd.Timedelta(days=180), now+pd.Timedelta(days=90))
                batches[slug] = (events, meta)
                coverage.append(meta.get("name", slug))
                if meta.get("lastSyncedAt"):
                    latest.append(meta["lastSyncedAt"])
            except ProviderError:
                if slug == league:
                    raise
                warnings.append(f"No se pudo consultar el calendario de {slug}.")
            if progress:
                progress(0.35 * (i+1)/len(slugs))
        candidates = []
        for slug, (events, meta) in batches.items():
            for event in events:
                try:
                    f = parse_event(event, slug, meta.get("name", slug))
                    if (team_id in (f["home_id"], f["away_id"]) and f["status"] == "STATUS_SCHEDULED"
                            and now < utc(f["date"]) <= now+pd.Timedelta(days=90)):
                        candidates.append(f)
                except (KeyError, ValueError, TypeError, IndexError):
                    continue
        fixture = min(candidates, key=lambda f: utc(f["date"])) if candidates else None
        result = {"fixture": fixture, "history": pd.DataFrame(), "players": pd.DataFrame(columns=PLAYER_COLUMNS), "warnings": warnings,
                  "source": "worldcup26.ir · API comunitaria", "source_url": "https://worldcup26.ir/football-api",
                  "fetched_at": datetime.now(timezone.utc).isoformat(), "count": count,
                  "coverage": coverage, "source_sync": max(latest) if latest else None,
                  "scope": "Próximo registrado en las competiciones consultadas. Puede haber otros partidos fuera de esta cobertura.",
                  "history_scope": "Liga seleccionada y copas disponibles · últimos 180 días"}
        if fixture is None:
            warnings.append("No hay un próximo partido programado en los siguientes 90 días dentro de la cobertura consultada.")
            return result
        wanted, completed = {fixture["home_id"], fixture["away_id"]}, {}
        for slug, (events, meta) in batches.items():
            for event in events:
                try:
                    row = completed_event(event, slug, meta.get("name", slug))
                    if (wanted.intersection({row["home_id"], row["away_id"]})
                            and now-pd.Timedelta(days=180) <= utc(event["date"]) < now.normalize()):
                        completed[row["id"]] = (event, slug, meta.get("name", slug), row)
                except (KeyError, ValueError, TypeError, IndexError):
                    continue
        chosen = set()
        for ident in wanted:
            matching = [v for v in completed.values() if ident in (v[3]["home_id"], v[3]["away_id"])]
            chosen.update(v[3]["id"] for v in sorted(matching, key=lambda v: utc(v[0]["date"]), reverse=True)[:count])
        canonical_names = {fixture[f"{side}_id"]: fixture[f"{side}_team"] for side in ("home", "away")}
        rows, player_records, missing_summary = [], [], 0
        for i, ident in enumerate(sorted(chosen)):
            event, slug, name, row = completed[ident]
            try:
                payload = self.get(f"{slug}/summary", {"event": ident}, ttl=21600)
                if str((payload.get("header") or {}).get("id")) != str(ident):
                    raise ValueError("Resumen de otro partido")
                row = completed_event(event, slug, name, payload)
                player_records.extend(r for r in player_rows(payload, parse_event(event, slug, name)) if r["team_id"] in wanted)
            except (ProviderError, ValueError, KeyError, TypeError):
                missing_summary += 1
            for side in ("home", "away"):
                row[f"{side}_team"] = canonical_names.get(row[f"{side}_id"], row[f"{side}_team"])
            rows.append(row)
            if progress:
                progress(0.35 + 0.65*(i+1)/len(chosen))
        if missing_summary:
            warnings.append(f"{missing_summary} resúmenes ampliados no disponibles. Se conservan los datos recibidos; los campos ausentes quedan vacíos.")
        result["history"] = pd.DataFrame(rows)
        result["players"] = pd.DataFrame(player_records, columns=PLAYER_COLUMNS)
        return result
