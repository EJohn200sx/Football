"""Portable SQLAlchemy repository; atomic imports with conflict rejection."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json
import pandas as pd
from sqlalchemy import Boolean, Column, Date, DateTime, JSON, MetaData, String, Table, create_engine, select
from sqlalchemy.engine import Engine, make_url

metadata = MetaData()
matches = Table("matches", metadata,
    Column("match_id", String(32), primary_key=True), Column("date", Date, nullable=False, index=True),
    Column("competition", String(120), nullable=False, index=True), Column("season", String(30), nullable=False),
    Column("is_simulated", Boolean, nullable=False), Column("payload", JSON, nullable=False))
imports = Table("imports", metadata, Column("import_id", String(32), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False), Column("metadata", JSON, nullable=False))


def connect(url: str) -> Engine:
    parsed = make_url(url)
    if parsed.get_backend_name() not in {"sqlite", "postgresql"}:
        raise ValueError("Sólo SQLite o PostgreSQL")
    if parsed.get_backend_name() == "sqlite" and parsed.database not in [None, "", ":memory:"]:
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, pool_pre_ping=True)
    metadata.create_all(engine)
    return engine


def store_matches(engine: Engine, frame: pd.DataFrame, provenance: dict) -> dict:
    inserted = duplicate = 0
    # Entire batch succeeds or rolls back; never silently overwrites a result.
    with engine.begin() as conn:
        for item in json.loads(frame.to_json(orient="records")):
            ident = item["match_id"]
            existing = conn.execute(select(matches.c.payload).where(matches.c.match_id == ident)).scalar_one_or_none()
            if existing is not None:
                semantic = [k for k in item if k not in {"source", "kickoff_raw"}]
                if any(existing.get(k) != item.get(k) for k in semantic):
                    raise ValueError(f"Conflicto con partido existente {ident}; revisar fuente/corrección manual")
                duplicate += 1
                continue
            conn.execute(matches.insert().values(match_id=ident, date=pd.Timestamp(item["date"]).date(),
                competition=item["competition"], season=item["season"], is_simulated=item["is_simulated"], payload=item))
            inserted += 1
        result = {**provenance, "inserted": inserted, "already_present": duplicate}
        conn.execute(imports.insert().values(import_id=uuid4().hex,
                     created_at=datetime.now(timezone.utc), metadata=result))
    return result


def load_matches(engine: Engine, competition: str | None = None, simulated: bool | None = None) -> pd.DataFrame:
    query = select(matches.c.payload).order_by(matches.c.date, matches.c.match_id)
    if competition is not None:
        query = query.where(matches.c.competition == competition)
    if simulated is not None:
        query = query.where(matches.c.is_simulated == simulated)
    with engine.connect() as conn:
        rows = conn.execute(query).scalars().all()
    return pd.DataFrame(rows)


def import_history(engine: Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.DataFrame(conn.execute(select(imports).order_by(imports.c.created_at.desc())).mappings().all())
