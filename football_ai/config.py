from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FIA_", env_file=".env", extra="ignore")
    database_url: SecretStr = SecretStr("sqlite:///data/football.db")
    artifact_dir: Path = Path("artifacts")
    seed: int = Field(default=42, ge=0)
    log_level: str = "INFO"
    public_mode: bool = False
    api_football_key: SecretStr = SecretStr("")


def configure_logging(level: str = "INFO") -> None:
    Path("logs").mkdir(exist_ok=True)
    handlers = [logging.StreamHandler(), RotatingFileHandler(
        "logs/football_ai.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")]
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=handlers, force=True)
