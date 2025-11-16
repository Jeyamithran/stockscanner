import os
from dataclasses import dataclass


@dataclass
class Settings:
    database_url: str = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_DSN") or os.environ.get("POSTGRES_URL")
    pplx_api_key: str = os.environ.get("PPLX_API_KEY")
    pplx_model_technical: str = os.environ.get("PPLX_MODEL_TECHNICAL", os.environ.get("PPLX_DEFAULT_MODEL", "sonar-reasoning-pro"))
    pplx_model_news_tech: str = os.environ.get("PPLX_MODEL_NEWS_TECH", os.environ.get("PPLX_NEWS_MODEL", "sonar-pro"))
    sql_echo: bool = os.environ.get("SQL_ECHO", "false").strip().lower() in {"1", "true", "yes", "on"}


settings = Settings()
