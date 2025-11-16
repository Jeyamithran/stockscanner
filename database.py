from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base

from config import settings

Base = declarative_base()

engine = None
SessionLocal = None

if settings.database_url:
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        future=True,
        echo=settings.sql_echo,
    )
    SessionLocal = scoped_session(
        sessionmaker(
            bind=engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            future=True,
        )
    )


def init_db() -> None:
    if engine:
        Base.metadata.create_all(bind=engine)
