from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from db.config import settings

SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL
# "postgresql://postgres:arpes_aardvark@db-1:5432/aardvark_db"
# "postgresql://192.168.112.2/aardvark_db?user=postgres&password=arpes_aardvark"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, #connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()