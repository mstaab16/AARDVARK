from typing import Union

from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from maestro_api.maestro_app import maestro_app
from dash_app.app import app as dash_app
# from aardvark_api.aardvark_app import aardvark_app

from db.base import Base
from db.database import engine

app = FastAPI()

def create_tables():
    Base.metadata.create_all(bind=engine)

def drop_tables():
    Base.metadata.drop_all(bind=engine)

@app.on_event("startup")
async def startup_event():
    print("Starting up")
    create_tables()
    print("Tables created")

@app.get("/")
def read_root():
    return {"Hello": "aardvark"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

app.mount("/maestro", maestro_app)
app.mount("/dashboard", WSGIMiddleware(dash_app.server))
# app.mount("/aardvark", maestro_app)