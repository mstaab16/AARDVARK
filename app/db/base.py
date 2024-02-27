from db.database import Base, engine
from db.models import Experiment, Measurement, Data, Decision, Report

Base.metadata.create_all(bind=engine)