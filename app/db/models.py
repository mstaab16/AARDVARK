from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, JSON, LargeBinary, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from db.database import Base

class Experiment(Base):
    __tablename__ = "experiment"

    experiment_id = Column(Integer, primary_key=True)
    motors = Column(JSONB)
    data_filepath = Column(String, default="")
    active = Column(Boolean, default=True)
    data_names_to_learn = Column(JSONB) # {"data_name_0": False, "data_name_1": True, ...}


    measurements = relationship("Measurement", back_populates="experiment")
    data = relationship("Data", back_populates="experiment")
    decisions = relationship("Decision", back_populates="experiment")
    reports = relationship("Report", back_populates="experiment")

class Measurement(Base):
    __tablename__ = "measurement"

    measurement_id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiment.experiment_id"))
    decision_id = Column(Integer, ForeignKey("decision.decision_id"))
    positions = Column(JSONB)
    measured = Column(Boolean, default=False)
    measurement_time = Column(String, default="")
    ai_cycle = Column(Integer, default=None)
    thubmnail = Column(LargeBinary)

    experiment = relationship("Experiment", back_populates="measurements")
    decision = relationship("Decision", back_populates="measurements")
    data = relationship("Data", back_populates="measurement")

class Data(Base):
    __tablename__ = "data"

    data_id = Column(Integer, primary_key=True)
    measurement_id = Column(Integer, ForeignKey("measurement.measurement_id"))
    experiment_id = Column(Integer, ForeignKey("experiment.experiment_id"))
    message = Column(String)
    fieldname = Column(String)
    data_cycle = Column(Integer)
    data = Column(String)
    data_info = Column(JSONB)

    measurement = relationship("Measurement", back_populates="data")
    experiment = relationship("Experiment", back_populates="data")

class Decision(Base):
    __tablename__ = "decision"

    decision_id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiment.experiment_id"))
    method = Column(JSONB)

    experiment = relationship("Experiment", back_populates="decisions")
    measurements = relationship("Measurement", back_populates="decision")

class Report(Base):
    __tablename__ = "report"

    report_id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiment.experiment_id"))
    name = Column(String)
    args = Column(JSONB)
    description = Column(String)
    data = Column(JSONB)
    time = Column(DateTime(timezone=True), server_default=func.now())

    experiment = relationship("Experiment", back_populates="reports")