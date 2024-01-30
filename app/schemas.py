from datetime import datetime
from pydantic import BaseModel


# class ExperimentBase(BaseModel):
#     motor: dict
#     data_filepath: str
#     active: bool

# class ExperimentCreate(ExperimentBase):
#     pass

# class Experiment(ExperimentBase):
#     experiment_id: int

#     class Config:
#         from_attributes = True

# class Experiment(ExperimentBase):
#     experiment_id: int

#     class Config:
#         from_attributes = True

# class MoveBase(BaseModel):
#     experiment_id: int
#     positions: dict

# class MoveCreate(MoveBase):
#     pass

# class Move(MoveBase):
#     move_id: int
#     positions: dict
#     measured: bool
#     measurement_time: datetime | None

#     class Config:
#         from_attributes = True

# class DecisionBase(BaseModel):
#     experiment_id: int
#     method: dict

# class DecisionCreate(DecisionBase):
#     pass

# class Decision(DecisionBase):
#     decision_id: int

#     class Config:
#         from_attributes = True

# class ReportBase(BaseModel):
#     experiment_id: int
#     name: str
#     args: dict
#     description: str
#     data: dict
#     time: datetime | None

# class ReportCreate(ReportBase):
#     pass

# class Report(ReportBase):
#     report_id: int
#     experiment_id: int
#     name: str
#     args: dict
#     description: str
#     data: dict
#     time: datetime | None

#     class Config:
#         from_attributes = True
