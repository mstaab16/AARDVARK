from pydantic import BaseModel

class MotorConfigFake(BaseModel):
    name: str
    bounds: list[float]
    delta: float

class DataDefFake(BaseModel):
    name: str
    shape: list[int]

class StartupMessageFake(BaseModel):
    message_type: str = "startup"
    motors: list[MotorConfigFake]
    data_defs: list[DataDefFake]

class PositionRequestFake(BaseModel):
    name: str = 'pos_request'

class DataMessageFake(BaseModel):
    datasets: list[bytes]

class MotorPositionMessageFake(BaseModel):
    ordered_positions: list[float]

# Message classes for communications between
# the Real Maestro LV Client and the Translator
# class MaestroLVStartupMessage(BaseModel):
#     pass

# class MaestroLVPositionRequest(BaseModel):
#     pass

# class MaestroLVPositionResponse(BaseModel):
#     pass

# class MaestroLVDataMessage(BaseModel):
#     pass

# class MaestroLVShutdownMessage(BaseModel):
#     pass

# Message classes for communications between
# the RunEnegineManager and the Translator
class REManagerMotorDefMessage(BaseModel):
    # defn = {
            #     'theta': (LVSignal, 'theta', {'value': 0.1, 'bounds': (-15.0, 15.0)}),
            #     'phi': (LVSignal, 'phi', {'value': -3, 'bounds': (-15.0, 15.0)}),
            #     'img': (LVSignal, 'img', {'value': np.zeros((64,64)), 'write_access': False}),
            # }
    name: str
    # value: float
    bounds: list[float]
    delta: float

class REManagerDataMessage(BaseModel):
    names: list[str]
    datasets: list[bytes]
    shapes: list[list[int]]

class REManagerDataDefMessage(BaseModel):
    name: str
    shape: list[int]

class REManagerStartupMessage(BaseModel):
    motor_defs: list[REManagerMotorDefMessage]
    data_defs: list[REManagerDataDefMessage]
    max_count: int

class REManagerMotorPositionMessage(BaseModel):
    positions: list[float]
    names: list[str]


# Message classes for communications between
# the FakeLVClient and the Translator
class FakeLVStartupMessage(BaseModel):
    message: str = 'init'
    max_count: int
    motor_defs: list[REManagerMotorDefMessage]

class FakeLVPositionRequest(BaseModel):
    message: str = 'fake labview position request'

class FakeLVPositionResponse(BaseModel):
    positions: list[float]

class FakeLVDataMessage(REManagerDataMessage):
    pass

class FakeLVShutdownMessage(BaseModel):
    message: str = 'shutdown'
