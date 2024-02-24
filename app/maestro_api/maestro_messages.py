from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

########################################################################################
#                                   FOR STARTUP MESSAGE
########################################################################################
class AIModeparm(BaseModel):
    device_name: str = Field(..., alias='device::name')
    enabled_: bool = Field(..., alias='enabled?')
    high: float
    low: float
    min_step: float = Field(..., alias='min. step')

    class Config:
        populate_by_name = True


class RangeItem(BaseModel):
    End: int
    N: int
    Start: int

    class Config:
        populate_by_name = True


class Subdevice(BaseModel):
    hi: float
    lo: float
    name: str
    parms: List
    units: str

    class Config:
        populate_by_name = True


class DeviceDescriptor(BaseModel):
    NEXUS_Path_Class_rel_to_entry: str = Field(
        ..., alias='NEXUS Path:Class (rel. to /entry)'
    )
    device_name: str = Field(..., alias='device name')
    subdevices: List[Subdevice]

    class Config:
        populate_by_name = True


class ScanDescriptorItem(BaseModel):
    num_positions: int = Field(..., alias='# positions')
    Offsets: List[int]
    Range: List[RangeItem]
    Scan_Type: str = Field(..., alias='Scan Type')
    Tab_Posns_: List[List[float]] = Field(..., alias='Tab. Posns.')
    device_descriptor: DeviceDescriptor = Field(..., alias='device descriptor')

    class Config:
        populate_by_name = True


class ScanDescriptors(BaseModel):
    Scan_Descriptor: List[ScanDescriptorItem] = Field(..., alias='Scan Descriptor')
    Scan_Devices_in_Parallel: bool = Field(..., alias='Scan Devices in Parallel?')
    total_num_cycles: int = Field(..., alias='total # cycles')

    class Config:
        populate_by_name = True


class MaestroLVStartupMessage(BaseModel):
    AI_Controller: str = Field(..., alias='AI Controller')
    AIModeparms: List[AIModeparm]
    max_count: int = Field(..., alias='max #')
    method: str = "initialize"
    scan_descriptors: ScanDescriptors = Field(..., alias='scan descriptors')

    class Config:
        populate_by_name = True


########################################################################################
#                              FOR POSITION REQUEST MESSAGE
########################################################################################
class MaestroLVPositionRequest(BaseModel):
    current_AI_cycle: int = Field(..., alias='current AI cycle')
    method: str = "move"

    class Config:
        populate_by_name = True


########################################################################################
#                                      FOR DATA MESSAGE
########################################################################################
class FitsDescriptor(BaseModel):
    fieldname: str
    data_dimensions: str = Field(..., alias='data dimensions')
    string_length: int = Field(..., alias='string length')
    numeric_type_if_not_string_: str = Field(..., alias='numeric type (if not string)')
    dataunitname_if_not_string_: str = Field(..., alias='dataunitname (if not string)')
    dimensions: List[int]
    scaleoffset: List[float]
    scaledelta: List[float]
    unitnames: List[str]
    axisnames: List[str]
    Data: bytes
    
    class Config:
        populate_by_name = True
    
class DataMessage(BaseModel):
    current_data_cycle: int = Field(..., alias='current data cycle')
    current_AI_cycle: int = Field(..., alias='current AI cycle')
    method: str = 'sending newdata'

    class Config:
        populate_by_name = True

class MaestroLVDataMessage(BaseModel):
    message: DataMessage
    fits_descriptors: List[FitsDescriptor] = Field(..., alias='fits descriptors')

    class Config:
        populate_by_name = True

########################################################################################
#                                     FOR CLOSE MESSAGE
########################################################################################
class MaestroLVCloseMessage(BaseModel):
    current_data_cycle: int = Field(default=0, alias='current data cycle')
    current_ai_cycle: int = Field(default=0, alias='current AI cycle')
    method: str = "closing"

    class Config:
        populate_by_name = True

class MaestroLVAbortMessage(BaseModel):
    # current_data_cycle: int = Field(default=0, alias='current data cycle')
    # current_ai_cycle: int = Field(default=0, alias='current AI cycle')
    method: str = "abort"

    class Config:
        populate_by_name = True


########################################################################################
#                                ANY RESPONSE MESSAGE
########################################################################################
class MaestroLVResponse(BaseModel):
    status: str

    class Config:
        populate_by_name = True
    
class MaestroLVResponseOK(MaestroLVResponse):
    status:str = 'OK'

class MaestroLVResponseError(MaestroLVResponse):
    status:str = 'stop'

########################################################################################
#                                FOR POSITION RESPONSE MESSAGE
########################################################################################
class MotorPosition(BaseModel):
    axis_name: str = Field(..., alias='axis name')
    value: float

    class Config:
        populate_by_name = True

# from pydantic import RootModel

class MaestroLVPositionResponse(MaestroLVResponse):
    positions: List[MotorPosition]

    class Config:
        populate_by_name = True

