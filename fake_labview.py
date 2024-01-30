import requests
import numpy as np
import base64
from app.maestro_api.maestro_messages import *
import time

class FakeLV:
    def __init__(self, translator_addr = 'http://einstein.dhcp.lbl.gov/maestro'):
        self.aardvark_url = "http://127.0.0.1/maestro"

        self.current_ai_cycle = 0
        self.current_data_cycle = 0
        self.max_ai_cycle = 121
        self.startup()
        self.operation_loop()

    # def get_translator_response(self):
    #     msg: MaestroLVResponse = self.socket.recv_json()
    #     if msg.get("status") == MaestroLVResponseError().status:
    #         print("Recieved an Error from the Translator...\nShutting down...")
    #         exit()
    #     return msg


    def startup(self):
        # msg = FakeLVStartupMessage(
        #     max_count=5,
        #     motor_defs=[
        #     REManagerMotorDefMessage(name='x', bounds=[0.,1.], delta=0.05),
        #     REManagerMotorDefMessage(name='y', bounds=[0.,1.], delta=0.1)
        #     ]
        # )
        msg = MaestroLVStartupMessage(
            AI_Controller = 'Aardvark',
            AIModeparms=[
                AIModeparm(device_name="motors::X", enabled_=True, high=10, low=-10, min_step=1.99),
                AIModeparm(device_name="motors::Y", enabled_=True, high=10, low=-10, min_step=1.99)
            ],
            # This is the number of AI cycles to go through
            max_count=self.max_ai_cycle,
            method="initialize",
            scan_descriptors=ScanDescriptors(
                Scan_Descriptor=[
                    ScanDescriptorItem(num_positions=89,
                                        Offsets=[0,0],
                                        Range=[RangeItem(End=0,N=1,Start=0)],
                                        Scan_Type="Computed",
                                        Tab_Posns_=[[0.0,0.1]],
                                        device_descriptor=DeviceDescriptor(
                                            NEXUS_Path_Class_rel_to_entry="sample",
                                            device_name="None",
                                            subdevices=[
                                                Subdevice(hi=0,lo=0,name="null",parms=[],units="")
                                            ]))
                ],
                Scan_Devices_in_Parallel=False,
                # This is how many frames per cycle *I think*
                total_num_cycles=1,
            )
        )
        print("LV: Sending startup message to translator...")
        response = self.post_to_aardvark("/init/", msg)    
        print(f"LV: Recieved response:\n{response}")

    def post_to_aardvark(self, endpoint, msg):
        response = requests.post(self.aardvark_url + endpoint, headers={'accept': 'application/json', 'Content-Type': 'application/json'}, data=msg.model_dump_json())
        return response.json()

    def operation_loop(self):
        while True:
            print(f"LV: Sending position request {self.current_ai_cycle}")
            self.request_position()
            print("LV: Sending data for that position...")
            self.send_data()
            self.current_ai_cycle += 1
            self.current_data_cycle = 0
            if self.current_ai_cycle >= self.max_ai_cycle:
                self.send_translator_close()
                break

    def send_translator_close(self):
        message = MaestroLVCloseMessage()
        response = self.post_to_aardvark("/close/", message)

    def request_position(self):
        position_request = MaestroLVPositionRequest(current_AI_cycle=self.current_ai_cycle)
        response = self.post_to_aardvark("/request_next_position/", position_request)
        self.position = MaestroLVPositionResponse(**response)

    def send_data(self):
        # time.sleep(.1)
        data = self.generate_data()
        data_bytes = data.tobytes(order='F')
        data_message = DataMessage(
                    current_data_cycle=self.current_data_cycle,
                    current_AI_cycle=self.current_ai_cycle,
                    )
        message = MaestroLVDataMessage(
                message=data_message,
                fits_descriptors = [
                    FitsDescriptor(
                        **{
                        # The descriptor here says u8 but on the LabView it says f64 (Double). Not sure why
                        "Data": base64.encodebytes(np.array([time.time()]).astype(np.float64).tobytes()),
                        "axisnames": [],
                        "data dimensions": "string",
                        "dataunitname (if not string)": "sec",
                        "dimensions": [0],
                        "fieldname": "time",
                        "numeric type (if not string)": "B:u8",
                        "scaledelta": [],
                        "scaleoffset": [],
                        "string length": 10,
                        "unitnames": []
                        }
                    ),
                    FitsDescriptor(
                        **{
                        "Data": base64.encodebytes(b"\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000"),
                        "axisnames": [],
                        "data dimensions": "string",
                        "dataunitname (if not string)": "",
                        "dimensions": [0],
                        "fieldname": "null",
                        "numeric type (if not string)": "B:u8",
                        "scaledelta": [],
                        "scaleoffset": [],
                        "string length": 10,
                        "unitnames": []
                        }
                    ),
                    FitsDescriptor(
                        **{
                            #TODO: The data may be as i32 not f64 or u8 according to the real fits descriptor on labview
                            "Data": base64.encodebytes(data_bytes),
                            "axisnames": ["Energy","Angle"],
                            "data dimensions": "string",
                            "dataunitname (if not string)": "arb",
                            "dimensions": [*data.shape],
                            "fieldname": "Fixed_Spectra0",
                            "numeric type (if not string)": "B:u8",
                            "scaledelta": [25,500],
                            "scaleoffset": [2.6499999999999999112,104],
                            "string length": 10,
                            "unitnames": ["eV","pixels"]
                        }
                    )
                ]
                )
        response = self.post_to_aardvark("/data/", message)
        self.current_data_cycle += 1
        print(f'LV: Recieved msg after sending data: {response}')
    
    def generate_data(self):
        print(self.position)
        fac = 1
        if self.position.positions[0].value < 0:
            fac = 0.1
        if self.position.positions[0].value > 0.2:
            fac = 2
        data_shape = (1024, 1024)
        return (fac*np.random.uniform(0,1e7, data_shape)).astype(np.int32)


f = FakeLV()