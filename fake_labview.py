import zmq
import numpy as np
import base64
from messages import *
from maestro_messages import *
import time

class FakeLV:
    def __init__(self, translator_addr = 'tcp://einstein.dhcp.lbl.gov:5550'):
        self.translator_addr = translator_addr
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(translator_addr)
        self.current_ai_cycle = 0
        self.current_data_cycle = 0
        self.max_ai_cycle = 25
        self.startup()
        self.operation_loop()

    def get_translator_response(self):
        msg: MaestroLVResponse = self.socket.recv_json()
        if msg.get("status") == MaestroLVResponseError().status:
            print("Recieved an Error from the Translator...\nShutting down...")
            exit()
        return msg


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
                AIModeparm(device_name="motors::X", enabled_=True, high=10, low=-10, min_step=0.01),
                AIModeparm(device_name="motors::Y", enabled_=True, high=10, low=-10, min_step=0.05)
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
                                           ]
                                           )
                                       )
                ],
                Scan_Devices_in_Parallel=False,
                # This is how many frames per cycle *I think*
                total_num_cycles=1,
            )
        )
        print("LV: Sending startup message to translator...")
        self.socket.send_string(msg.model_dump_json())
        print("LV: Sent.")
        # resp = self.socket.recv()
        resp = self.get_translator_response()
        print(f"LV: Recieved response: {resp}")

    def operation_loop(self):
        while True:
            print("LV: Sending position request")
            self.send_pos_req()
            print("LV: Waiting for next position...")
            self.recieve_next_position()
            print("LV: Sending data for that position...")
            # time.sleep(0.5)
            self.send_data()
            self.current_ai_cycle += 1
            if self.current_ai_cycle >= self.max_ai_cycle:
                self.send_translator_close()
                break

    def send_translator_close(self):
        self.socket.send_string(MaestroLVCloseMessage().model_dump_json())

    def send_pos_req(self):
        self.socket.send_string(MaestroLVPositionRequest(current_AI_cycle=self.current_ai_cycle).model_dump_json())
    
    def recieve_next_position(self):
        # msg = self.socket.recv_json()
        msg = self.get_translator_response()
        print(msg)
        position = MaestroLVPositionResponse(**msg)
        self.position = position
        print(f'LV: Now at position: {self.position}')

    def send_data(self):
        data = self.generate_data()
        data_bytes = data.tobytes(order='F')
        data_message=DataMessage(
                    current_data_cycle=self.current_data_cycle,
                    current_AI_cycle=self.current_ai_cycle,
                    )
        self.socket.send_string(
            MaestroLVDataMessage(
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
                ).model_dump_json()
                )
        self.current_data_cycle += 1
        msg = self.get_translator_response()
        print(f'LV: Recieved msg after sending data: {msg}')
    
    def generate_data(self):
        print(self.position)
        fac = 1
        if self.position.positions[0].value < 0:
            fac = 0.1
        if self.position.positions[0].value > 0.2:
            fac = 2
            
        return (fac*np.random.uniform(0,1e7, (512, 512))).astype(np.int32)


f = FakeLV()