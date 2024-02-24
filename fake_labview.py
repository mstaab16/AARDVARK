import requests
import numpy as np
import base64
from app.maestro_api.maestro_messages import *
from fake_crystals import FakeVoronoiCrystal
import time

import matplotlib.pyplot as plt

class FakeLV:
    def __init__(self):
        self.aardvark_url = "http://einstein.dhcp.lbl.gov/maestro"

        self.current_ai_cycle = 0
        self.current_data_cycle = 0
        self.max_ai_cycle = 200
        self.measurement_delay = 0.00001
        self.num_energies = 1024
        self.num_angles = 1024
        self.dead_time = 0
        self.start_time = time.perf_counter()
        self.crystal_bounds = [[-1,1], [-1,1]]
        self.min_steps = [0.01, 0.01]
        self.fake_crystal = FakeVoronoiCrystal(num_crystallites=4, num_angles = self.num_angles, num_energies = self.num_energies,
                                               bounds=self.crystal_bounds, min_steps=self.min_steps)
        self.waiting_times = []
        self.startup()
        self.operation_loop()

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
            # AIModeparms=[
            #     AIModeparm(device_name="motors::X", enabled_=True, low=0, high=1, min_step=0.01),
            #     AIModeparm(device_name="motors::Y", enabled_=True, low=0, high=1, min_step=0.01)
            # ],
            AIModeparms=[
                AIModeparm(device_name="motors::X", enabled_=True, low=self.crystal_bounds[0][0], high=self.crystal_bounds[0][1], min_step=self.min_steps[0]),
                AIModeparm(device_name="motors::Y", enabled_=True, low=self.crystal_bounds[1][0], high=self.crystal_bounds[1][1], min_step=self.min_steps[1])
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
            ellapsed_time = time.perf_counter() - self.start_time
            dead_fraction = self.dead_time / ellapsed_time
            print(f"Experiment time: {ellapsed_time/60:0.2f}min\tDead time: {self.dead_time/60:0.2f}min | {dead_fraction:0.2%}")
            start = time.perf_counter()
            self.request_position()
            time_waiting_for_position = time.perf_counter() - start
            self.dead_time += time_waiting_for_position
            self.send_data()
            self.current_ai_cycle += 1
            self.current_data_cycle = 0
            plt.clf()
            plt.plot(self.waiting_times)
            plt.savefig("waiting_times.png")
            if self.current_ai_cycle >= self.max_ai_cycle:
                self.send_translator_close()
                plt.clf()
                plt.hist(self.waiting_times, bins=50)
                plt.yscale('log')
                plt.title(f"Total dead time: {self.dead_time/60:.02f}min")
                plt.savefig("waiting_times_hist.png")
                break

    def send_translator_close(self):
        message = MaestroLVCloseMessage()
        response = self.post_to_aardvark("/close/", message)

    def request_position(self):
        position_request = MaestroLVPositionRequest(current_AI_cycle=self.current_ai_cycle)
        start = time.perf_counter()
        response = self.post_to_aardvark("/request_next_position/", position_request)
        position_request_time = (time.perf_counter()-start)*1000
        self.waiting_times.append(position_request_time)
        print(f"LV: Position request took {position_request_time:.02f}ms round trip.")
        self.position = MaestroLVPositionResponse(**response)

    def send_data(self):
        print("Taking Data...")
        time.sleep(self.measurement_delay)
        data = self.generate_data()
        data_bytes = data.tobytes(order='F')
        # print(data_bytes)
        data_message = DataMessage(
                    current_data_cycle=self.current_data_cycle,
                    current_AI_cycle=self.current_ai_cycle,
                    method="fake_labview newdata",
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
                            "fieldname": "Fixed_Spectra5",
                            "numeric type (if not string)": "B:u8",
                            "scaledelta": [25,500],
                            "scaleoffset": [2.6499999999999999112,104],
                            "string length": 10,
                            "unitnames": ["eV","pixels"]
                        }
                    )
                ]
                )
        print("Sending Data...")
        response = self.post_to_aardvark("/data/", message)
        self.current_data_cycle += 1
        print(f'LV: Recieved msg after sending data: {response}')
    
    def generate_data(self):
        # print(self.position)
        # fac = 1
        # if self.position.positions[0].value < 0:
        #     fac = 0.1
        # if self.position.positions[0].value > 0.2:
        #     fac = 2
        # data_shape = (128, 128)
        # return (fac*np.random.uniform(0,1e7, data_shape)).astype(np.int32)
        x, y = [motor_position.value for motor_position in self.position.positions]
        measured_x, measured_y, measured_data = self.fake_crystal.measure(x,y)
        return (measured_data * 1e6).astype(np.int32)


f = FakeLV()