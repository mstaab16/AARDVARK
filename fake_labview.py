import requests
import numpy as np
import base64
from app.maestro_api.maestro_messages import *
import fake_crystals
import time
import zmq

import numpy as np
import uuid
import matplotlib.pyplot as plt

class FakeLV:
    def __init__(self):
        self.aardvark_url = "tcp://einstein.dhcp.lbl.gov:5550"
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REQ)
        self.socket.connect(self.aardvark_url)

        self.current_ai_cycle = 0
        self.current_data_cycle = 0
        self.max_ai_cycle = 500
        self.measurement_delay = 0.00001
        self.num_energies = 256
        self.num_angles = 256
        self.dead_time = 0
        self.start_time = time.perf_counter()
        # self.crystal_bounds = [[-1,1], [-1,1]]
        # self.min_steps = [0.01, 0.01]
        # self.fake_crystal = fake_crystals.FakeVoronoiCrystal(num_crystallites=6, num_angles = self.num_angles, num_energies = self.num_energies,
        #                                        bounds=self.crystal_bounds, min_steps=self.min_steps)
        self.fake_crystal = fake_crystals.FakeGrapheneCrystal()

        self.waiting_times = []
        start = time.perf_counter()
        self.startup()
        self.operation_loop()
        total_time = (time.perf_counter()-start)
        print(f"Total time: {total_time/60:.02f}min")
        average_time_per_cycle = total_time/self.max_ai_cycle
        print(f"Average time per cycle: {average_time_per_cycle:.02f}s | {1/average_time_per_cycle:.02f} Hz")

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
                AIModeparm(device_name="motors::X", enabled_=True, low=self.fake_crystal.xmin, high=self.fake_crystal.xmax, min_step=self.fake_crystal.xdelta),
                AIModeparm(device_name="motors::Y", enabled_=True, low=self.fake_crystal.ymin, high=self.fake_crystal.ymax, min_step=self.fake_crystal.ydelta),
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
        response = self.send_to_aardvark(msg)
        print(f"LV: Recieved response:\n{response}")

    # def post_to_aardvark(self, endpoint, msg):
    #     response = requests.post(self.aardvark_url + endpoint, headers={'accept': 'application/json', 'Content-Type': 'application/json'}, data=msg.model_dump_json())
    #     return response.json()
    def send_to_aardvark(self, msg):
        start = time.perf_counter()
        self.socket.send_string(msg.model_dump_json(by_alias=True))
        response = self.socket.recv_json()
        print(f"LV: Round trip req-rep time: {(time.perf_counter()-start)*1000:.02f}ms")
        return response

    def operation_loop(self):
        while True:
            print("LV: " + "-"*20 + f" {self.current_ai_cycle} " + "-"*20)
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
        response = self.send_to_aardvark(message)

    def request_position(self):
        position_request = MaestroLVPositionRequest(current_AI_cycle=self.current_ai_cycle)
        start = time.perf_counter()
        response = self.send_to_aardvark(position_request)
        position_request_time = (time.perf_counter()-start)*1000
        self.waiting_times.append(position_request_time)
        print(f"LV: Position request took {position_request_time:.02f}ms round trip.")
        self.position = MaestroLVPositionResponse(**response)

    def send_data(self):
        print("Taking Data...")
        time.sleep(self.measurement_delay)
        data = self.generate_data()
        filename = "/mnt/MAESTROdata/Eli-CAD-3/test/"
        filename += str(uuid.uuid4())
        start = time.perf_counter_ns()
        data.tofile(filename)
        print(f"LV: Data took {(time.perf_counter_ns()-start)/1e6:.02f}ms to write to file.")

        filename_to_send = 'X:\\' + '\\'.join(filename.split("/")[3:])


        data_message = DataMessage(
                    current_data_cycle=self.current_data_cycle,
                    current_AI_cycle=self.current_ai_cycle,
                    method="sending newdata",
                    )
        message = MaestroLVDataMessage(
                message=data_message,
                fits_descriptors = [
                    FitsDescriptor(
                        **{
                            #TODO: The data may be as i32 not f64 or u8 according to the real fits descriptor on labview
                            "Data": filename_to_send,
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
        response = self.send_to_aardvark(message)
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
        start = time.perf_counter()
        x, y = [motor_position.value for motor_position in self.position.positions]
        measured_x, measured_y, measured_data = self.fake_crystal.measure(x,y)
        result = (measured_data * 1e6).astype(np.int32, order='F')
        print(f"LV: Generating data took {(time.perf_counter()-start)*1000:.02f}ms")
        return result


f = FakeLV()