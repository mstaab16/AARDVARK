import zmq
import numpy as np
from motors import LVSignal
from pydantic import BaseModel, ValidationError
from messages import *




# print(StartupMessageFake(motors=[MotorConfigFake(name='x', bounds=[-1.0,1.0], delta=0.1), MotorConfigFake(name='y', bounds=[0.0,0.5], delta=0.01)]).model_dump_json())

# print(DataMessageFake(datasets=[np.arange(16**2).reshape((16,16))]))
# arr = str((16*np.arange(2**2)).reshape((2,2)).astype(np.float64).tobytes().hex())
# newarr = ' '.join([arr[i*4:i*4+4] for i in range(len(arr)//4 - 1)])
# for i in range(len(arr)//16):
#     print(arr[i*16:i*16 +16])

#This works!
# arr = 2 * np.ones((2,2)).astype(np.float64)
# msg = arr.tobytes()
# data = np.frombuffer(msg, dtype=np.float64).reshape(arr.shape)
# print(arr)
# print(msg.hex())
# print(data)

# import time
# from time import perf_counter_ns
# context = zmq.Context()
# socket = context.socket(zmq.REP)
# socket.bind('tcp://131.243.156.33:5555')
# while True:
#     print(f"Waiting for message...")
#     msg = socket.recv()
#     print("Message recieved")
#     start = perf_counter_ns()
#     rows = 1024
#     cols = 1024
#     # data = msg[8:]
#     # data = np.frombuffer(msg, dtype=np.dtype(np.float64).newbyteorder('>')).reshape((rows,cols))
#     data = np.frombuffer(msg, dtype=np.float64).reshape((rows,cols))
#     update = f"Message with {data.shape=} took {(perf_counter_ns()-start)/1e6}ms to convert in python."
#     print(data)
#     print(update)
#     # print(f'{rows=}')
#     # print(f'{cols=}')
#     # print(f'{data=}')
#     time.sleep(0.1)
#     socket.send_string(update)


class Translator:
    def __init__(self, translator_addr = 'tcp://127.0.0.1:5551', re_addr = 'tcp://127.0.0.1:5550', use_real_lv_messages=True):
        # setup zmq communication
        self.addr = translator_addr
        self.re_addr = re_addr
        self.num_data_collected = 0
        self.StartupMessage = MaestroLVStartupMessage if use_real_lv_messages else FakeLVStartupMessage
        self.PositionRequest = MaestroLVPositionRequest if use_real_lv_messages else FakeLVPositionRequest
        self.PositionResponse = MaestroLVPositionResponse if use_real_lv_messages else FakeLVPositionResponse
        self.DataMessage = MaestroLVDataMessage if use_real_lv_messages else FakeLVDataMessage
        self.ShutDownMessage = MaestroLVShutdownMessage if use_real_lv_messages else FakeLVShutdownMessage
        self.startup()
        self.operation_loop()
        
    
    def startup(self):
        self.context = zmq.Context()
        self.lv_socket = self.context.socket(zmq.REP)
        self.lv_socket.bind(self.addr)
        self.re_socket = self.context.socket(zmq.REQ)
        self.re_socket.connect(self.re_addr)

        print("Translator: Waiting for message from lv_socket...")
        start_msg = FakeLVStartupMessage(**self.lv_socket.recv_json())
        self.max_count = start_msg.max_count
        # print(type(message))
        # print(self.StartupMessage(**message))
        self.lv_socket.send(b'ok')
        first_positions = [np.mean(motor.bounds) for motor in start_msg.motor_defs]
        first_pos_req = self.lv_socket.recv_json()
        print(f'T: First positions: {first_positions}')
        print(f'T: First position request: {first_pos_req}')
        first_position_response = self.PositionResponse(positions=first_positions)
        print(f"First position response: {first_position_response}")
        self.lv_socket.send_string(first_position_response.model_dump_json())
        first_data_msg = self.DataMessage(**self.lv_socket.recv_json())
        self.lv_socket.send(b'Data Recieved OK.')

        re_motor_defs = [
            REManagerMotorDefMessage(name=motor.name, bounds=motor.bounds, delta=motor.delta)
            for motor in start_msg.motor_defs
        ]
        re_data_defs = [
            REManagerDataDefMessage(name=name, shape=shape)
            for name, shape in zip(first_data_msg.names, first_data_msg.shapes)
        ]
        re_start_msg = REManagerStartupMessage(motor_defs=re_motor_defs, data_defs=re_data_defs, max_count=self.max_count)
        self.re_socket.send_string(re_start_msg.model_dump_json())
        print(self.re_socket.recv())
        self.operation_loop()

    def operation_loop(self):
        while True:
            print("-"*20,"\n","T: Waiting for LV Pos Req\n","-"*20)
            self.get_lv_pos_req()
            print("-"*20,"\n","T: Getting pos from RunEngineManager\n","-"*20)
            self.get_pos_from_RE_manager()
            print("-"*20,"\n","T: Sending Next Position to LV\n","-"*20)
            self.send_next_pos_to_lv()
            print("-"*20,"\n","T: Waiting for LV Dataset\n","-"*20)
            self.get_lv_dataset()
            print("-"*20,"\n",f"T: Sending data #{self.num_data_collected} to RunEngineManager\n","-"*20)
            self.send_data_to_RE_manager()
            print("-"*20,"\n","T: Telling Labview OK\n","-"*20)
            self.send_lv_OK()

    def get_lv_pos_req(self):
        try:
            msg = self.lv_socket.recv_json()
            pos_req = self.PositionRequest(**msg)
        except ValidationError as e:
            print(f"There was an error while validating the position request.\nThe message was:\n{msg}")
            raise ValidationError(e)
        self.position_request = pos_req

    def get_pos_from_RE_manager(self):
        self.re_socket.send_string(self.position_request.model_dump_json())
        try:
            msg = self.re_socket.recv_json()
            # print(msg)
            # raise Exception
            next_pos = REManagerMotorPositionMessage(**msg)
        except ValidationError as e:
            print(f"There was an error while validating the position request.\nThe message was:\n{msg}")
            raise ValidationError(e)
        self.next_position = next_pos

    def send_next_pos_to_lv(self):
        self.lv_socket.send_string(self.next_position.model_dump_json())

    def get_lv_dataset(self):
        try:
            msg = self.lv_socket.recv_json()
            new_data = self.DataMessage(**msg)
        except ValidationError as e:
            print(f"There was an error while validating the position request.\nThe message was:\n{msg}")
            raise ValidationError(e)
        self.current_dataset = new_data
        self.num_data_collected += 1

    def send_data_to_RE_manager(self):
        d = self.current_dataset.model_dump_json()
        self.re_socket.send_string(d)
        response = self.re_socket.recv()

    def send_lv_OK(self):
        self.lv_socket.send(b'OK')


if __name__ == "__main__":
    t = Translator(use_real_lv_messages=False)
