import zmq
import numpy as np
import base64
from messages import *
import time

class FakeLV:
    def __init__(self, translator_addr = 'tcp://127.0.0.1:5551'):
        self.translator_addr = translator_addr
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(translator_addr)
        self.startup()
        self.operation_loop()

    def startup(self):
        msg = FakeLVStartupMessage(
            max_count=5,
            motor_defs=[
            REManagerMotorDefMessage(name='x', bounds=[0.,1.], delta=0.05),
            REManagerMotorDefMessage(name='y', bounds=[0.,1.], delta=0.1)
            ]
        )
        print("LV: Sending startup message to translator...")
        self.socket.send_string(msg.model_dump_json())
        print("LV: Sent.")
        resp = self.socket.recv()
        print(f"LV: Recieved response: {resp}")

    def operation_loop(self):
        while True:
            print("LV: Sending position request")
            self.send_pos_req()
            print("LV: Waiting for next position...")
            self.recieve_next_position()
            print("LV: Sending data for that position...")
            time.sleep(0.5)
            self.send_data()

    def send_pos_req(self):
        self.socket.send_string(FakeLVPositionRequest().model_dump_json())
    
    def recieve_next_position(self):
        msg = self.socket.recv_json()
        print(msg)
        position = REManagerMotorPositionMessage(**msg)
        self.position = position.positions
        print(f'LV: Now at position: {self.position}')

    def send_data(self):
        data = self.generate_data()
        data_bytes = base64.encodebytes(data.tobytes())
        self.socket.send_string(FakeLVDataMessage(names=['ARPES'],datasets=[data_bytes], shapes=[list(data.shape)]).model_dump_json())
        msg = self.socket.recv()
        print(f'LV: Recieved msg after sending data: {msg}')
    
    def generate_data(self):
        return np.random.uniform(0,1,(1024,1024)).astype(np.float64)


f = FakeLV()