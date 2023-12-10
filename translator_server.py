import zmq
import numpy as np
from motors import LVSignal
from pydantic import BaseModel, ValidationError
from messages import *
from maestro_messages import MaestroLVStartupMessage, MaestroLVPositionRequest, MaestroLVPositionResponse, MaestroLVDataMessage, MaestroLVShutdownMessage, MotorPosition


class Translator:
    def __init__(self, translator_addr = 'tcp://127.0.0.1:5551', re_addr = 'tcp://127.0.0.1:5550', use_real_lv_messages=True):
        # setup zmq communication
        self.addr = translator_addr
        self.re_addr = re_addr
        self.StartupMessage = MaestroLVStartupMessage if use_real_lv_messages else FakeLVStartupMessage
        self.PositionRequest = MaestroLVPositionRequest if use_real_lv_messages else FakeLVPositionRequest
        self.PositionResponse = MaestroLVPositionResponse if use_real_lv_messages else FakeLVPositionResponse
        self.DataMessage = MaestroLVDataMessage if use_real_lv_messages else FakeLVDataMessage
        self.ShutDownMessage = MaestroLVShutdownMessage if use_real_lv_messages else FakeLVShutdownMessage
        self.startup()

    def startup(self):
        self.num_data_collected = 0
        self.context = zmq.Context()
        self.lv_socket = self.context.socket(zmq.REP)
        self.lv_socket.bind(self.addr)
        self.re_socket = self.context.socket(zmq.REQ)
        self.re_socket.connect(self.re_addr)

        print("Translator: Waiting for message from lv_socket...")
        start_msg = MaestroLVStartupMessage(**self.lv_socket.recv_json())
        self.max_count = start_msg.max_count
        # print(type(message))
        # print(self.StartupMessage(**message))
        self.lv_socket.send(b'ok')
        first_positions = [MotorPosition(axis_name=motor.device_name, value=np.mean([motor.high, motor.low])) for motor in start_msg.AIModeparms]
        first_pos_req = self.lv_socket.recv_json()
        print(f'T: First positions: {first_positions}')
        print(f'T: First position request: {first_pos_req}')
        first_position_response = self.PositionResponse(first_positions)
        print(f"First position response: {first_position_response}")
        self.lv_socket.send_string(first_position_response.model_dump_json())
        first_data_msg = self.DataMessage(**self.lv_socket.recv_json())
        self.lv_socket.send(b'Data Recieved OK.')

        re_motor_defs = [
            REManagerMotorDefMessage(name=motor.device_name, bounds=[motor.low, motor.high], delta=motor.min_step)
            for motor in start_msg.AIModeparms
        ]
        #TODO: Only currently keeping the 2d detector!
        re_data_defs = [
            REManagerDataDefMessage(name=desc.fieldname, shape=desc.dimensions)
            for desc in first_data_msg.fits_descriptors if len(desc.dimensions) == 2
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
        next_position_msg = MaestroLVPositionResponse(
            [
                MotorPosition(axis_name=name, value=value)
                for name, value in zip(self.next_position.names, self.next_position.positions)
            ])
        self.lv_socket.send_string(next_position_msg.model_dump_json())

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
        d = self.current_dataset
        #TODO: Currently only using the 2d dataset
        names=[]
        datasets=[]
        shapes=[]
        for desc in d.fits_descriptors:
            if len(desc.dimensions) == 2:
                names.append(desc.fieldname)
                datasets.append(desc.Data)
                shapes.append(desc.dimensions)
        d = REManagerDataMessage(
            names=names,
            datasets=datasets,
            shapes=shapes,
        )
        self.re_socket.send_string(d.model_dump_json())
        response = self.re_socket.recv()

    def send_lv_OK(self):
        self.lv_socket.send(b'OK')


if __name__ == "__main__":
    t = Translator(use_real_lv_messages=True)
