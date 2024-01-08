import time
import zmq
import numpy as np
from motors import LVSignal
from pydantic import BaseModel, ValidationError
from messages import *
from maestro_messages import MaestroLVStartupMessage, MaestroLVPositionRequest, MaestroLVPositionResponse, MaestroLVResponse, MaestroLVResponseError, MaestroLVResponseOK, MaestroLVDataMessage, MaestroLVCloseMessage, MaestroLVAbortMessage, MotorPosition

class AbortError(Exception):
    """An Exception to stop translator when an Abort is seen."""
    pass

class CloseError(Exception):
    """An Exception to stop translator when an Close is seen."""
    pass

class Translator:
    def __init__(self, translator_addr = 'tcp://*:5550', re_addr = 'tcp://einstein.dhcp.lbl.gov:5551', use_real_lv_messages=True):
        # setup zmq communication
        self.addr = translator_addr
        self.re_addr = re_addr
        self.context = zmq.Context()
        self.lv_socket = self.context.socket(zmq.REP)
        self.lv_socket.bind(self.addr)
        self.re_socket = self.context.socket(zmq.REQ)
        self.re_socket.connect(self.re_addr)

        self.startup()

    def startup(self):
        self.num_data_collected = 0
        print("Translator: Waiting for message from lv_socket...")
        msg_json = self.recv_lv_json()
        start_msg = MaestroLVStartupMessage(**msg_json)
        # self.lv_socket.send(b'OK. Recieved Startup')
        self.send_lv_OK()
        self.max_count = start_msg.max_count
        self.current_ai_cycle = 0
        # print(type(message))
        # print(MaestroLVStartupMessage(**message))
        first_positions = [MotorPosition(axis_name=motor.device_name, value=np.mean([motor.high, motor.low])) for motor in start_msg.AIModeparms]
        first_pos_req = self.recv_lv_json()
        print(f'T: First positions: {first_positions}')
        print(f'T: First position request: {first_pos_req}')
        first_position_response = MaestroLVPositionResponse(status='OK', positions=first_positions)
        print(f"First position response: {first_position_response}")
        self.lv_socket.send_string(first_position_response.model_dump_json())
        data_message = self.recv_lv_json()
        first_data_msg = MaestroLVDataMessage(**data_message)
        self.send_lv_OK()
        # self.lv_socket.send(b'Data Recieved OK.')

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
        print(self.recv_re_msg())
        # print(self.re_socket.recv())
        self.operation_loop()

    def operation_loop(self):
        try:
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
        except (AbortError, CloseError) as e:
            print("T: Resetting...")
            self.startup()

    def recv_lv_json(self):
        msg_json = self.lv_socket.recv_json()
        method = msg_json.get('method')
        print(f"T: Method recieved {method}")
        if method == MaestroLVAbortMessage().method:
            self.send_lv_OK()
            self.send_re_restart()
            # self.lv_socket.send(b'Aborting.')
            raise AbortError
        elif method == MaestroLVCloseMessage().method:
            self.send_lv_OK()
            # self.send_re_restart()
            # self.lv_socket.send(b'Closing.')
            raise CloseError
        return msg_json

    def send_re_restart(self):
        self.re_socket.send_string(RERestartMessage().model_dump_json())
        self.recv_re_msg()
        self.startup()
    
    def recv_re_msg(self):
        cutoff = 120.0 #seconds
        start_time = time.time()
        while time.time() - start_time < cutoff:
            try:
                msg = self.re_socket.recv(flags=zmq.NOBLOCK)
                return msg
            except zmq.Again as e:
                pass
        else:
            self.send_lv_Error()
            raise TimeoutError(f"Did not recieve a message back from the REManager in {cutoff}s.\nAn error has been sent to LV.")

    def get_lv_pos_req(self):
        try:
            msg = self.recv_lv_json()
            pos_req = MaestroLVPositionRequest(**msg)
        except ValidationError as e:
            print(f"There was an error while validating the position request.")
            raise ValidationError(e)
        self.position_request = pos_req

    def get_pos_from_RE_manager(self):
        self.re_socket.send_string(self.position_request.model_dump_json())
        try:
            msg = self.recv_re_msg()
            # print(msg)
            # raise Exception
            next_pos = REManagerMotorPositionMessage.model_validate_json(msg)
        except ValidationError as e:
            print(f"There was an error while validating the position request.\nThe message was:\n{msg}")
            raise ValidationError(e)
        self.next_position = next_pos

    def send_next_pos_to_lv(self):
        next_position_msg = MaestroLVPositionResponse(
            status='OK',
            positions=[
                MotorPosition(axis_name=name, value=value)
                for name, value in zip(self.next_position.names, self.next_position.positions)
            ])
        self.lv_socket.send_string(next_position_msg.model_dump_json())

    def get_lv_dataset(self):
        try:
            msg = self.recv_lv_json()
            new_data = MaestroLVDataMessage(**msg)
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
        response = self.recv_re_msg()

    def send_lv_OK(self):
        self.lv_socket.send_string(MaestroLVResponseOK().model_dump_json())
    
    def send_lv_Error(self):
        self.lv_socket.send_string(MaestroLVResponseError().model_dump_json())


if __name__ == "__main__":
    t = Translator(use_real_lv_messages=True)
