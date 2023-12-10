import zmq
import base64

from bluesky import RunEngine
from bluesky.plans import count, scan
from bluesky.plan_stubs import mv
from bluesky_adaptive.on_stop import recommender_factory
from bluesky_adaptive.per_start import adaptive_plan


import numpy as np
from sim_ARPES import simulate_ARPES_measurement
from ophyd.sim import SynAxis
from ophyd import Device, Component, Signal
from ophyd.device import create_device_from_components

from messages import *
from translator_server import Translator
from motors import LVSignal
from agents import MangerAgent


class SynMotor(SynAxis):
    def __init__(self, *args, bounds, delta, **kwargs):
        self.bounds = bounds
        self.delta = delta
        super().__init__(*args, **kwargs)

    def read(self):
        print(f"Reading {self.name}")
        res = super().read()
        return res
    
    def set(self, value):
        print(f"Setting {self.name} = {value}")
        res = super().set(value)
        return res


class SynDetector(Device):
    data = Component(Signal, value=0)
    def __init__(self, *args, data_shape, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay = 1 if kwargs.get('delay') is None else kwargs.get('delay')  # simulated exposure time delay
#         self.manipulator = kwargs.get('manipulator') # Either None or a Manipulator for simulated
        # self.index = index
        self.data_shape = data_shape
        self.data.put(np.zeros(self.data_shape).astype(np.float64))

    def read(self):
        print(f"Reading Detector")
        res = super().read()
        return res

    # def trigger(self):
    #     # No need to actually trigger because on each start we will set the data
    #     print("Triggering Detector")
    #     # self.data_component.set(np.ones(self.data_shape))
    #     pass

    # def set(self, value):
    #     self.data.put(value)

    def collect_asset_docs(self):
        yield from []

class RunEngineManager:
    def __init__(self):
        self.RE = RunEngine()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind('tcp://127.0.0.1:5550')
        print('RE: Listening for position request...')
        start_msg = REManagerStartupMessage(**self.socket.recv_json())
        self.max_count = start_msg.max_count
            # defn = {
            #     'theta': (LVSignal, 'theta', {'value': 0.1, 'bounds': (-15.0, 15.0)}),
            #     'phi': (LVSignal, 'phi', {'value': -3, 'bounds': (-15.0, 15.0)}),
            #     'img': (LVSignal, 'img', {'value': np.zeros((64,64)), 'write_access': False}),
            # }
        # motor_defn = {
        #     motor.name: (LVSignal, motor.name, {'value': np.mean(motor.bounds), 'bounds': motor.bounds, 'delta': motor.delta})
        #     for motor in start_msg.motor_defs
        # }

        # data_defn = {
        #     data.name: (LVSignal, data.name, {'value': np.zeros(data.shape).astype(np.float64)})
        #     for data in start_msg.data_defs 
        # }

        # print(motor_defn)
        # print(data_defn)

        self.motors = [SynMotor(name=motor.name, value=np.mean(motor.bounds), bounds=motor.bounds, delta=motor.delta) for motor in start_msg.motor_defs]
        self.detectors = [SynDetector(name=data.name, data_shape=data.shape) for data in start_msg.data_defs]

        # print(motors)
        # print(detectors)
        self.socket.send(b'Startup OK.')

        self.RE.subscribe(self.update_datasets, name='stop')
        self.RE.subscribe(self.update_motor_coordinates, name='start')

        print(self.start_RE())
        
        # while True:
        #     print('RE: Listening for position request...')
        #     print(self.socket.recv_json())
        #     pos = REManagerMotorPositionMessage(positions=np.random.uniform(0,1,2).tolist())
        #     print(f'RE: Sending position: {pos}')
        #     self.socket.send_string(pos.model_dump_json())
        #     print(f'RE: Waiting for data...')
        #     self.socket.recv_json()
        #     print(f'RE: Sending OK.')
        #     self.socket.send(b'OK')

    def update_motor_coordinates(self, *args, **kwargs):
        pos_req_msg = self.socket.recv()
        positions = [motor.get().readback for motor in self.motors]
        names = [motor.name for motor in self.motors]
        new_pos_msg = REManagerMotorPositionMessage(positions=positions, names=names)
        self.socket.send_string(new_pos_msg.model_dump_json())

    def update_datasets(self, *args, **kwargs):
        data_incoming_req = self.socket.recv_json()
        data_message = REManagerDataMessage(**data_incoming_req)
        if len(data_message.names) != len(self.detectors):
            raise ValueError("Different number of data names and detectors?")
        for name, data_bytes, data_shape, detector in zip(data_message.names, data_message.datasets, data_message.shapes, self.detectors):
            print(name)
            print(detector.data.name)
            if name != detector.name:
                raise ValueError("Somehow the data order is different than when defined. Change to a dict.")
            np_arr = np.frombuffer(base64.decodebytes(data_bytes), dtype=np.float64).reshape(data_shape)
            detector.data.put(np_arr)
        # new_pos_msg = REManagerMotorPositionMessage(positions=positions)
        self.socket.send_string('RE: Updated data Ok.') 

    def start_RE(self):
        unique_ids = self.RE(
            self.with_agent(
                MangerAgent(
                        max_count=self.max_count,
                        input_bounds=[motor.bounds for motor in self.motors],
                        input_min_spacings=[motor.delta for motor in self.motors],
                        n_independent_keys=len(self.motors),
                        dependent_shape=self.detectors[0].data_shape,
                        # search_motors=search_motors,
                        # motor_bounds=motor_bounds,
                ),
                search_motors=self.motors,
                initial_positions=[motor.get().readback for motor in self.motors], 
                detectors=self.detectors,
                # dependent_keys=[detector.name for detector in detectors]
            ),
            # callback,
        )


    def my_reccomender_plan(self, detectors, agent, search_motors, initial_positions):
        recommender = agent

        cb, queue = recommender_factory(
            adaptive_obj=recommender,
            independent_keys=[m.name for m in search_motors],
            dependent_keys=[detector.data.name for detector in detectors],
            target_keys=[m.name for m in search_motors],
            target_transforms=dict(),
            max_count=agent.max_count,
        )
            
        yield from adaptive_plan(
            dets=detectors,
            first_point={m: p for m, p in zip(search_motors, initial_positions)},
            to_recommender=cb,
            from_recommender=queue,
        )

    def with_agent(self, agent, detectors, search_motors, initial_positions):
        return (
            yield from self.my_reccomender_plan(
                agent=agent,
                detectors=detectors,
                search_motors=search_motors,
                initial_positions=initial_positions,
            )
        )



if __name__ == "__main__":
    # class SynARPESDetector(Device):
    #     data_shape = (256, 256)
    #     image = Component(Signal, value=np.zeros(data_shape))

    #     def __init__(self, *args, **kwargs):
    #         super().__init__(*args, **kwargs)
    #         self.delay = 1 if kwargs.get('delay') is None else kwargs.get('delay')  # simulated exposure time delay
    # #         self.manipulator = kwargs.get('manipulator') # Either None or a Manipulator for simulated

    #     def trigger(self):
    # #         arr = generate_image(self.data_shape)
    #         #arr = generate_batch(1, noise=0.5, k_resolution=0.01, e_resolution=0.01, num_angles=256, num_energies=256, random_bands=False)[0][0]
    #         arrs = np.zeros((256,256)).astype(np.float64)
    #         # Update the internal signal with a simulated image.
    #         self.image.set(arrs[0])
    #         # Simulate the exposure and readout time with a tunable "delay".

    #     def collect_asset_docs(self):
    #         yield from []
            
    # detectors = [SynARPESDetector(name='detector')]
    re_man = RunEngineManager()

    # motor_defs = [
    #     REManagerMotorDefMessage(**{'name': 'x', 'bounds': (0., 1.), 'delta': 0.01}),
    #     REManagerMotorDefMessage(**{'name': 'y', 'bounds': (0., 1.), 'delta': 0.05})
    # ]

    # data_defs = [
    #     REManagerDataDefMessage(**{'name':'ARPES', 'shape':[4,4]})
    # ]



    # # motors = [SynMotor(name=motor.name, value=np.mean(motor.bounds), bounds=motor.bounds, delta=motor.delta) for motor in motor_defs]
    # detectors = [SynDetector(name=data.name, data_shape=data.shape) for data in data_defs]
    # d = detectors[0]
    # print(d.data)
    # print('-'*20)
    # d.data.put(69 * np.ones((4,4)).astype(np.float64))
    # print(d.data)
    # print([motor.get().readback for motor in motors])
    # for detector in detectors:
    #     print(detector.read())



        # print(detector.trigger())
        # print(detector.read())
    # m = MotorController(defn=defn)
    # print(m)


    # controller = MyDynamicDevice(defn, component_class=LVSignal)
    # print(controller.components)
    # print(controller.read())
    # controller.components['theta'].set(1)
    # controller.components['theta'].get()



    # exit()

    # import threading

    # class TimerStatus(DeviceStatus):
    #     """Simulate the time it takes for a detector to acquire an image."""

    #     def __init__(self, device, delay):
    #         super().__init__(device)
    #         self.delay = delay  # for introspection purposes
    #         threading.Timer(delay, self.set_finished).start()

    # class MySynAxis(SynAxis):
    #     def read(self):
    #         print(f"Reading {self.name}")
    #         res = super().read()
    #         return res
        
    #     def set(self, value):
    #         print(f"Setting {self.name}")
    #         res = super().set(value)
    #         return res

    # class Manipulator(Device):
    #     x = MySynAxis(name='x')
    #     y = MySynAxis(name='y')
    #     z = MySynAxis(name='z')
    #     theta = MySynAxis(name='theta')
    #     phi = MySynAxis(name='phi')
    #     omega = MySynAxis(name='omega')


    # class SynARPESDetector(Device):
    #     data_shape = (256, 256)
    #     image = Component(Signal, value=np.zeros(data_shape))
    #     manipulator = Manipulator(name='manipulator')#, read_attrs=['x','y','z','theta','phi','omega'])

    #     def __init__(self, *args, **kwargs):
    #         super().__init__(*args, **kwargs)
    #         self.delay = 1 if kwargs.get('delay') is None else kwargs.get('delay')  # simulated exposure time delay
    # #         self.manipulator = kwargs.get('manipulator') # Either None or a Manipulator for simulated

    #     def read(self):
    #         print(f"Reading SynARPESDetector")
    #         res = super().read()
    #         return res

    #     def trigger(self):
    # #         arr = generate_image(self.data_shape)
    #         #arr = generate_batch(1, noise=0.5, k_resolution=0.01, e_resolution=0.01, num_angles=256, num_energies=256, random_bands=False)[0][0]
    #         arrs = simulate_ARPES_measurement(
    #             polar = self.manipulator.theta.get().readback,
    #             tilt = self.manipulator.phi.get().readback, 
    #             azimuthal = self.manipulator.omega.get().readback,
    #             photon_energy=100.0, noise_level=0.1,
    #             acceptance_angle=30.0, num_angles=256,
    #             num_energies=256, temperature=30.0,
    #             k_resolution=0.011, e_resolution=0.025,
    #             energy_range=(-0.7, 0.1), random_bands=False
    #         )
    #         # Update the internal signal with a simulated image.
    #         self.image.set(arrs[0])
    #         # Simulate the exposure and readout time with a tunable "delay".
    #         return TimerStatus(self, self.delay)

    #     def collect_asset_docs(self):
    #         yield from []
            
    # detector = SynARPESDetector(name='detector')

    # search_motors = [detector.manipulator.theta, detector.manipulator.omega, detector.manipulator.phi]
    # motor_bounds = np.array([[-10,10], [0,360], [-10,10]])
    # initial_positions = np.array([0, 0, 2])

    # RE = RunEngine()

    # def stop_printer(name, doc):
    #     print("-"*20)
    #     print(name)
    #     print("-"*20)
    #     # print(doc)
    #     # print("-"*20)

    # RE.subscribe(stop_printer, name='all')

    # unique_ids = RE(
    #     with_agent(
    #         MangerAgent(
    #                 max_count=3,
    #                 input_bounds=motor_bounds,
    #                 input_min_spacings=[1,1,1],
    #                 n_independent_keys=3,
    #                 dependent_shape=(256,256),
    #                 # search_motors=search_motors,
    #                 # motor_bounds=motor_bounds,
    #         ),
    #         search_motors=search_motors,
    #         initial_positions=initial_positions, 
    #         detector=detector,
    #     ),
    #     # callback,
    # )