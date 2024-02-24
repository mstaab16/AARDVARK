import time
import numpy as np
import base64
import json
import logging

from sqlalchemy.orm import load_only

from db.database import get_db
from db.models import Experiment, Decision, Measurement, Data, Report

from celery_workers import agents
from celery_workers.celery_app import app

from maestro_api import maestro_messages as mm

# db_gen = get_db()
# db = next(db_gen)
# db = next(get_db())

class ExperimentWatcher:
    def __init__(self, experiment_id):
        self.db_gen = get_db()
        self.db = next(self.db_gen)
        self.experiment_id = experiment_id
        self.experiment = self.db.query(Experiment).get(experiment_id)
        self.motors = [mm.AIModeparm.model_validate_json(motor) for motor in self.experiment.motors['motors']]
        self.motor_lows = np.array([motor.low for motor in self.motors])
        self.motor_highs = np.array([motor.high for motor in self.motors])
        self.motor_min_steps = np.array([motor.min_step for motor in self.motors])
        self.max_positions_per_motor = [int(np.ceil((motor.high - motor.low) / motor.min_step)) for motor in self.motors]
        self.all_possible_measurement_indices = np.arange(np.prod(self.max_positions_per_motor))
        self.measured_indices = np.array([])
        self.data_ids = []
        self.data_indices = []
        self.x = []
        self.y = []
        
        self.pipeline = [
            # agents.PCAAgent(n_components=.95),
            agents.KMeansAgent(n_clusters=5, experiment_id=self.experiment_id), 
            agents.GPytorchAgent(input_bounds=[[low,high] for low, high in zip(self.motor_lows, self.motor_highs)], input_min_spacings=self.motor_min_steps, experiment_id=self.experiment_id)
            ]

    def run(self):
        # Create the first decision
        num_random_measurements = 10
        num_boundary_measurements = 3

        startup_decision = Decision(
            experiment_id = self.experiment_id,
            method = f"Startup: Corners and {num_random_measurements} random measurements.")
        self.db.add(startup_decision)
        self.db.commit()

        corners = [[self.motor_lows[0], self.motor_lows[1]],
                   [self.motor_highs[0], self.motor_lows[1]],
                   [self.motor_lows[0], self.motor_highs[1]],
                   [self.motor_highs[0], self.motor_highs[1]],
                   ]
        boundary_measurements = []
        boundary_measurements.extend([[self.motor_lows[0], y] for y in np.linspace(self.motor_lows[1], self.motor_highs[1], num_boundary_measurements)])
        boundary_measurements.extend([[x, self.motor_highs[1]] for x in np.linspace(self.motor_lows[0], self.motor_highs[0], num_boundary_measurements)])
        boundary_measurements.extend([[self.motor_highs[0], y] for y in np.linspace(self.motor_lows[1], self.motor_highs[1], num_boundary_measurements)[::-1]])
        boundary_measurements.extend([[x, self.motor_lows[1]] for x in np.linspace(self.motor_lows[0], self.motor_highs[0], num_boundary_measurements)[::-1]])
        for pos in boundary_measurements:
            pos_dict = {motor.device_name: pos[i] for i, motor in enumerate(self.motors)}
            measurement = Measurement(experiment_id=self.experiment.experiment_id, decision_id=startup_decision.decision_id,
                                    positions=pos_dict, measured=False, measurement_time="", ai_cycle=None)
            self.db.add(measurement)

        # Create the first measurements
        measurements = self.create_random_measurements(num_random_measurements, startup_decision.decision_id)
        self.db.add_all(measurements)
        self.db.commit()
        # time.sleep(5)

        while True:
            loop_start = time.perf_counter()
            # Reset loop_end each time because sometimes we will continue the loop early
            loop_end = time.perf_counter()
            # Check if experiment is still active
            self.experiment = self.db.query(Experiment).get(self.experiment_id)
            if not self.experiment.active:
                print("Experiment is no longer active. Shutting down...")
                return
            new_measurements = self.update_data()
            if len(self.data_ids) < len(boundary_measurements) + num_random_measurements* 0.75:
                print("Not enough data to train on")
                time.sleep(1)
                continue
            self.measured_positions = self.db.query(Measurement.positions).filter(Measurement.experiment_id == self.experiment_id).all()
            # print("*"*20, f"{len(measured_positions)=}")
            self.measured_positions = np.array([list(x[0].values()) for x in self.measured_positions])
            if len(self.measured_positions) > 0:
                self.measured_indices = position_to_index(self.measured_positions, self.motor_lows, self.motor_min_steps, self.max_positions_per_motor)
            #     print("*"*20, f"{measured_indices.shape=}")
                # print(len(self.measured_indices))
                # print(len(np.unique(self.measured_indices)))
                # print(self.measured_indices)
            #     self.all_positions_measured = np.zeros(self.max_positions_per_motor, dtype=np.bool_)
            #     self.all_positions_measured[*measured_indices.T] = True
                # self.all_positions_measured[measured_indices.T] = True
            #     print("*"*20, self.all_positions_measured.sum())
            # print(self.all_positions_measured)
            # Create a new decision
            decision = Decision(experiment_id=self.experiment_id, method={"method": "AARDVARK"})
            self.db.add(decision)
            self.db.commit()

            # Calculate next moves
            
            # time.sleep(1)
            # remaining_measurements = np.prod(self.max_positions_per_motor) - self.all_positions_measured.sum()
            # num = np.min([num, remaining_measurements])
            # # print(f"{remaining_measurements=}, {num=}")
            if len(self.measured_indices) == len(self.all_possible_measurement_indices):
                print("All measurements have been suggested?!?!?")
                continue
            num_to_suggest=25
            # measurements = self.create_random_measurements(num_to_suggest, decision.decision_id)
            measurements = self.create_smart_measurements(num_to_suggest, decision.decision_id)
            # non_duplicates = []
            # for measurement in measurements:
            #     idx = position_to_index(np.array([list(measurement.positions.values())]), self.motor_lows, self.motor_min_steps, self.max_positions_per_motor)
            #     if idx not in measured_indices:
            #         non_duplicates.append(measurement)
            # measurements = self.create_smart_measurements(10, decision.decision_id)
            # Order the moves in a reasonable way

            # Remove all unmeasured moves from the database
            self.db.query(Measurement).filter(Measurement.measured == False).delete()

            # Add the new moves to the database
            self.db.add_all(measurements)

            # Commit the changes
            self.db.commit()
            loop_end = time.perf_counter()
            loop_time = (loop_end - loop_start)
            suggestions_per_sec = new_measurements/loop_time
            print(f"Loop time: {loop_time:.02f}\tNum measured: {new_measurements:.1f}\tSuggestions per sec: {suggestions_per_sec:.1f}")
            if suggestions_per_sec > 10 or suggestions_per_sec < 5:
                num_to_suggest = 2 * suggestions_per_sec * loop_time
                num_to_suggest = max(num_to_suggest, 5)
                print(f"UPDATING NUM TO SUGGEST TO: {num_to_suggest}")


    def update_data(self):
        print("*"*20)
        print('Getting all data')
        start = time.perf_counter_ns()
        query = self.db.query(Measurement.positions, Data.data_id, Data.data, Data.data_info).join(Measurement.data)\
                .filter(~Data.data_id.in_(self.data_ids))\
                .filter(Measurement.experiment_id == self.experiment_id, Data.fieldname == 'Fixed_Spectra5').all()
        
        # data, data_info = db.query(Data.data, Data.data_info).filter(Data.experiment_id == experiment_id, Data.fieldname == "Fixed_Spectra5").order_by(Data.measurement_id.desc()).first()
        for pos, data_id, data, data_info in query:
            data_info = json.loads(data_info)
            self.data_ids.append(data_id)
            self.data_indices.append(position_to_index(np.array([list(pos.values())]), self.motor_lows, self.motor_min_steps, self.max_positions_per_motor))
            self.x.append(np.array(list(pos.values())))
            self.y.append(np.frombuffer(data, dtype=np.int32).reshape(*data_info['dimensions'], order='F').flatten())
        # positions_and_ids = self.db.query(Measurement.positions, Data.data_id).join(Measurement.data).filter(Measurement.experiment_id == self.experiment_id, Data.fieldname == 'Fixed_Spectra0').all()
        # if len(positions_and_ids) == 0:
        #     print("No data found")
        #     return
        # positions = np.array([list(pos.values()) for pos, _ in positions_and_ids])
        # all_data_ids = np.array([i for _, i in positions_and_ids])
        # indices = position_to_index(positions, self.motor_lows, self.motor_min_steps, self.max_positions_per_motor)
        # print(f'{indices=}')
        # print(f'{all_data_ids=}')
        # # keep_indices = np.where(np.isin(indices, self.data_indices, invert=True))
        # keep_indices = np.setdiff1d(indices, self.data_indices)
        # print(f'{keep_indices=}')
        # new_ids = all_data_ids[keep_indices].tolist()
        # if len(new_ids) == 0:
        #     print("No new data found")
        #     return
        # new_stuff = self.db.query(Data.data_id, Data.data, Measurement.positions).join(Data.measurement).filter(Data.data_id.in_(new_ids)).order_by(Data.data_id).all()
        # print(f"Found {len(new_stuff)} new data points")
        # self.data_indices.extend(keep_indices)
        # self.data_ids.extend([i for i, _, _ in new_stuff])
        # self.x.extend([np.array(list(pos.values())) for _, _, pos in new_stuff])
        # self.y.extend([np.frombuffer(base64.decodebytes(d)).reshape(128,128) for _, d, _ in new_stuff])
        # # self.x.update({idx: np.array(list(pos.values())) for idx, _, pos in new_stuff})
        # # self.y.update({idx: d for idx, d, _ in new_stuff})

        print(f'Updated with {len(query)} new datasets in {(time.perf_counter_ns() - start) / 1e6:.02} ms.')
        return len(query)

    def create_random_measurements(self, num_measurements, decision_id):
        decision = self.db.query(Decision).get(decision_id)
        experiment = self.db.query(Experiment).get(decision.experiment_id)
        options = np.setdiff1d(self.all_possible_measurement_indices, self.measured_indices)
        random_indices = np.random.choice(options, np.min([num_measurements, len(options)]),replace=False)
        positions = index_to_position(np.array(random_indices), self.motor_lows, self.motor_min_steps, self.max_positions_per_motor)
        measurements = []
        for position in positions:
            pos_dict = {motor.device_name: position[i] for i, motor in enumerate(self.motors)}
            measurement = Measurement(experiment_id=experiment.experiment_id, decision_id=decision_id,
                                    positions=pos_dict, measured=False, measurement_time="", ai_cycle=None)
            measurements.append(measurement)
        return measurements

    def create_smart_measurements(self, num_measurements, decision_id):
        decision = self.db.query(Decision).get(decision_id)
        experiment = self.db.query(Experiment).get(decision.experiment_id)

        x = np.array(self.x)
        y = np.array(self.y)
        for agent in self.pipeline:
            x, y = agent.tell(x, y)
        measurements = []
        # agent_output = agent.ask(num_measurements)
        # print(agent_output)
        for position in agent.ask(num_measurements):
            pos_dict = {motor.device_name: float(position[i]) for i, motor in enumerate(self.motors)}
            measurement = Measurement(experiment_id=experiment.experiment_id, decision_id=decision_id,
                                    positions=pos_dict, measured=False, measurement_time="", ai_cycle=None)
            measurements.append(measurement)
        return measurements 

@app.task
def setup_experiment_watcher(experiment_id):
    experiment_watcher = ExperimentWatcher(experiment_id)
    experiment_watcher.run()

@app.task
def save_data(msg):
    data = mm.MaestroLVDataMessage(**msg)
    db_gen = get_db()
    db = next(db)
    experiment_id = db.query(Experiment).order_by(Experiment.experiment_id.desc()).first().experiment_id

    current_measurement = db.query(Measurement).filter(Measurement.experiment_id == experiment_id, Measurement.measured == True).order_by(Measurement.measurement_id.desc()).first()

    current_measurement.ai_cycle = data.message.current_AI_cycle
    for fd in data.fits_descriptors:
        logging.info(f"Recieved Data with field name: {fd.fieldname}")
        if data.message.method == "fake_labview":
            logging.info('Decoding data first...')
            dataset = base64.decodebytes(fd.Data)
            # logging.info(dataset)
        else:
            dataset = fd.Data
        fd_without_data = dict(fd)
        del fd_without_data["Data"]
        # logging.info("SAVING DATA: ", fd.Data)
        new_data = Data(
            experiment_id = experiment_id,
            measurement_id = current_measurement.measurement_id,
            message = data.message.model_dump_json(),
            fieldname = fd.fieldname,
            data_cycle = data.message.current_data_cycle,
            data = dataset,
            data_info = json.dumps(fd_without_data),
        )
        db.add(new_data)

    db.commit()

def position_to_index(positions, motor_lows, motor_min_steps, max_positions_per_motor):
    indices = np.floor((positions - motor_lows) / motor_min_steps).astype(np.int_).T
    return np.ravel_multi_index(indices, max_positions_per_motor, mode='clip')

def index_to_position(indices, motor_lows, motor_min_steps, max_positions_per_motor):
    indices = np.array(np.unravel_index(indices, max_positions_per_motor))
    positions = (indices * motor_min_steps[:, np.newaxis] + motor_lows[:, np.newaxis]).T
    return positions

# @app.task
# def setup_experiment_watcher(experiment_id):
#     print("Setting up experiment watcher")

#     # Represent all posisble motor positions as a numpy array of i32s
#     experiment = db.query(Experiment).get(experiment_id)
#     motors = [mm.AIModeparm.model_validate_json(motor) for motor in experiment.motors['motors']]
#     max_positions_per_motor = [int((motor.high - motor.low) / motor.min_step) for motor in motors]
#     all_positions_measured = np.zeros(max_positions_per_motor, dtype=np.bool_)

#     # Create the first decision
#     num_random_measurements = 10

#     new_decision = Decision(
#         experiment_id = experiment_id,
#         method = f"Startup: {num_random_measurements} random measurements.")
#     db.add(new_decision)
#     db.commit()

#     # Create the first measurements
#     create_random_measurements(num_random_measurements, new_decision.decision_id, all_positions_measured, max_positions_per_motor)

#     while True:
#         # Check if experiment is still active
#         experiment = db.query(Experiment).get(experiment_id)
#         if not experiment.active:
#             print("Experiment is no longer active. Shutting down...")
#             return
#         # Create a new decision
#         decision = Decision(experiment_id=experiment_id, method={"method": "AARDVARK"})
#         db.add(decision)
#         db.commit()

#         # Calculate next moves
#         measurements = create_random_measurements(10, decision.decision_id, all_positions_measured, max_positions_per_motor)
#         # Order the moves in a reasonable way

#         # Remove all unmeasured moves from the database
#         db.query(Measurement).filter(Measurement.measured == False).delete()

#         # Add the new moves to the database
#         db.add_all(measurements)

#         # Commit the changes
#         db.commit()

# @app.task
# def create_smart_experiments(num_measurements: int, decision_id: int) -> list[Measurement]:
#     pass

# # def create_random_measurements(num_measurements: int, decision_id: int) -> list[Measurement]:
# @app.task
# def create_random_measurements(num_measurements, decision_id, all_positions_measured, max_positions_per_motor):
#     decision = db.query(Decision).get(decision_id)
#     experiment = db.query(Experiment).get(decision.experiment_id)
    
#     motors = [mm.AIModeparm.model_validate_json(motor) for motor in experiment.motors['motors']]
#     measurements = []
#     indices_not_measured = np.where(all_positions_measured == False)
#     indices_not_measured_flat = np.ravel_multi_index(indices_not_measured, max_positions_per_motor)
#     random_indices = np.random.choice(indices_not_measured_flat, num_measurements)
#     random_indices = np.unravel_index(random_indices, max_positions_per_motor)
#     positions = index_to_position(np.array(random_indices), motors, max_positions_per_motor)
#     for position in positions:
#         pos_dict = {motor.device_name: position[i] for i, motor in enumerate(motors)}
#         measurement = Measurement(experiment_id=experiment.experiment_id, decision_id=decision_id,
#                                 positions=pos_dict, measured=False, measurement_time="", ai_cycle=None)
#         measurements.append(measurement)
#     return measurements
