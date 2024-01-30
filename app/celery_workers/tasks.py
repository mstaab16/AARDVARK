from celery import Celery, Task
import time
import numpy as np
import json

from db.database import get_db
from db.models import Experiment, Decision, Measurement, Data, Report

# from celery_workers import agents

from maestro_api import maestro_messages as mm

app = Celery('tasks', broker='redis://redis/0', backend='redis://redis/0')
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
        
        # self.pipeline = [
        #     agents.KMeansAgent(n_clusters=4, experiment_id=self.experiment_id), 
        #     agents.GPytorchAgent(input_bounds=[[low,high] for low, high in zip(self.motor_lows, self.motor_highs)], input_min_spacings=self.motor_min_steps, experiment_id=self.experiment_id)
        #     ]

    def run(self):
        # Create the first decision
        num_random_measurements = 10

        new_decision = Decision(
            experiment_id = self.experiment_id,
            method = f"Startup: {num_random_measurements} random measurements.")
        self.db.add(new_decision)
        self.db.commit()

        # Create the first measurements
        measurements = self.create_random_measurements(num_random_measurements, new_decision.decision_id)

        while True:
            # Check if experiment is still active
            self.experiment = self.db.query(Experiment).get(self.experiment_id)
            if not self.experiment.active:
                print("Experiment is no longer active. Shutting down...")
                return
            self.measured_positions = self.db.query(Measurement.positions).filter(Measurement.experiment_id == self.experiment_id).all()
            # print("*"*20, f"{len(measured_positions)=}")
            self.measured_positions = np.array([list(x[0].values()) for x in self.measured_positions])
            if len(self.measured_positions) > 0:
                self.measured_indices = position_to_index(self.measured_positions, self.motor_lows, self.motor_min_steps, self.max_positions_per_motor)
            #     print("*"*20, f"{measured_indices.shape=}")
                print(len(self.measured_indices))
                print(len(np.unique(self.measured_indices)))
                print(self.measured_indices)
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
            self.get_all_data()
            # time.sleep(1)
            # remaining_measurements = np.prod(self.max_positions_per_motor) - self.all_positions_measured.sum()
            # num = np.min([num, remaining_measurements])
            # # print(f"{remaining_measurements=}, {num=}")
            if len(self.measured_indices) == len(self.all_possible_measurement_indices):
                print("All measurements have been suggested?!?!?")
                continue
            num=5
            measurements = self.create_random_measurements(num, decision.decision_id)
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

    def get_all_data(self):
        print('Getting all data')
        start = time.perf_counter_ns()
        data = self.db.query(Data).filter(Data.experiment_id == self.experiment_id, Data.fieldname == 'Fixed_Spectra0').all()
        data = np.array([json.loads(x.data) for x in data])
        print(f'Got all data in {(time.perf_counter_ns() - start) / 1e9} seconds')
        return data

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

    # def create_smart_measurements(self, num_measurements, decision_id):
    #     decision = self.db.query(Decision).get(decision_id)
    #     experiment = self.db.query(Experiment).get(decision.experiment_id)

    #     x = self.measured_positions
    #     y = self.get_all_data()
    #     for agent in self.pipeline:
    #         x, y = agent.tell(x)
    #     measurements = []
    #     for position in agent.ask(num_measurements):
    #         pos_dict = {motor.device_name: position[i] for i, motor in enumerate(self.motors)}
    #         measurement = Measurement(experiment_id=experiment.experiment_id, decision_id=decision_id,
    #                                 positions=pos_dict, measured=False, measurement_time="", ai_cycle=None)
    #         measurements.append(measurement)
    #     return measurements 

@app.task
def setup_experiment_watcher(experiment_id):
    experiment_watcher = ExperimentWatcher(experiment_id)
    experiment_watcher.run()

def position_to_index(positions, motor_lows, motor_min_steps, max_positions_per_motor):
    indices = np.floor((positions - motor_lows) / motor_min_steps).astype(np.int_).T
    return np.ravel_multi_index(indices, max_positions_per_motor)

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
