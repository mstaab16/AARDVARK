import time
import numpy as np
import base64
import json
import logging
from scipy.stats import qmc
from scipy.interpolate import griddata

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
        self.data_names = []
        
        self.pipeline = [
            agents.DKLAgent(input_bounds=[[low,high] for low, high in zip(self.motor_lows, self.motor_highs)], input_min_spacings=self.motor_min_steps, experiment_id=self.experiment_id),
            # # agents.PCAAgent(n_components=50),
            # agents.UMAPAgent(n_components=3),
            # agents.KMeansAgent(n_clusters=6, experiment_id=self.experiment_id),
            # # agents.HDBSCANAgent(n_clusters=6), 
            # agents.GPytorchAgent(input_bounds=[[low,high] for low, high in zip(self.motor_lows, self.motor_highs)], input_min_spacings=self.motor_min_steps, experiment_id=self.experiment_id)
            ]
        
        self.use_raster_grid = False
        grid_shape = (60, 60)
        xs = np.linspace(self.motor_lows[0], self.motor_highs[0], grid_shape[0])
        ys = np.linspace(self.motor_lows[1], self.motor_highs[1], grid_shape[1])
        # self.grid_positions = []
        self.griddata_meshgrid = np.meshgrid(xs, ys)
        # for i, y in enumerate(ys):
        #     if i % 2 == 0:
        #         self.grid_positions.extend([[x, y] for x in xs])
        #     else:
        #         self.grid_positions.extend([[x, y] for x in xs[::-1]])
        xs = np.random.uniform(self.motor_lows[0], self.motor_highs[0], 1000)
        ys = np.random.uniform(self.motor_lows[1], self.motor_highs[1], 1000)
        self.grid_positions = np.array([xs, ys]).T.tolist()


    def run(self):
        # Create the first decision
        if self.use_raster_grid:
            num_to_suggest=25
            decision = Decision(
                experiment_id = self.experiment_id,
                method = f"Startup: Raster grid of {len(self.grid_positions)} measurements.")
            self.db.add(decision)
            self.db.commit()
            measurements = []
            for pos in self.grid_positions:
                pos_dict = {motor.device_name: pos[i] for i, motor in enumerate(self.motors)}
                measurement = Measurement(experiment_id=self.experiment.experiment_id, decision_id=decision.decision_id,
                                        positions=pos_dict, measured=False, measurement_time="", ai_cycle=None)
                measurements.append(measurement)
            self.db.add_all(measurements)
            self.db.commit()
            print(f"Set up raster grid of {len(measurements)} measurements.")
        else:
            num_to_suggest=25
            num_random_measurements = 100
            num_boundary_measurements = 3

            startup_decision = Decision(
                experiment_id = self.experiment_id,
                method = f"Startup: Corners and {num_random_measurements} random measurements.")
            self.db.add(startup_decision)
            self.db.commit()

            # corners = [[self.motor_lows[0], self.motor_lows[1]],
            #         [self.motor_highs[0], self.motor_lows[1]],
            #         [self.motor_lows[0], self.motor_highs[1]],
            #         [self.motor_highs[0], self.motor_highs[1]],
            #         ]
            # boundary_measurements = []
            # boundary_measurements.extend([[self.motor_lows[0], y] for y in np.linspace(self.motor_lows[1], self.motor_highs[1], num_boundary_measurements)])
            # boundary_measurements.extend([[x, self.motor_highs[1]] for x in np.linspace(self.motor_lows[0], self.motor_highs[0], num_boundary_measurements)])
            # boundary_measurements.extend([[self.motor_highs[0], y] for y in np.linspace(self.motor_lows[1], self.motor_highs[1], num_boundary_measurements)[::-1]])
            # boundary_measurements.extend([[x, self.motor_lows[1]] for x in np.linspace(self.motor_lows[0], self.motor_highs[0], num_boundary_measurements)[::-1]])
            # for pos in boundary_measurements:
            #     pos_dict = {motor.device_name: pos[i] for i, motor in enumerate(self.motors)}
            #     measurement = Measurement(experiment_id=self.experiment.experiment_id, decision_id=startup_decision.decision_id,
            #                             positions=pos_dict, measured=False, measurement_time="", ai_cycle=None)
            #     self.db.add(measurement)

            # Create the first measurements
            measurements = self.create_random_measurements(num_random_measurements, startup_decision.decision_id)
            num_random_measurements = len(measurements)
            self.db.add_all(measurements)
            self.db.commit()
            # time.sleep(5)

        while True:
            loop_start = time.perf_counter()
            # Reset loop_end each time because sometimes we will continue the loop early
            loop_end = time.perf_counter()
            # Check if experiment is still active
            experiment_active = self.db.query(Experiment.active).filter(Experiment.experiment_id == self.experiment_id).first()[0]
            if not experiment_active:
                print(f"Experiment is no longer active. Shutting down...")
                return
            new_measurements = self.update_data()
            if len(self.data_ids) < 0.75 * num_random_measurements: # len(self.data_ids) < len(boundary_measurements) + num_boundary_measurements * 0.9:
                print(f"Not enough data to train on {self.experiment_id} | active={experiment_active}")
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
            if not self.use_raster_grid:
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
            # measurements = self.create_random_measurements(num_to_suggest, decision.decision_id)
            measurements = self.create_smart_measurements(num_to_suggest, decision.decision_id)
            if len(self.data_ids) < num_random_measurements:
                continue
            # non_duplicates = []
            # for measurement in measurements:
            #     idx = position_to_index(np.array([list(measurement.positions.values())]), self.motor_lows, self.motor_min_steps, self.max_positions_per_motor)
            #     if idx not in measured_indices:
            #         non_duplicates.append(measurement)
            # measurements = self.create_smart_measurements(10, decision.decision_id)
            # Order the moves in a reasonable way
            if not self.use_raster_grid:
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
                num_to_suggest = int(max(num_to_suggest, 5))
                print(f"UPDATING NUM TO SUGGEST TO: {num_to_suggest}")


    def update_data(self):
        print("*"*20)
        print('Getting all data')
        start = time.perf_counter_ns()
        # query = self.db.query(Experiment.data_names_to_learn).filter(Experiment.experiment_id == self.experiment_id)
        
        query = self.db.query(Experiment.data_names_to_learn).filter(Experiment.experiment_id == self.experiment_id).first()[0]
        if query:
            query = json.loads(query)
            if not self.data_names:
                longest_name = max(query.keys(), key=len)
                self.data_names.append(longest_name)
        
        query = self.db.query(Measurement.positions, Data.data_id, Data.data, Data.data_info).join(Measurement.data)\
                .filter(~Data.data_id.in_(self.data_ids))\
                .filter(Measurement.experiment_id == self.experiment_id, Data.fieldname.in_(self.data_names)).order_by(Data.data_id).all()
        
        # data, data_info = db.query(Data.data, Data.data_info).filter(Data.experiment_id == experiment_id, Data.fieldname == "Fixed_Spectra5").order_by(Data.measurement_id.desc()).first()
        for pos, data_id, data, data_info in query:
            data_info = json.loads(data_info)
            self.data_ids.append(data_id)
            self.data_indices.append(position_to_index(np.array([list(pos.values())]), self.motor_lows, self.motor_min_steps, self.max_positions_per_motor))
            self.x.append(np.array(list(pos.values())))
            self.y.append(np.fromfile(data, dtype=np.int32).reshape(*data_info['dimensions'], order='F').flatten())

        print(f'Updated with {len(query)} new datasets in {(time.perf_counter_ns() - start) / 1e6:.02} ms.')
        return len(query)

    def create_random_measurements(self, num_measurements, decision_id):
        decision = self.db.query(Decision).get(decision_id)
        experiment = self.db.query(Experiment).get(decision.experiment_id)
        # options = np.setdiff1d(self.all_possible_measurement_indices, self.measured_indices)
        # random_indices = np.random.choice(options, np.min([num_measurements, len(options)]),replace=False)
        # positions = index_to_position(np.array(random_indices), self.motor_lows, self.motor_min_steps, self.max_positions_per_motor)
        range1 = (self.motor_highs[0] - self.motor_lows[0])
        range2 = (self.motor_highs[1] - self.motor_lows[1])
        # r1 = range1
        # r2 = range2
        engine = qmc.PoissonDisk(d=2, radius=0.75/np.sqrt(num_measurements))
        # positions = engine.random(n=num_measurements)
        positions = engine.fill_space()
        # positions = all_pos[np.random.choice(len(all_pos), num_measurements, replace=False)]
        positions = np.array(positions) * np.array([range1, range2]) + self.motor_lows

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
            if self.use_raster_grid and isinstance(agent, agents.KMeansAgent):
                agent.report(self.griddata_meshgrid)
        measurements = []
        # agent_output = agent.ask(num_measurements)
        # print(agent_output)
        try: 
            positions = agent.ask(num_measurements)
        except NotImplementedError as e:
            if self.use_raster_grid:
                positions = self.grid_positions
            else:
                raise e
            
        for position in positions:
            pos_dict = {motor.device_name: float(position[i]) for i, motor in enumerate(self.motors)}
            measurement = Measurement(experiment_id=experiment.experiment_id, decision_id=decision_id,
                                    positions=pos_dict, measured=False, measurement_time="", ai_cycle=None)
            measurements.append(measurement)
        return measurements 

# class Pipeline:
#     def __init__(self, name, experiment_id, pipeline):
#         self.name = name
#         self.pipeline = pipeline
#         self.db_gen = get_db()
#         self.db = next(self.db_gen)
#         self.experiment_id = experiment_id

#     def tell
    


@app.task
def setup_experiment_watcher(experiment_id):
    experiment_watcher = ExperimentWatcher(experiment_id)
    # experiment_watcher.register_pipeline()
    experiment_watcher.run()

@app.task
def scatter_plot_report(experiment_id, name, x, y, extra_data = None):
    x = np.asarray(x).tolist()
    y = np.asarray(y).tolist()
    data = {
        "x": x,
        "y": y,
    }
    if extra_data is not None:
        data.update(extra_data)
    save_report(experiment_id, name, data)

@app.task
def image_report(experiment_id, name, points, values, xi, method, extra_data = None):
    matrix = griddata(points, values, xi, method=method)

    matrix = matrix.tolist()
    data = {
        "image": matrix,
    }
    if extra_data is not None:
        data.update(extra_data)
    save_report(experiment_id, name, data)

@app.task
def save_report(experiment_id, name, data, **kwargs):
    db_gen = get_db()
    db = next(db_gen)
    try:
        report = Report(experiment_id=experiment_id, name=name, data=data, **kwargs)
    except TypeError as e:
        logging.log(logging.ERROR, f"Error: {e}")
        return
    db.add(report)
    db.commit()
    logging.log(logging.INFO, f"Saved report: {name}")

@app.task
def save_data(msg):
    print("Save data task initiated...")
    data = mm.MaestroLVDataMessage(**msg)
    db_gen = get_db()
    db = next(db_gen)
    experiment_id = db.query(Experiment).order_by(Experiment.experiment_id.desc()).first().experiment_id
    data_names = db.query(Experiment.data_names_to_learn).filter(Experiment.experiment_id == experiment_id).first()
    data_names = {} if data_names[0] is None else json.loads(data_names[0])

    current_measurement = db.query(Measurement).filter(Measurement.experiment_id == experiment_id, Measurement.measured == True).order_by(Measurement.measurement_id.desc()).first()

    current_measurement.ai_cycle = data.message.current_AI_cycle
    for fd in data.fits_descriptors:
        print(f"Recieved Data with field name: {fd.fieldname}")
        if fd.fieldname not in data_names:
            data_names[fd.fieldname] = False if fd.fieldname == 'time' or fd.fieldname == '' else True
            
        # if data.message.method == "fake_labview":
        #     print('Decoding data first...')
        #     dataset = base64.decodebytes(fd.Data)
        #     # print(dataset)
        # else:
        data_filepath = fd.Data
        # print("root path: /mnt/MAESTROdata/")
        root_path = '/mnt/MAESTROdata/'
        data_filepath = root_path + '/'.join(data_filepath.split('\\')[1:])
        # print(f"Data filepath: {data_filepath}")
        # print("SAVING DATA: ", fd.Data)
        new_data = Data(
            experiment_id = experiment_id,
            measurement_id = current_measurement.measurement_id,
            message = data.message.model_dump_json(),
            fieldname = fd.fieldname,
            data_cycle = data.message.current_data_cycle,
            data = data_filepath,
            data_info = json.dumps(dict(fd)),
        )
        db.add(new_data)
    db.query(Experiment).filter(Experiment.experiment_id == experiment_id).update({"data_names_to_learn": json.dumps(data_names)})
    db.commit()

@app.task
def plot_griddata(*,xs, ys, grid_points, scatter_xs, scatter_ys, bounds, filename):
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    matrix = griddata(xs, ys, (grid_points[0], grid_points[1]), method='nearest')
    plt.imshow(matrix, cmap='terrain', origin='lower', extent=np.ravel(bounds))
    cb = plt.colorbar() 
    plt.scatter(scatter_xs, scatter_ys, c='r', marker='x')
    plt.savefig(filename)
    plt.clf()


def position_to_index(positions, motor_lows, motor_min_steps, max_positions_per_motor):
    indices = np.floor((positions - motor_lows) / motor_min_steps).astype(np.int_).T
    return np.ravel_multi_index(indices, max_positions_per_motor, mode='clip')

def index_to_position(indices, motor_lows, motor_min_steps, max_positions_per_motor):
    indices = np.array(np.unravel_index(indices, max_positions_per_motor))
    positions = (indices * motor_min_steps[:, np.newaxis] + motor_lows[:, np.newaxis]).T
    return positions
