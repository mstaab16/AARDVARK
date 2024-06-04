'''
Allow or deny clients based on IP address.

Strawhouse, which is plain text with filtering on IP addresses. It still
uses the NULL mechanism, but we install an authentication hook that checks
the IP address against a list and allows or denies it
accordingly.

Author: Chris Laws
'''

# adding whitelist security to the server following this zmq guide:
# https://github.com/zeromq/pyzmq/blob/v25.1.2/examples/security/strawhouse.py

import logging
import sys
import time
import json
import base64

import zmq
import zmq.auth
from zmq.auth.thread import ThreadAuthenticator
from sqlalchemy.orm.exc import StaleDataError

import maestro_api.maestro_messages as mm
from db.models import Experiment, Measurement, Data, Decision, Report
from db.database import get_db, engine
from db.base import Base
from celery_workers.celery_app import app as celery_app

def create_tables():
    Base.metadata.create_all(bind=engine)

def drop_tables():
    Base.metadata.drop_all(bind=engine)

# EINSTEIN IPs
whitelist = ['131.243.73.200', '172.17.0.1', '172.25.0.1']

#ELI IPs:
whitelist.extend(['131.243.88.203'])

#Backup Computer IP:
whitelist.extend(['131.243.73.174'])

class Server:
    def __init__(self):
        create_tables()
        self.startup()
        self.operate()
        self.shutdown()

    def startup(self):
        logging.info("Getting database connection")
        self.db_gen = get_db()
        self.db = next(self.db_gen)
        self.ctx = zmq.Context.instance()
        self.auth = ThreadAuthenticator(self.ctx)
        logging.info("Starting strawhouse server")
        self.auth.start()
        logging.info(f"Allowing clients with IP addresses: {whitelist}")
        self.auth.allow(*whitelist)
        self.server = self.ctx.socket(zmq.REP)
        self.server.zap_domain = b'global'
        logging.info("Binding to tcp://*:5550")
        self.server.bind('tcp://*:5550')

    def shutdown(self):
        self.server.close()
        self.auth.stop()
        self.ctx.term()

    def operate(self):
        while True:
            msg = self.server.recv_json()
            # logging.info(f"Received message: {msg}")
            result = self.handle_request(msg)
            self.server.send(result.model_dump_json(by_alias=True).encode('utf-8'))

    def handle_request(self, msg):
        result = None
        start = time.perf_counter_ns()
        method = msg.get('method')
        if method is None:
            method = msg.get('message').get('method')

        if method is None:
            logging.error(f"Unknown message: {msg}")
            
        logging.info(f"Handling request with method: {method}")

        if method == "initialize":
            result = self.handle_initialize(msg)

        elif method == "sending newdata":
            result = self.handle_data(msg)
        
        elif method == "close":
            result = self.handle_close(msg)

        elif method == "move":
            result = self.handle_position(msg)
        
        elif method == "closing":
            result = self.handle_abort(msg)
        
        elif method == "abort":
            result = self.handle_abort(msg)

        else:
            logging.error(f"Unknown method: {method}")

        logging.info(f"Returning result for {method} request after {(time.perf_counter_ns() - start)/1e6:.03f}ms")
        if result is None:
            logging.error(f"Result for {msg} request is None")
        return result


    # ----------------- Handlers -----------------
    def handle_initialize(self, startup: dict):
        startup =  mm.MaestroLVStartupMessage(**startup)
        # Set up the experiment
        new_experiment = Experiment(
            motors = {"motors" : [x.model_dump_json() for x in startup.AIModeparms]},
            data_filepath = "",
            active = True
            )
        # Add the experiment to the database
        self.db.add(new_experiment)
        self.db.commit()
        experiment_id = self.db.query(Experiment).order_by(Experiment.experiment_id.desc()).first().experiment_id

        logging.info("Sending task to setup experiment watcher")
        celery_app.send_task("celery_workers.tasks.setup_experiment_watcher", args=(experiment_id,))
        logging.info("Sent task to setup experiment watcher")
        # TODO: Need to return experiment id to LabView
        return mm.MaestroLVResponseOK()
    
    def handle_data(self, msg):
        logging.info("Handling data request")
        # logging.info(msg)
        celery_app.send_task("celery_workers.tasks.save_data", args=(msg,))
        return mm.MaestroLVResponseOK()

    def handle_position(self, msg):
        msg = mm.MaestroLVPositionRequest(**msg)
        start = time.perf_counter()
        logging.info("*"*10 + "TRYING TO GET A NEW POSITION" + "*"*10)
        # Get the next position from the database
        experiment = self.db.query(Experiment).order_by(Experiment.experiment_id.desc()).first()
        while True:
            next_measurement = \
                self.db.query(Measurement)\
                .filter(Measurement.experiment_id == experiment.experiment_id, Measurement.measured == False)\
                .order_by(Measurement.measurement_id).first()
            if next_measurement:
                next_measurement.measured = True
                try:
                    self.db.commit()
                    break
                except StaleDataError as e:
                    self.db.rollback()
                    logging.info("!!!!!!!!!!!! Stale Data Error")
                    continue
            time.sleep(0.1)

        pos_response: mm.MaestroLVPositionResponse = mm.MaestroLVPositionResponse(
                    status="OK",
                    positions=[mm.MotorPosition(axis_name=axis_name, value=next_measurement.positions[axis_name])
                                for axis_name in next_measurement.positions]
                )
        logging.info("*"*10 + f"RETURNING POSITION AFTER {(time.perf_counter() - start)*1000:.02f}ms" + "*"*10)
        # logging.info("Returning Position....")
        return pos_response

    def handle_close(self, msg):
        msg = mm.MaestroLVCloseMessage(**msg)
        # Shutdown the experiment
        experiment = self.db.query(Experiment).order_by(Experiment.experiment_id.desc()).first()
        experiment.active = False
        self.db.commit()
        logging.info("+"*10 + "SHOULD BE CLOSING NOW" + "+"*10)
        return mm.MaestroLVResponseOK()

    def handle_abort(self, msg):
        msg = mm.MaestroLVAbortMessage(**msg)
        # Shutdown the experiment
        experiment = self.db.query(Experiment).order_by(Experiment.experiment_id.desc()).first()
        logging.info("+"*10 + f"Setting experiment {experiment.experiment_id} to {False}" + "+"*10)
        experiment.active = False
        self.db.commit()
        experiment = self.db.query(Experiment).order_by(Experiment.experiment_id.desc()).first()
        logging.info("+"*10 + f"Experiment {experiment.experiment_id} activity is now {experiment.active}" + "+"*10)
        logging.info("+"*10 + "SHOULD BE ABORTING NOW" + "+"*10)
        return mm.MaestroLVResponseOK()



# def run() -> None:
#     '''Run strawhouse client'''

#     allow_test_pass = False
#     deny_test_pass = False

#     ctx = zmq.Context.instance()

#     # Start an authenticator for this context.
#     auth = ThreadAuthenticator(ctx)
#     auth.start()

#     # Part 1 - demonstrate allowing clients based on IP address
#     auth.deny('127.0.0.1')

#     server = ctx.socket(zmq.REP)
#     server.zap_domain = b'global'  # must come before bind
#     server.bind('tcp://*:9000')

#     client_allow = ctx.socket(zmq.REQ)
#     client_allow.connect('tcp://127.0.0.1:9000')

#     client_allow.send(b"Hello")

#     msg = server.recv()
#     logging.info(msg)
#     client_allow.close()

#     # Part 2 - demonstrate denying clients based on IP address
#     auth.stop()

#     # auth = ThreadAuthenticator(ctx)
#     # auth.start()

#     # auth.deny('127.0.0.1')

#     # client_deny = ctx.socket(zmq.REQ)
#     # client_deny.connect('tcp://127.0.0.1:9000')

#     # if server.poll(50, zmq.POLLOUT):
#     #     server.send(b"Hello")

#     #     if client_deny.poll(50):
#     #         msg = client_deny.recv()
#     #     else:
#     #         deny_test_pass = True
#     # else:
#     #     deny_test_pass = True

#     # client_deny.close()

#     # auth.stop()  # stop auth thread

#     # if allow_test_pass and deny_test_pass:
#     #     logging.info("Strawhouse test OK")
#     # else:
#     #     logging.error("Strawhouse test FAIL")

def main():
    server = Server()

if __name__ == '__main__':
    if zmq.zmq_version_info() < (4, 0):
        raise RuntimeError(
            "Security is not supported in libzmq version < 4.0. libzmq version {}".format(
                zmq.zmq_version()
            )
        )

    if '-v' in sys.argv:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

    server = Server()
    # del server
