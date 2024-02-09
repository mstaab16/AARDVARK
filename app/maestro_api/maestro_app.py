from fastapi import FastAPI, Depends, BackgroundTasks

from . import maestro_messages as mm
# import celery_workers.tasks as tasks
# from celery.execute import send_task
from celery_workers.celery_app import app as celery_app

from db.models import Experiment, Measurement, Data, Decision, Report

from db.database import get_db
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

import json
import asyncio
import numpy as np
import base64
import pickle
from skimage.transform import resize

maestro_app = FastAPI()


@maestro_app.get("/test")
def get_health_check():
    return mm.MaestroLVResponseOK()

@maestro_app.post("/test")
def post_health_check(message: mm.MaestroLVResponseOK):
    print('Message recieved:')
    print(message)
    return mm.MaestroLVResponseOK()

########################################################################################
#                                API FOR MAESTRO LABVIEW TO USE
########################################################################################

# read means server is reading the (init, data, etc.) message from the client
@maestro_app.post("/init")
def initialize(startup: mm.MaestroLVStartupMessage, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    # Set up the experiment
    new_experiment = Experiment(
        motors = {"motors" : [x.model_dump_json() for x in startup.AIModeparms]},
        data_filepath = "",
        active = True
        )
    # Add the experiment to the database
    db.add(new_experiment)
    db.commit()
    experiment_id = db.query(Experiment).order_by(Experiment.experiment_id.desc()).first().experiment_id

    print("Sending task to setup experiment watcher")
    celery_app.send_task("celery_workers.tasks.setup_experiment_watcher", args=(experiment_id,))
    print("Sent task to setup experiment watcher")

    # TODO: Need to return experiment id to LabView
    return mm.MaestroLVResponseOK()

def process_data(data: mm.MaestroLVDataMessage, db: Session):
    experiment_id = db.query(Experiment).order_by(Experiment.experiment_id.desc()).first().experiment_id
    # current_measurement = db.query(Measurement).filter(Measurement.experiment_id == experiment_id, Measurement.measured == False).order_by(Measurement.measurement_id.desc()).first()
    current_measurement = db.query(Measurement).filter(Measurement.experiment_id == experiment_id, Measurement.measured == True).order_by(Measurement.measurement_id.desc()).first()
    # update that measurement's ai_cycle from null to the current ai_cycle
    # if current_measurement.ai_cycle == None:
    #     print("Updating ai_cycle")
    #     # db.update()
    current_measurement.ai_cycle = data.message.current_AI_cycle
    for fd in data.fits_descriptors:
        fd_without_data = dict(fd)
        del fd_without_data["Data"]
        
        new_data = Data(
            experiment_id = experiment_id,
            measurement_id = current_measurement.measurement_id,
            message = data.message.model_dump_json(),
            fieldname = fd.fieldname,
            data_cycle = data.message.current_data_cycle,
            data = fd.Data,
            data_info = json.dumps(fd_without_data),
        )
        db.add(new_data)

        if fd.fieldname == 'Fixed_Spectrum0':
            image = np.frombuffer(base64.decodebytes(fd.Data), dtype=np.int32).reshape(128,128)
            thumbnail = resize(image, (64,64), anti_aliasing=True)
            # thumbnail /= np.max(thumbnail)
            # thumbnail *= 1e9
            # thumbnail = thumbnail.astype(np.int32)
            current_measurement.thumbnail = pickle.dumps(thumbnail)

    db.commit()
    # if this is the last data cycle, mark measurement as measured
    # if True: #data.message.current_data_cycle == 0:
    #     current_measurement.measured = True
    #     db.commit()

@maestro_app.post("/data")
def data(data: mm.MaestroLVDataMessage, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    # TODO: Need to make LabView send over the experiment id so we can have multiple experiments at onece
    # for now we will just use the most recent experiment (biggest id)
    # Send data to background process for processing and return
    background_tasks.add_task(process_data, data, db)
    return mm.MaestroLVResponseOK()

@maestro_app.post("/request_next_position")
async def next_position(pos_req: mm.MaestroLVPositionRequest, db: Session = Depends(get_db)) -> mm.MaestroLVPositionResponse:
    print("*"*10 + "TRYING TO GET A NEW POSITION" + "*"*10)
    # Get the next position from the database
    experiment = db.query(Experiment).order_by(Experiment.experiment_id.desc()).first()
    while True:
        next_measurement = \
            db.query(Measurement)\
            .filter(Measurement.experiment_id == experiment.experiment_id, Measurement.measured == False)\
            .order_by(Measurement.measurement_id).first()
        if next_measurement:
            next_measurement.measured = True
            try:
                db.commit()
                break
            except StaleDataError as e:
                db.rollback()
                print("!!!!!!!!!!!! Stale Data Error")
                continue
        await asyncio.sleep(0.1)

    pos_response: mm.MaestroLVPositionResponse = mm.MaestroLVPositionResponse(
                status="OK",
                positions=[mm.MotorPosition(axis_name=axis_name, value=next_measurement.positions[axis_name])
                            for axis_name in next_measurement.positions]
            )
    print("*"*10 + "RETURNING POSITION" + "*"*10)
    # print("Returning Position....")
    return pos_response

@maestro_app.post("/close")
def close(message: mm.MaestroLVCloseMessage, db: Session = Depends(get_db)):
    # Shutdown the experiment
    experiment = db.query(Experiment).order_by(Experiment.experiment_id.desc()).first()
    experiment.active = False
    db.commit()
    return mm.MaestroLVResponseOK()

@maestro_app.post("/abort")
def abort(message: mm.MaestroLVAbortMessage, db: Session = Depends(get_db)):
    # Shutdown the experiment
    experiment = db.query(Experiment).order_by(Experiment.experiment_id.desc()).first()
    experiment.active = False
    db.commit()
    return mm.MaestroLVResponseOK()

########################################################################################
#                                BACKEND FUNCTIONS
########################################################################################


# def background_add(x, y):
#     result = tasks.add.delay(x, y)
#     tasks.try_db.delay()
#     print(result.get(timeout=3))

# @maestro_app.post("/test")
# def test(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
#     background_tasks.add_task(celery.signature("compute_new_moves").delay)
#     return {"message": "Started background task"}