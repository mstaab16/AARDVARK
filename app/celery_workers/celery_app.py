from celery import Celery

app = Celery('tasks', broker='redis://redis/0', backend='redis://redis/0')