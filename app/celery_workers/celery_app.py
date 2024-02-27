from celery import Celery

app = Celery('tasks', 
             broker='redis://redis/0', 
             backend='redis://redis/0',
             accept_content=['pickle'])
app.conf.task_serializer = 'pickle'
app.conf.result_serializer = 'pickle'