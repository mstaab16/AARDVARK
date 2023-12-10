import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5555')

print("Waiting for message...")
msg = socket.recv()
print(msg)
socket.send(b'OK')
