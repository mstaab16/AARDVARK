import numpy as np
import time

path_str = '/mnt/MAESTROdata/'
file_str = 'M:\\Eli-CAD-3\\test\\9342ae23f3aa1c78a94739f2a518aec1'

file = file_str.split('\\')
file = '/'.join(file[1:])

# print(file)

file_path = path_str + file
print(file_path)

data = np.fromfile(file_path, dtype=np.uint32)
print(data.shape[0]/1000)
