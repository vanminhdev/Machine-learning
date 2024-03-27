import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.python.client import device_lib

# print(device_lib.list_local_devices())

print(device_lib.list_local_devices())

#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
