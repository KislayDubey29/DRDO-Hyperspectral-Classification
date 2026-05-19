import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

print("\nGPU Devices:")
print(tf.config.list_physical_devices('GPU'))