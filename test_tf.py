import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

print("Starting TensorFlow import...")
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution enabled: {tf.executing_eagerly()}")
print("TensorFlow successfully imported!")
