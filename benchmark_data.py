import time
import os
import numpy as np
from sklearn.datasets import fetch_openml
import tensorflow as tf
import cvae.lib.data_reader_array as dra
import cvae.lib.data_reader_tf2 as dra2

# Load MNIST data
print("Loading MNIST data...")
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X = mnist.data.astype('float32')
y = mnist.target

# Parameters
BATCH_SIZE = 64
NUM_BATCHES = 100
LOGDIR = 'temp_benchmark'

# Create directory if it doesn't exist
os.makedirs(LOGDIR, exist_ok=True)

def benchmark_old_pipeline():
    print("Setting up old pipeline...")
    # Disable eager execution for TF1.x compatibility
    tf.compat.v1.disable_eager_execution()
    
    # Create coordinator and session
    coord = tf.train.Coordinator()
    sess = tf.compat.v1.Session()
    
    # Create data reader
    reader = dra.DataReader(X, feature_normalization=True, coord=coord, logdir=LOGDIR)
    
    # Initialize variables
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # Start queue runners
    threads = reader.start_threads(sess)
    
    # Benchmark batch fetching
    print("Running old pipeline benchmark...")
    start_time = time.time()
    for i in range(NUM_BATCHES):
        batch = sess.run(reader.dequeue_feature(BATCH_SIZE))
        if i % 10 == 0:
            print(f"Old pipeline: processed {i}/{NUM_BATCHES} batches")
    end_time = time.time()
    
    # Clean up
    coord.request_stop()
    coord.join(threads)
    sess.close()
    
    return end_time - start_time

def benchmark_new_pipeline():
    print("Setting up new pipeline...")
    # Enable eager execution for TF2.x
    tf.compat.v1.enable_eager_execution()
    
    # Create data loader
    loader = dra2.DataLoader(
        features=X,
        batch_size=BATCH_SIZE,
        feature_normalization=True,
        logdir=LOGDIR
    )
    
    # Get dataset
    dataset = loader.get_dataset()
    
    # Benchmark batch fetching
    print("Running new pipeline benchmark...")
    start_time = time.time()
    for i, batch in enumerate(dataset.take(NUM_BATCHES)):
        _ = batch.numpy()
        if i % 10 == 0:
            print(f"New pipeline: processed {i}/{NUM_BATCHES} batches")
    end_time = time.time()
    
    return end_time - start_time

if __name__ == "__main__":
    # Note: We can only run one benchmark at a time due to eager execution state
    print("\nBenchmarking new pipeline...")
    new_time = benchmark_new_pipeline()
    print(f"New pipeline time: {new_time:.3f}s")
    print(f"Batches per second: {NUM_BATCHES/new_time:.1f}")
    
    print("\nBenchmarking old pipeline...")
    old_time = benchmark_old_pipeline()
    print(f"Old pipeline time: {old_time:.3f}s")
    print(f"Batches per second: {NUM_BATCHES/old_time:.1f}")
    
    print(f"\nSpeedup: {old_time/new_time:.2f}x")
