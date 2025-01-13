import os
import numpy as np
import tensorflow as tf
from time import time
import matplotlib.pyplot as plt
import pandas as pd
from memory_profiler import profile
import psutil
import gc
import shutil

# Import both implementations
from cvae.cvae import CompressionVAE as CVAE_TF1
from cvae.cvae_tf2 import CompressionVAE as CVAE_TF2

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def get_gpu_memory():
    """Get GPU memory usage if available."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return tf.config.experimental.get_memory_info('GPU:0')['peak']
        return 0
    except:
        return 0

def get_process_memory():
    """Get current process memory usage."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def calculate_reconstruction_error(model, x):
    """Calculate reconstruction error (MSE) on input data."""
    if isinstance(model, CVAE_TF1):
        # For TF1, create reconstruction op in the same graph and run it
        with model.graph.as_default():
            reconstruction = model.net.decode(model.embeddings)
            mse = tf.reduce_mean(tf.square(model.test_feature_placeholder - reconstruction))
            error = model.sess.run(mse, feed_dict={model.test_feature_placeholder: x})
        return error
    else:
        # For TF2, we can use eager execution
        z = model.embed(x)
        x_recon = model.decode(z)
        return np.mean((x - x_recon) ** 2)

def load_mnist():
    """Load MNIST dataset and normalize to [0, 1]."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten images
    x_train = x_train.reshape((len(x_train), -1))
    x_test = x_test.reshape((len(x_test), -1))
    
    return x_train, y_train, x_test, y_test

@profile
def benchmark_model(model_class, x_train, x_test, logdir, num_epochs=50):
    """Benchmark a single model implementation."""
    metrics = {}
    
    # Clean up output directory if it exists
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    
    # Clear any existing models and free memory
    gc.collect()
    if tf.config.list_physical_devices('GPU'):
        tf.keras.backend.clear_session()
    
    # Record initial memory
    metrics['initial_memory'] = get_process_memory()
    metrics['initial_gpu_memory'] = get_gpu_memory()
    
    # Model creation time and memory
    start_time = time()
    model = model_class(
        X=x_train,
        X_valid=x_test,
        dim_latent=2,
        cells_encoder=[512, 256],
        batch_size=128,
        batch_size_test=128,
        logdir=logdir,
        feature_normalization=True
    )
    metrics['creation_time'] = time() - start_time
    metrics['creation_memory'] = get_process_memory() - metrics['initial_memory']
    metrics['creation_gpu_memory'] = get_gpu_memory() - metrics['initial_gpu_memory']
    
    # Calculate steps based on epochs
    steps_per_epoch = len(x_train) // 128  # Using batch size of 128
    num_steps = steps_per_epoch * num_epochs
    
    # Training time and memory
    start_time = time()
    if isinstance(model, CVAE_TF2):
        model.train(
            learning_rate=1e-3,
            num_steps=num_steps,
            dropout_rate=0.25,
            test_every=50,
            lr_scheduling=True
        )
    else:
        model.train(
            learning_rate=1e-3,
            num_steps=num_steps,
            dropout_keep_prob=0.75,
            test_every=50,
            lr_scheduling=True
        )
    metrics['training_time'] = time() - start_time
    metrics['training_memory'] = get_process_memory() - metrics['initial_memory']
    metrics['training_gpu_memory'] = get_gpu_memory() - metrics['initial_gpu_memory']
    
    # Embedding time
    start_time = time()
    _ = model.embed(x_test)
    metrics['embedding_time'] = time() - start_time
    
    # Calculate reconstruction error
    metrics['reconstruction_error'] = calculate_reconstruction_error(model, x_test)
    
    return metrics

def plot_metrics(tf1_metrics, tf2_metrics, output_dir):
    """Plot comparison of metrics between TF1 and TF2 implementations."""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Time metrics
    time_metrics = ['creation_time', 'training_time', 'embedding_time']
    x = np.arange(len(time_metrics))
    width = 0.35
    
    axes[0,0].bar(x - width/2, [tf1_metrics[m] for m in time_metrics], width, label='TF1.x')
    axes[0,0].bar(x + width/2, [tf2_metrics[m] for m in time_metrics], width, label='TF2.x')
    axes[0,0].set_ylabel('Time (seconds)')
    axes[0,0].set_title('Time Performance')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(time_metrics, rotation=45)
    axes[0,0].legend()
    
    # Memory metrics
    memory_metrics = ['creation_memory', 'training_memory']
    x = np.arange(len(memory_metrics))
    
    axes[0,1].bar(x - width/2, [tf1_metrics[m]/1e6 for m in memory_metrics], width, label='TF1.x')
    axes[0,1].bar(x + width/2, [tf2_metrics[m]/1e6 for m in memory_metrics], width, label='TF2.x')
    axes[0,1].set_ylabel('Memory (MB)')
    axes[0,1].set_title('Memory Usage')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(memory_metrics, rotation=45)
    axes[0,1].legend()
    
    # GPU Memory metrics
    gpu_metrics = ['creation_gpu_memory', 'training_gpu_memory']
    x = np.arange(len(gpu_metrics))
    
    axes[1,0].bar(x - width/2, [tf1_metrics[m]/1e6 for m in gpu_metrics], width, label='TF1.x')
    axes[1,0].bar(x + width/2, [tf2_metrics[m]/1e6 for m in gpu_metrics], width, label='TF2.x')
    axes[1,0].set_ylabel('GPU Memory (MB)')
    axes[1,0].set_title('GPU Memory Usage')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(gpu_metrics, rotation=45)
    axes[1,0].legend()
    
    # Reconstruction error
    errors = [tf1_metrics['reconstruction_error'], tf2_metrics['reconstruction_error']]
    axes[1,1].bar(['TF1.x', 'TF2.x'], errors)
    axes[1,1].set_ylabel('MSE')
    axes[1,1].set_title('Reconstruction Error')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_comparison.png'))
    plt.close()
    
    # Save metrics to CSV
    df = pd.DataFrame({
        'Metric': list(tf1_metrics.keys()),
        'TF1.x': list(tf1_metrics.values()),
        'TF2.x': list(tf2_metrics.values())
    })
    df.to_csv(os.path.join(output_dir, 'benchmark_results.csv'), index=False)

def main():
    """Run benchmarking comparison between TF1 and TF2 implementations."""
    num_epochs = 4

    # Load MNIST dataset
    x_train, y_train, x_test, y_test = load_mnist()
    
    # Create output directory
    output_dir = 'benchmark_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Benchmark TF1 implementation
    print("\nBenchmarking TF1 implementation...")
    tf1_metrics = benchmark_model(CVAE_TF1, x_train, x_test, 'output_tf1', num_epochs=num_epochs)
    
    # Benchmark TF2 implementation
    print("\nBenchmarking TF2 implementation...")
    tf2_metrics = benchmark_model(CVAE_TF2, x_train, x_test, 'output_tf2', num_epochs=num_epochs)
    
    # Plot comparison
    plot_metrics(tf1_metrics, tf2_metrics, output_dir)
    
    # Print summary
    print("\nBenchmark Results:")
    print("-" * 50)
    metrics = ['creation_time', 'training_time', 'embedding_time']
    for metric in metrics:
        print(f"{metric}:")
        print(f"  TF1: {tf1_metrics[metric]:.2f}s")
        print(f"  TF2: {tf2_metrics[metric]:.2f}s")
        speedup = tf1_metrics[metric] / tf2_metrics[metric]
        print(f"  Speedup: {speedup:.2f}x")
        print()
    
if __name__ == '__main__':
    main()
