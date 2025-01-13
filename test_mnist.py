import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from sklearn.datasets import fetch_openml
from cvae import cvae
import matplotlib.pyplot as plt

# Load MNIST data
print("Loading MNIST data...")
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X = mnist.data.astype('float32')
X = np.array(X)  # Ensure it's a numpy array
y = mnist.target

# Normalize the data
X = X / 255.0

# Create and train model
print("Creating CVAE model...")
embedder = cvae.CompressionVAE(X, dim_latent=2)

print("Training model...")
try:
    embedder.train()
except KeyboardInterrupt:
    print("\nTraining interrupted. Using current model state.")

# Embed the data
print("\nEmbedding data...")
z = embedder.embed(X)

# Visualize the embedding
print("\nVisualizing embedding...")
embedder.visualize(z, labels=[int(label) for label in y], filename='mnist_embedding.png')
plt.savefig('mnist_embedding.png')
print("Embedding visualization saved as 'mnist_embedding.png'")
