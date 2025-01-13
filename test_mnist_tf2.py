import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_openml
from cvae.cvae_tf2 import CompressionVAE
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
embedder = CompressionVAE(
    X=X,
    X_valid=None,
    dim_latent=2,
    cells_encoder=[512, 256],
    batch_size=128,
    batch_size_test=128,
    logdir='output_tf2',
    feature_normalization=True,
    tb_logging=True
)

# print("Training model...")
# try:
#     embedder.train(
#         learning_rate=1e-3,
#         num_steps=10000,
#         dropout_rate=0.25,
#         test_every=100,
#         checkpoint_every=1000
#     )
# except KeyboardInterrupt:
#     print("\nTraining interrupted. Using current model state.")

# Embed the data
print("\nEmbedding data...")
z = embedder.embed(X)

# Visualize the embedding
print("\nVisualizing embedding...")
embedder.visualize(
    z.numpy(),
    labels=[int(label) for label in y],
    filename='mnist_embedding_tf2.png'
)
plt.savefig('mnist_embedding_tf2.png')
print("Embedding visualization saved as 'mnist_embedding_tf2.png'")
