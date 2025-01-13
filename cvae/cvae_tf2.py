import os
import random
import json
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# Ensure TF2 behavior only for this module
if hasattr(tf, 'executing_eagerly') and not tf.executing_eagerly():
    print("Enabling eager execution for TF2 implementation")
    tf.compat.v1.enable_eager_execution()

from cvae.lib.data_reader_tf2 import DataLoader
import cvae.lib.model_iaf_tf2 as model_tf2


class CompressionVAE:
    """Variational Autoencoder (VAE) for vector compression/dimensionality reduction.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Training data for the VAE.
        Alternatively, X can be the path to a root-directory containing npy files (potentially nested), each
        representing a single feature vector. This allows for handling of datasets that are too large to fit
        in memory.
        Can be None (default) only if a model with this name has previously been trained. Otherwise None will
        raise an exception.

    X_valid : array, shape (n__valid_samples, n_features), optional (default: None)
        Validation data. If not provided, X is split into training and validation data

    train_valid_split : float, optional (default: 0.9)
        Specifies in what ratio to split X into training and validation data (after randomizing the data).
        Ignored if X_valid provided.

    dim_latent : int, optional (default: 2)
        Dimension of latent space (i.e. number of features of embeddings)

    iaf_flow_length : int, optional (default: 5)
        Number of IAF Flow layers to use in the model.
        For details, see https://arxiv.org/abs/1606.04934.

    cells_encoder : list of int, optional (default: None)
        The length of this list determines the number of layers of the encoder and decoder, and the values
        determine the number of units per layer (reversed order for decoder).
        If None, this is automatically chosen based on number of features and latent dimension.

    initializer : string, optional (default: 'orthogonal')
        Initializer to use for weights of model.

    batch_size : int, optional (default: 64)
        Batch size to use for training.

    batch_size_test : int, optional (default: 64)
        Batch size to use for testing.

    logdir : string, optional (default: 'temp')
        Location for where to save the model and other related files. Can also be used to restart from an already
        trained model.
        If 'temp' (default), any previously stored data is deleted and model/data are initialised from scratch.

    feature_normalization : bool, optional (default: True)
        If True (default), normalization of all data is applied internally, based on training data statistics.

    tb_logging : bool, optional (default: False)
        If True, create tensorboard summaries with loss data etc.
    """
    def __init__(self,
                 X=None,
                 X_valid=None,
                 train_valid_split=0.9,
                 dim_latent=2,
                 iaf_flow_length=5,
                 cells_encoder=None,
                 initializer='orthogonal',
                 batch_size=64,
                 batch_size_test=64,
                 logdir='temp',
                 feature_normalization=True,
                 tb_logging=False):
        
        self.dim_latent = dim_latent
        self.iaf_flow_length = iaf_flow_length
        self.cells_encoder = cells_encoder
        self.initializer = initializer
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.logdir = os.path.abspath(logdir)
        self.feature_normalization = feature_normalization
        self.tb_logging = tb_logging
        
        # --- Check for existing model ---
        self.is_trained = False
        
        if logdir == 'temp' and os.path.exists(self.logdir):
            shutil.rmtree(self.logdir)
        
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
            
        # Check for existing files
        self.has_checkpoint = os.path.exists(os.path.join(self.logdir, 'checkpoint'))
        self.has_params = os.path.exists(os.path.join(self.logdir, 'params.json'))
        
        # --- Prepare data ---
        if isinstance(X, str):
            # Directory path provided
            self.train_data = DataLoader(
                X, 
                batch_size=batch_size,
                feature_normalization=feature_normalization,
                shuffle=True,
                logdir=self.logdir
            )
            self.valid_data = DataLoader(
                X,  # Same directory for validation
                batch_size=batch_size_test,
                feature_normalization=feature_normalization,
                shuffle=False,
                logdir=self.logdir,
                validation=True,
                train_ratio=train_valid_split
            )
            self.dim_feature = self.train_data.feature_dim
            
        elif isinstance(X, np.ndarray):
            # Split data if validation not provided
            if X_valid is None:
                num_data = len(X)
                indices = list(range(num_data))
                random.shuffle(indices)
                split_index = int(train_valid_split * num_data)
                X_train = X[indices[:split_index]]
                X_valid = X[indices[split_index:]]
            else:
                X_train = X
                
            self.dim_feature = X.shape[1]
            
            # Create data loaders
            self.train_data = DataLoader(
                X_train,
                batch_size=batch_size,
                feature_normalization=feature_normalization,
                shuffle=True,
                logdir=self.logdir
            )
            self.valid_data = DataLoader(
                X_valid,
                batch_size=batch_size_test,
                feature_normalization=feature_normalization,
                shuffle=False,
                logdir=self.logdir
            )
            
        elif X is None and self.has_checkpoint:
            # Load existing model
            with open(os.path.join(self.logdir, 'params.json'), 'r') as f:
                self.param = json.load(f)
            self.dim_feature = self.param['dim_feature']
        else:
            raise ValueError('X must be either numpy array, directory path, or None for existing model')
            
        # --- Set up model parameters ---
        if self.has_params:
            print(f'Loading parameters from {self.logdir}')
            with open(os.path.join(self.logdir, 'params.json'), 'r') as f:
                self.param = json.load(f)
        else:
            # Determine model structure if not provided
            if cells_encoder is None:
                smallest_power = int(2 ** (self.dim_latent - 1).bit_length())
                largest_power = int(2 ** self.dim_feature.bit_length() / 2)
                powers_of_two = [smallest_power]
                while powers_of_two[-1] <= largest_power:
                    powers_of_two.append(powers_of_two[-1]*2)
                
                l2_index = int(len(powers_of_two) / 2)
                try:
                    model_layers = [
                        largest_power,
                        powers_of_two[l2_index+1]
                    ]
                except:
                    model_layers = [
                        largest_power,
                        int(largest_power/2)
                    ]
            else:
                model_layers = cells_encoder
                
            cells_hidden = min(model_layers[-1], 64)
            
            self.param = {
                "dim_latent": self.dim_latent,
                "dim_feature": self.dim_feature,
                "cells_encoder": model_layers,
                "cells_hidden": cells_hidden,
                "iaf_flow_length": self.iaf_flow_length,
                "dim_autoregressive_nl": cells_hidden,
                "feature_normalization": self.feature_normalization
            }
            
            with open(os.path.join(self.logdir, 'params.json'), 'w') as f:
                json.dump(self.param, f, indent=2)
                
        # --- Create model ---
        print('Creating model...')
        self.model = model_tf2.VAEModel(
            self.param,
            self.batch_size,
            self.dim_feature,
            dropout_rate=0.0
        )
        
        # Build model by running a forward pass with dummy data
        dummy_batch = tf.zeros((self.batch_size, self.dim_feature))
        _ = self.model(dummy_batch, training=False)
        
        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-4)
        
        # Load weights if checkpoint exists
        if self.has_checkpoint:
            latest = tf.train.latest_checkpoint(self.logdir)
            if latest is not None:
                print(f'Loading weights from {latest}')
                checkpoint = tf.train.Checkpoint(
                    optimizer=self.optimizer,
                    model=self.model
                )
                status = checkpoint.restore(latest)
                status.expect_partial()  # Suppress warnings about optimizer state
                self.is_trained = True
            else:
                print('No checkpoint found')
                
        if self.tb_logging:
            self.summary_writer = tf.summary.create_file_writer(self.logdir)
            
    @tf.function
    def train_step(self, x, training=True):
        """Single training step."""
        with tf.GradientTape() as tape:
            _ = self.model(x, training=training)
            loss = tf.reduce_sum(self.model.losses)
            
        if training:
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            self.is_trained = True  # Set model as trained after first training step
            
        return loss
        
    def train(self,
              learning_rate=1e-3,
              num_steps=int(5e4),
              dropout_rate=0.25,
              test_every=50,
              lr_scheduling=True,
              lr_scheduling_steps=5,
              lr_scheduling_factor=5,
              lr_scheduling_min=1e-5,
              checkpoint_every=2000):
        """Train the model."""
        
        # Set up checkpointing
        checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.model
        )
        manager = tf.train.CheckpointManager(
            checkpoint, 
            self.logdir,
            max_to_keep=3
        )
        
        # Training loop setup
        step = 0
        min_test_loss = float('inf')
        lr_patience = 0
        current_lr = learning_rate
        self.optimizer.learning_rate.assign(current_lr)
        
        # Calculate steps per epoch
        steps_per_epoch = len(self.train_data.dataset)
            
        # Training loop
        while step < num_steps:
            for x_batch in self.train_data.dataset:
                start_time = time.time()
                
                loss = self.train_step(x_batch, training=True)
                current_epoch = step / steps_per_epoch
                
                if step % test_every == 0:
                    # Compute validation loss
                    test_losses = []
                    for x_test in self.valid_data.dataset:
                        test_loss = self.train_step(x_test, training=False)
                        test_losses.append(test_loss)
                    avg_test_loss = tf.reduce_mean(test_losses)
                    
                    duration = (time.time() - start_time) / test_every
                    print(f'step {step}; epoch {current_epoch:.2f} - loss = {loss:.3f}, test_loss = {avg_test_loss:.3f}, lr = {current_lr:.5f}, ({duration:.3f} sec/step)')
                    
                    if self.tb_logging:
                        with self.summary_writer.as_default():
                            tf.summary.scalar('train_loss', loss, step=step)
                            tf.summary.scalar('test_loss', avg_test_loss, step=step)
                    
                    # Learning rate scheduling
                    if lr_scheduling:
                        if avg_test_loss < min_test_loss:
                            min_test_loss = avg_test_loss
                            lr_patience = 0
                        else:
                            lr_patience += 1
                            
                        if lr_patience >= lr_scheduling_steps:
                            current_lr = current_lr / lr_scheduling_factor
                            if current_lr < lr_scheduling_min:
                                print(f'Learning rate {current_lr:.2e} below minimum. Stopping training.')
                                return
                            print(f'Decreasing learning rate to {current_lr:.2e}')
                            self.optimizer.learning_rate.assign(current_lr)
                            lr_patience = 0
                
                if step % checkpoint_every == 0:
                    save_path = manager.save()
                    print(f'Saved checkpoint at step {step}: {save_path}')
                
                step += 1
                if step >= num_steps:
                    break
                    
        # Final checkpoint
        manager.save()
        self.is_trained = True
        
    def embed(self, X, batch_size=None):
        """Embed data into latent space."""
        if not self.is_trained:
            raise RuntimeError('Model must be trained before embedding')
            
        if batch_size is None:
            batch_size = self.batch_size_test
            
        # Create dataset
        dataset = DataLoader(
            X,
            batch_size=batch_size,
            feature_normalization=self.feature_normalization,
            shuffle=False,
            logdir=self.logdir
        ).dataset
        
        # Get embeddings
        embeddings = []
        for x_batch in dataset:
            z, _, _ = self.model.encode(x_batch, training=False)
            embeddings.append(z)
            
        return tf.concat(embeddings, axis=0)
    
    def decode(self, z):
        """Decode from latent space."""
        if not self.is_trained:
            raise RuntimeError('Model must be trained before decoding')
            
        # Convert to tensor if numpy array
        if isinstance(z, np.ndarray):
            z = tf.convert_to_tensor(z, dtype=tf.float32)
            
        return self.model.decode(z, training=False)
    
    def visualize(self, z, labels=None, categories=None, filename=None):
        """Visualize 2D latent space."""
        assert z.shape[1] == 2, "Visualization only available for 2D embeddings."

        fig, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor='w', edgecolor='k')
        if labels is None:
            s = ax.scatter(z[:, 0], z[:, 1], s=7)
        else:
            # Check if labels are provided as indices or strings
            if type(labels[0]) == int:
                pass
            elif type(labels[0]) == str:
                # Find unique categories and convert string labels to indices
                categories = list(set(labels))
                str_to_int = {cat: k for k, cat in enumerate(categories)}
                labels = [str_to_int[label] for label in labels]
            else:
                raise Exception('Label needs to be list of integer or string labels.')

            cmap = plt.get_cmap('jet', np.max(labels) - np.min(labels) + 1)
            s = ax.scatter(z[:, 0], z[:, 1], s=7, c=labels, cmap=cmap, vmin=np.min(labels) - .5,
                           vmax=np.max(labels) + .5)
            cax = plt.colorbar(s, ticks=np.arange(np.min(labels), np.max(labels) + 1))
            if categories is not None:
                cax.ax.set_yticklabels(categories)

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
            
    def visualize_latent_grid(self,
                            xy_range=(-4.0, 4.0),
                            grid_size=10,
                            shape=(28, 28),
                            clip=(0, 255),
                            figsize=(12, 12),
                            filename=None):
        """Visualize latent space by decoding a grid of points."""
        if self.dim_latent != 2:
            raise ValueError('Can only visualize 2D latent space')
            
        # Create grid
        x = np.linspace(xy_range[0], xy_range[1], grid_size)
        y = np.linspace(xy_range[0], xy_range[1], grid_size)
        xx, yy = np.meshgrid(x, y)
        z_grid = np.column_stack((xx.ravel(), yy.ravel()))
        
        # Decode grid points
        x_decoded = self.decode(z_grid)
        
        # Reshape and arrange into image grid
        x_decoded = np.reshape(x_decoded, (grid_size, grid_size, *shape))
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        for i in range(grid_size):
            for j in range(grid_size):
                ax = plt.subplot(grid_size, grid_size, i * grid_size + j + 1)
                img = x_decoded[i, j]
                if clip:
                    img = np.clip(img, clip[0], clip[1])
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
