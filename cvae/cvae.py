import sys
import os
import random
import json
import time
import shutil

import numpy as np
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass

# import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cvae.lib.data_reader_array as dra
import cvae.lib.data_reader as dr
import cvae.lib.model_iaf as model
import cvae.lib.functions as fun


# Save model to checkpoint
def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


# Load model from checkpoint
def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


class CompressionVAE(object):
    """
    Variational Autoencoder (VAE) for vector compression/dimensionality reduction.

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

        self.trained_once_this_session = False

        # --- Check for existing model ---

        # Set flag to indicate that the model has not been trained yet
        self.is_trained = False

        # If using temporary model directory (default), delete any previously stored models
        if logdir == 'temp' and os.path.exists(self.logdir):
            shutil.rmtree(self.logdir)

        # Check if a model with the same name already exists
        # If no, create directory
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        # Do checkpoint, parameter, and norm files exist?
        self.has_checkpoint = os.path.exists(f'{self.logdir}/checkpoint')
        self.has_params = os.path.exists(f'{self.logdir}/params.json')
        self.has_norm = os.path.exists(f'{self.logdir}/norm.pkl')
        self.has_dataset_file = os.path.exists(f'{self.logdir}/data_train.pkl')
        self.has_data = False

        # --- Prepare data ---

        # Check if data is provided as array or as directory.
        self.dataset_type = None

        if type(X) == str:
            self.dataset_type = 'string'

            if self.has_dataset_file:
                print('This model has already been associated with a dataset from a directory. To create a new '
                      'dataset, delete the data_train.pkl and data_valid.pkl files in the model directory.')
                _, _, self.dim_feature = dr.load_dataset_file(f'{self.logdir}/data_train.pkl')
            else:
                print(f'Preparing train and validation datasets from feature directory {X}.')
                self.dim_feature = fun.prepare_dataset(data_dir=os.path.abspath(X),
                                                       logdir=self.logdir,
                                                       train_ratio=train_valid_split)
                self.has_dataset_file = True

            self.X = None
            self.X_valid = None
            self.has_data = True

        elif type(X) == np.ndarray:
            self.dataset_type = 'array'
            self.dim_feature = X.shape[1]
            # Split data into train and validation or use provided validation data
            if X_valid is not None:
                assert X_valid.shape[1] == self.dim_feature, "Train and validation data has different feature dimensions!"
                self.X = X.astype(np.float32)
                self.X_valid = X_valid.astype(np.float32)
            else:
                # Randomize data
                num_data = len(X)
                indices = list(range(num_data))
                random.shuffle(indices)
                # Split data (and ensure it's float)
                split_index = int(train_valid_split * num_data)
                train_indices = indices[:split_index]
                valid_indices = indices[split_index:]
                self.X = X[train_indices].astype(np.float32)
                self.X_valid = X[valid_indices].astype(np.float32)
            self.has_data = True

        # elif X is None:
        #     self.X = None
        #     self.X_valid = None
        #     if self.has_dataset_file:
        #         print(f'Reloading dataset file {self.logdir}/data_train.pkl from previous instance of this model.')
        #         _, _, self.dim_feature = dr.load_dataset_file(f'{self.logdir}/data_train.pkl')
        #         self.has_data = True
        #     else:
        #         if self.has_checkpoint:
        #             self.has_data = False
        #         else:
        #             raise Exception(
        #                 'Model needs to be initialised with X provided as numpy array or path to directory.')
        else:
            raise Exception('Unsupported input type for X. Needs to be numpy array or string with path to directory '
                            'containing npy files. ')

        # --- Prepare parameter file ---

        # If parameter file for this model already exists, load it. Otherwise create one.
        if self.has_params:
            print(f'Existing parameter file found for model {self.logdir}.\n'
                  f'Loading stored parameters. Some input parameters might be ignored.')
            with open(f'{self.logdir}/params.json', 'r') as f:
                self.param = json.load(f)
            # Set dim_feature if not previously known
            if X is None and not self.has_dataset_file:
                self.dim_feature = self.param['dim_feature']
        else:
            # If not given, determine model structure
            # NOTE: The reasoning here is a bit arbitrary/dodgy, should probably put some more thought into this
            # and improve it.
            # TODO: This does not give very good results yet...
            if cells_encoder is None:
                # Get all the powers of two between latent dim and feature dim
                smallest_power = int(2 ** (self.dim_latent - 1).bit_length())
                largest_power = int(2 ** self.dim_feature.bit_length() / 2)
                powers_of_two = [smallest_power]
                while powers_of_two[-1] <= largest_power:
                    powers_of_two.append(powers_of_two[-1]*2)

                # By default, use two layers, one with largest power of two, the second roughly half-way between
                # input and output dimension
                l2_index = int(len(powers_of_two) / 2)
                try:
                    model_layers = [largest_power,
                                    powers_of_two[l2_index+1]]
                except:
                    model_layers = [largest_power,
                                    int(largest_power/2)]

            else:
                model_layers = cells_encoder

            # Number of hidden cells is smaller of the last layer size or 64
            cells_hidden = min(model_layers[-1], 64)

            if self.dataset_type == 'string':
                dataset_file = f'{self.logdir}/data_train.pkl'
                dataset_file_valid = f'{self.logdir}/data_valid.pkl'
            else:
                dataset_file = None
                dataset_file_valid = None

            self.param = {
                "dataset_file": dataset_file,
                "dataset_file_valid": dataset_file_valid,
                "dim_latent": self.dim_latent,
                "dim_feature": self.dim_feature,
                "cells_encoder": model_layers,
                "cells_hidden": cells_hidden,
                "iaf_flow_length": self.iaf_flow_length,
                "dim_autoregressive_nl": cells_hidden,
                "initial_s_offset": 1.0,
                "feature_normalization": self.feature_normalization
            }

            # Write to json for future re-use of this model
            with open(f'{self.logdir}/params.json', 'w') as outfile:
                json.dump(self.param, outfile, indent=2)

        # --- Set up VAE model ---
        self.graph = tf.Graph()

        with self.graph.as_default():

            # Create coordinator.
            self.coord = tf.train.Coordinator()

            # Set up batchers.
            with tf.name_scope('create_inputs'):
                if self.dataset_type == 'string':
                    self.reader = dr.DataReader(self.param['dataset_file'],
                                                self.param,
                                                f'{self.logdir}/params.json',
                                                self.coord,
                                                self.logdir)
                    self.test_batcher = dr.Batcher(self.param['dataset_file_valid'],
                                                   self.param,
                                                   f'{self.logdir}/params.json',
                                                   self.logdir)
                else:
                    self.reader = dra.DataReader(self.X, self.feature_normalization, self.coord, self.logdir)
                    self.test_batcher = dra.Batcher(self.X_valid, self.feature_normalization, self.logdir)
                self.train_batch = self.reader.dequeue_feature(self.batch_size)

            # Get normalisation data
            if self.feature_normalization:
                self.mean = self.test_batcher.mean
                self.norm = self.test_batcher.norm

            num_test_data = self.test_batcher.num_data
            self.test_batches_full = int(self.test_batcher.num_data / self.batch_size_test)
            self.test_batch_last = num_test_data - (self.test_batches_full * self.batch_size_test)

            # Placeholder for test features
            self.test_feature_placeholder = tf.placeholder_with_default(
                input=tf.zeros([self.batch_size, self.dim_feature], dtype=tf.float32),
                shape=[None, self.dim_feature])

            # Placeholder for dropout
            self.dropout_placeholder = tf.placeholder_with_default(input=tf.cast(1.0, dtype=tf.float32), shape=(),
                                                                   name="KeepProb")

            # Placeholder for learning rate
            self.lr_placeholder = tf.placeholder_with_default(input=tf.cast(1e-4, dtype=tf.float32), shape=(),
                                                                  name="LearningRate")

            print('Creating model.')
            self.net = model.VAEModel(self.param,
                                      self.batch_size,
                                      input_dim=self.dim_feature,
                                      keep_prob=self.dropout_placeholder,
                                      initializer=self.initializer)
            print('Model created.')

            self.embeddings = self.net.embed(self.test_feature_placeholder)

            print('Setting up loss.')
            self.loss = self.net.loss(self.train_batch)
            self.loss_test = self.net.loss(self.test_feature_placeholder, test=True)
            print('Loss set up.')

            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder,
                                               epsilon=1e-4)
            trainable = tf.trainable_variables()
            # for var in trainable:
            #     print(var)
            self.optim = optimizer.minimize(self.loss, var_list=trainable)

            # Set up logging for TensorBoard.
            if self.tb_logging:
                self.writer = tf.summary.FileWriter(self.logdir)
                self.writer.add_graph(tf.get_default_graph())
                run_metadata = tf.RunMetadata()
                self.summaries = tf.summary.merge_all()

            # Set up session
            print('Setting up session.')
            config = tf.ConfigProto(log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            print('Session set up.')

            # Saver for storing checkpoints of the model.
            self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=2)

            # Try to load model
            try:
                self.saved_global_step = load(self.saver, self.sess, self.logdir)
                if self.saved_global_step is None:
                    # The first training step will be saved_global_step + 1,
                    # therefore we put -1 here for new or overwritten trainings.
                    self.saved_global_step = -1
                    print(f'No model found to restore. Initialising new model.')
                else:
                    print(f'Restored trained model from step {self.saved_global_step}.')
            except:
                print("Something went wrong while restoring checkpoint.")
                raise

    def train(self,
              learning_rate=1e-3,
              num_steps=int(5e4),
              dropout_keep_prob=0.75,
              overwrite=False,
              test_every=50,
              lr_scheduling=True,
              lr_scheduling_steps=5,
              lr_scheduling_factor=5,
              lr_scheduling_min=1e-5,
              checkpoint_every=2000):
        """
        Train the model

        Parameters
        ----------
        learning_rate : float, optional (default: 1e-3)
            Learning rate for training. If lr_scheduling is True, this is the initial learning rate.

        num_steps : int, optional (default: 5e4)
            Maximum number of training steps before stopping.

        dropout_keep_prob : float, optional (default: 0.75)
            Keep probability to use for dropout in encoder/decoder layers.

        overwrite : bool, optional (default: False)
            If False, does not allow for overwriting existing model data.
            Safety measure to prevent accidentally overwriting previously saved datasets/normalization values, and
            unintentional training continuation.

        test_every : int, optional (default: 50)
            A test step is performed after every test_every training steps.

        lr_scheduling : bool, optional (default: True)
            If True, learning rate scheduling is applied, automatically decreasing the learning rate when the test loss
            does not decrease any further for lr_scheduling_steps test steps. Once lr_scheduling_min is reached,
            assume model has converged and stop training.

        lr_scheduling_steps : int, optional (default: 5)
            If lr_scheduling is True, decrease learning rate after lr_scheduling_steps test steps without decrease
            in test loss.

        lr_scheduling_factor : int, optional (default: 5)
            Factor by which to decrease learning rate if lr_scheduling is True.

        lr_scheduling_min : int, optional (default: 50)
            Minimum learning rate. If lr_scheduling is True, training finishes once learning rate drops below this
            value.

        checkpoint_every : int, optional (default: 2000)
            Save the model after every checkpoint_every steps.
        """

        assert self.has_data, "Model is not associated with any data yet. " \
                              "Recreate CompressionVAE object for this model with X!"

        lr = learning_rate

        # Check if model already exists
        if self.has_checkpoint and self.has_params:
            print(f'Found existing model {self.logdir}.')
            self.is_trained = True

            # If model is trained and overwrite is False, stop here
            if not overwrite:
                print('To continue training this model, set overwrite=True. To train a new model, '
                      'specify a different logdir or use default "temp" directory.')
                return self
            else:
                print('Continuing model training.')

        with self.graph.as_default():

            if self.trained_once_this_session is False:
                print('Starting queues.')
                threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
                self.reader.start_threads(self.sess)
                print('Reader threads started.')
                self.trained_once_this_session = True

            last_saved_step = self.saved_global_step

            test_loss_history = []

            # Start training; If user interrupts, make sure model gets saved.
            try:
                for step in range(self.saved_global_step + 1, num_steps):
                    start_time = time.time()

                    epoch = self.reader.get_epoch(self.batch_size, step)

                    # Run the actual optimization step
                    if self.tb_logging:
                        summary, loss_value, _ = self.sess.run([self.summaries, self.loss, self.optim],
                                                               feed_dict={self.dropout_placeholder: dropout_keep_prob,
                                                                          self.lr_placeholder: lr})
                        self.writer.add_summary(summary, step)
                    else:
                        loss_value, _ = self.sess.run([self.loss, self.optim],
                                                      feed_dict={self.dropout_placeholder: dropout_keep_prob,
                                                                 self.lr_placeholder: lr})

                    # Test step
                    if step % test_every == 0:

                        test_losses = []

                        for step_test in range(self.test_batches_full + 1):

                            if step_test == self.test_batches_full:
                                test_batch_size = self.test_batch_last
                            else:
                                test_batch_size = self.batch_size_test

                            test_features = self.test_batcher.next_batch(test_batch_size)

                            loss_value_test = self.sess.run([self.loss_test],
                                                            feed_dict={self.test_feature_placeholder: test_features,
                                                                       self.dropout_placeholder: 1.0})

                            test_losses.append(loss_value_test)

                        mean_test_loss = np.mean(test_losses)
                        test_loss_history.append(mean_test_loss)

                        if self.tb_logging:
                            _summary = tf.Summary()
                            _summary.value.add(tag='test/test_loss', simple_value=mean_test_loss)
                            _summary.value.add(tag='test/test_loss_per_feat',
                                               simple_value=mean_test_loss / self.reader.dimension)
                            self.writer.add_summary(_summary, step)

                        duration = (time.time() - start_time) / test_every
                        print('step {:d}; epoch {:.2f} - loss = {:.3f}, test_loss = {:.3f}, lr = {:.5f}, ({:.3f} sec/step)'
                              .format(step, epoch, loss_value, mean_test_loss, lr, duration))

                        # Learning rate scheduling.
                        if lr_scheduling and len(test_loss_history) >= lr_scheduling_steps:
                            if test_loss_history[-lr_scheduling_steps] < min(
                                    test_loss_history[-lr_scheduling_steps + 1:]):
                                lr /= lr_scheduling_factor
                                print(f'No improvement on validation data for {lr_scheduling_steps} test steps. '
                                      f'Decreasing learning rate by factor {lr_scheduling_factor}')

                                # Check if training should be stopped
                                if lr <= lr_scheduling_min:
                                    print(f'Reached learning rate threshold of {lr_scheduling_min}. '
                                          f'Stopping.')
                                    break

                    if step % checkpoint_every == 0:
                        save(self.saver, self.sess, self.logdir, step)
                        last_saved_step = step

                    if step == num_steps - 1:
                        print(f'Reached training step limit of {num_steps} steps. '
                              f'Stopping.')

            except KeyboardInterrupt:
                print()
            finally:
                self.is_trained = True
                self.has_checkpoint = True
                self.saved_global_step = step

                if step > last_saved_step:
                    save(self.saver, self.sess, self.logdir, step)
                # self.coord.request_stop()
                # self.coord.join(threads)

        return self

    def embed(self,
              X,
              batch_size=64):
        """
        Embed data into the latent space of a trained model

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data to embed.

        batch_size : int, optional (default: 64)
            Batch size for processing input data.

        Returns
        -------
        z : array, shape (n_samples, dim_latent)
            Embedding of the input data in latent space.
        """

        X = X.astype(np.float32)

        num_data = X.shape[0]
        num_batches_full = int(num_data / batch_size)
        batch_last = num_data - (num_batches_full * batch_size)
        if batch_last > 0:
            num_batches = num_batches_full + 1
        else:
            num_batches = num_batches_full

        embs = []

        for k in range(num_batches):

            if k == num_batches_full:
                input_batch = X[k * batch_size:]
            else:
                input_batch = X[k * batch_size: (k + 1) * batch_size]

            # Normalize
            if self.feature_normalization:
                input_batch -= self.mean
                input_batch = np.divide(input_batch, self.norm, out=np.zeros_like(input_batch), where=self.norm != 0)

            emb = self.sess.run([self.embeddings],
                                feed_dict={self.test_feature_placeholder: input_batch})

            embs.append(emb[0])

        # Concatenate
        z = np.concatenate(embs, axis=0)

        return z

    def decode(self,
               z):
        """
        Decode latent vectors from latent space of a trained model

        Parameters
        ----------
        z : array, shape (n_samples, dim_latent)
            Latent vectors to decode.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Reconstruction of the data from latent code.
        """

        recon = self.net.decode(np.float32(z))
        reconstruction = self.sess.run(recon)

        # Reverse data normalisation
        if self.feature_normalization:
            reconstruction = np.multiply(reconstruction, self.norm)
            reconstruction += self.mean

        X = reconstruction

        return X

    def visualize(self,
                  z,
                  labels=None,
                  categories=None,
                  filename=None):
        """
        For 2d embeddings, visualize latent space.

        Parameters
        ----------
        z : array, shape (n_samples, 2)
            2D latent vectors to visualize.

        labels: array or list, shape (n_samples), optional (default: None)
            Label indices or strings for each embedding. If strings, categories parameter is ignored.

        categories: list of string, optional (default: None)
            Category names for indices in labels.

        filename: string, optional (default: None)
            If filename is given, save visualization to file. Otherwise display directly.

        """

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
        """
        Visualize latent space by scanning over a grid, decoding, and plotting as image.
        Note: This assumes that the data is image data with a single channel, and currently only works for
        two-dimensional latent spaces.

        Parameters
        ----------
        xy_range : (float, float), optional (default: (-4.0, 4.0))
            Range in the x and y directions over which to scan.

        grid_size: int, optional (default: 10)
            Number of cells along x and y directions.

        shape: (int, int), optional (default: (28, 28))
            Original shape of the image data, used to reshape the vectors to 2d images.

        clip: (float, float), optional (default: (0, 255))
            Before displaying the image, clip the decoded data in this range.

        figsize: (float, float), optional (default: (12.0, 12.0))

        filename: string, optional (default: None)
            If filename is given, save visualization to file. Otherwise display directly.

        """

        assert self.dim_latent == 2, "visualize_latent_grid only implemented for 2d latent spaces."

        xy_extent = xy_range[1] - xy_range[0]
        step_size = xy_extent / grid_size

        # Create grid of latent variables
        z_list = []
        for k in range(grid_size):
            for j in range(grid_size):
                z_list.append([xy_range[0] + (0.5 + k) * step_size,
                               xy_range[0] + (0.5 + j) * step_size])

        z_array = np.array(z_list)

        # Decode
        x_array = self.decode(z_array)

        # Arrange into image grid
        image = []
        for k in range(grid_size):
            row = []
            for j in range(grid_size):
                index = k * grid_size + j
                row.insert(0, np.reshape(x_array[index], shape))
            image.append(np.concatenate(row))

        # Concatenate into image
        image = np.concatenate(image, axis=1)

        # Apply clipping
        if clip is not None:
            image = np.clip(image, clip[0], clip[1])

        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='w', edgecolor='k')
        plt.imshow(image, cmap='Greys_r', extent=[xy_range[0], xy_range[1], xy_range[0], xy_range[1]])

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
