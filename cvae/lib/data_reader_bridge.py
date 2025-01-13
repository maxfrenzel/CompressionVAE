"""Bridge between TF1.x and TF2.x data loading implementations.

This module provides wrapper classes that maintain the same interface as the old
TF1.x DataReader and Batcher classes, but use the new TF2.x data loading implementation
internally.
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Union

from cvae.lib import data_reader_tf2 as dra2


class DataReaderBridge:
    """Bridge class that provides TF1.x DataReader interface but uses TF2.x loader internally."""
    
    def __init__(self,
                 feat_array: np.ndarray,
                 feature_normalization: bool,
                 coord: tf.compat.v1.train.Coordinator,
                 logdir: str,
                 queue_size: int = 128):
        """Initialize the data reader bridge.
        
        Args:
            feat_array: Input features array
            feature_normalization: Whether to normalize features
            coord: TF coordinator (not used in TF2 implementation)
            logdir: Directory to save normalization factors
            queue_size: Queue size (used for prefetch in TF2)
        """
        self.feat_array = feat_array
        self.normalize = feature_normalization
        self.num_data = feat_array.shape[0]
        self.dimension = feat_array.shape[1]
        self.coord = coord  # Kept for compatibility
        self.logdir = logdir
        self.queue_size = queue_size
        
        print('Total amount of data: ', self.num_data)
        print("Input feature dimension: ", self.dimension)
        
        # Create TF2 data loader
        self.loader = dra2.DataLoader(
            features=feat_array,
            feature_normalization=feature_normalization,
            logdir=logdir,
            cache=True
        )
        
        # Create placeholders and dataset ops compatible with TF1.x graph mode
        with tf.name_scope('data_reader'):
            self._batch_size_placeholder = tf.compat.v1.placeholder(tf.int64, shape=[], name='batch_size')
            self._dataset = self.loader.get_dataset()
            self._dataset = self._dataset.repeat()
            self._iterator = tf.compat.v1.data.make_initializable_iterator(self._dataset)
            self._next_batch = self._iterator.get_next()
            
            # Initialize iterator
            self._init_op = self._iterator.initializer
    
    def dequeue_feature(self, batch_size: int) -> tf.Tensor:
        """Get the next batch of features.
        
        Args:
            batch_size: Size of batch to return
            
        Returns:
            Tensor of shape [batch_size, dimension]
        """
        # Update batch size if needed
        if batch_size != self.loader.batch_size:
            self.loader = dra2.DataLoader(
                features=self.feat_array,
                batch_size=batch_size,
                feature_normalization=self.normalize,
                logdir=self.logdir,
                cache=True
            )
            # Reinitialize dataset with new batch size
            self._dataset = self.loader.get_dataset()
            self._dataset = self._dataset.repeat()
            self._iterator = tf.compat.v1.data.make_initializable_iterator(self._dataset)
            self._next_batch = self._iterator.get_next()
            self._init_op = self._iterator.initializer
        
        return self._next_batch
    
    def start_threads(self, sess: tf.compat.v1.Session, n_threads: int = 1) -> list:
        """Initialize the iterator in the session.
        
        Args:
            sess: TF1.x session
            n_threads: Number of threads (not used)
            
        Returns:
            Empty list (no threads needed)
        """
        sess.run(self._init_op)
        return []
        
    def get_epoch(self, batch_size: int, step: int) -> float:
        """Calculate the current epoch number.
        
        Args:
            batch_size: Batch size being used
            step: Current step number
            
        Returns:
            Current epoch number
        """
        return (batch_size * step) / self.num_data


class BatcherBridge:
    """Bridge class that provides TF1.x Batcher interface but uses TF2.x loader internally."""
    
    def __init__(self,
                 feat_array: np.ndarray,
                 feature_normalization: bool,
                 logdir: str,
                 shuffle: bool = False):
        """Initialize the batcher bridge.
        
        Args:
            feat_array: Input features array
            feature_normalization: Whether to normalize features
            logdir: Directory to save normalization factors
            shuffle: Whether to shuffle the data
        """
        self.feat_array = feat_array
        self.normalize = feature_normalization
        self.logdir = logdir
        self.shuffle = shuffle
        
        # Create TF2 data loader
        self.loader = dra2.DataLoader(
            features=feat_array,
            feature_normalization=feature_normalization,
            logdir=logdir,
            shuffle=shuffle,
            cache=True
        )
        
        # Store normalization factors
        if feature_normalization:
            self.mean = self.loader.mean
            self.norm = self.loader.norm
        
        self.num_data = len(feat_array)
        print('Total amount of data: ', self.num_data)
        
        # Create graph mode compatible ops
        with tf.name_scope('batcher'):
            self._batch_size_placeholder = tf.compat.v1.placeholder(tf.int64, shape=[], name='batch_size')
            self._dataset = self.loader.get_dataset()
            self._iterator = tf.compat.v1.data.make_initializable_iterator(self._dataset)
            self._next_batch = self._iterator.get_next()
            self._init_op = self._iterator.initializer
            self._initialized = False
            self._current_session = None
    
    def next_batch(self, batch_size: int, sess: Optional[tf.compat.v1.Session] = None) -> np.ndarray:
        """Get the next batch of features.
        
        Args:
            batch_size: Size of batch to return
            sess: TF session to use (optional, will try to get default session if None)
            
        Returns:
            Array of shape [batch_size, dimension]
        """
        # Get session
        if sess is None:
            sess = tf.compat.v1.get_default_session()
        if sess is None:
            raise RuntimeError("No TensorFlow session found. Either provide a session or ensure there is a default session.")
            
        # Initialize iterator if needed
        if not self._initialized or sess is not self._current_session:
            self._current_session = sess
            
            # Update batch size if needed
            if batch_size != self.loader.batch_size:
                self.loader = dra2.DataLoader(
                    features=self.feat_array,
                    batch_size=batch_size,
                    feature_normalization=self.normalize,
                    logdir=self.logdir,
                    shuffle=self.shuffle,
                    cache=True
                )
                self._dataset = self.loader.get_dataset()
                self._iterator = tf.compat.v1.data.make_initializable_iterator(self._dataset)
                self._next_batch = self._iterator.get_next()
                self._init_op = self._iterator.initializer
            
            sess.run(self._init_op)
            self._initialized = True
        
        try:
            batch = sess.run(self._next_batch)
        except tf.errors.OutOfRangeError:
            # Reinitialize if we've reached the end
            sess.run(self._init_op)
            batch = sess.run(self._next_batch)
            
        return batch
    
    def get_epoch(self, batch_size: int, step: int) -> float:
        """Calculate the current epoch number.
        
        Args:
            batch_size: Batch size being used
            step: Current step number
            
        Returns:
            Current epoch number
        """
        return (batch_size * step) / self.num_data
