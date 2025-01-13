import os
import numpy as np
import tensorflow as tf
import joblib
from typing import Optional, Tuple, Union


def normalize_features(feat_array: np.ndarray, logdir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize features and save/load normalization factors.
    
    Args:
        feat_array: Input features array of shape [n_samples, n_features]
        logdir: Directory to save normalization factors
        
    Returns:
        mean: Mean values for each feature
        norm: Normalization factors for each feature
    """
    # Create directory if it doesn't exist
    os.makedirs(logdir, exist_ok=True)
    
    norm_file = os.path.join(logdir, 'norm.pkl')
    
    if not os.path.isfile(norm_file):
        print('Calculating normalisation factors.')
        mean = np.mean(feat_array, axis=0)
        var = np.var(feat_array, axis=0)
        norm = np.sqrt(var)  # Normalize by standard deviation
        
        # Also store min/max for potential future use
        max_val = np.max(feat_array, axis=0)
        min_val = np.min(feat_array, axis=0)
        
        norm_dict = {
            'mean': mean,
            'norm': norm,
            'min_val': min_val,
            'max_val': max_val
        }
        joblib.dump(norm_dict, norm_file)
        print('Normalisation factors calculated.')
    else:
        print('Loading existing normalisation factors.')
        norm_dict = joblib.load(norm_file)
        mean = norm_dict['mean']
        norm = norm_dict['norm']
    
    return mean, norm


class DataLoader:
    """Modern TF2.x data loader using tf.data.Dataset.
    
    This replaces both the DataReader and Batcher classes from the TF1.x implementation.
    """
    
    def __init__(
        self,
        features: Union[np.ndarray, str],
        batch_size: int = 64,
        feature_normalization: bool = True,
        shuffle: bool = True,
        logdir: str = 'temp',
        cache: bool = True
    ):
        """Initialize the data loader.
        
        Args:
            features: Input features as numpy array or path to directory with .npy files
            batch_size: Batch size for training
            feature_normalization: Whether to normalize features
            shuffle: Whether to shuffle the data
            logdir: Directory to save normalization factors
            cache: Whether to cache the dataset in memory
        """
        self.batch_size = batch_size
        self.feature_normalization = feature_normalization
        self.shuffle = shuffle
        self.logdir = logdir
        
        # Create logdir if it doesn't exist
        os.makedirs(logdir, exist_ok=True)
        
        # Load and prepare features
        if isinstance(features, str):
            # TODO: Implement loading from directory
            raise NotImplementedError("Loading from directory not implemented yet")
        else:
            self.features = features.astype(np.float32)
            
        self.num_samples = len(self.features)
        self.feature_dim = self.features.shape[1]
        
        # Compute or load normalization factors
        if self.feature_normalization:
            self.mean, self.norm = normalize_features(self.features, self.logdir)
            
        # Create dataset
        self._create_dataset(cache=cache)
    
    def _normalize_features(self, features: tf.Tensor) -> tf.Tensor:
        """Normalize features using pre-computed factors."""
        if not self.feature_normalization:
            return features
            
        features = features - self.mean
        # Handle zero variance features
        features = tf.where(
            tf.not_equal(self.norm, 0),
            features / self.norm,
            tf.zeros_like(features)
        )
        return features
    
    def _create_dataset(self, cache: bool = True):
        """Create the tf.data.Dataset pipeline."""
        # Create initial dataset
        dataset = tf.data.Dataset.from_tensor_slices(self.features)
        
        # Normalize features
        if self.feature_normalization:
            dataset = dataset.map(
                self._normalize_features,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
        # Shuffle if requested
        if self.shuffle:
            dataset = dataset.shuffle(
                buffer_size=10000,
                reshuffle_each_iteration=True
            )
            
        # Cache if requested (speeds up training)
        if cache:
            dataset = dataset.cache()
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        self.dataset = dataset
    
    def get_iterator(self) -> tf.data.Dataset:
        """Get the dataset iterator."""
        return iter(self.dataset)
    
    def get_dataset(self) -> tf.data.Dataset:
        """Get the tf.data.Dataset object."""
        return self.dataset
