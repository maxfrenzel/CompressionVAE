import threading
import random
import tensorflow as tf
import numpy as np
import joblib
import os
import copy
from tqdm import tqdm


# Check or compute features
def normalize(feat_array, logdir):

    # Find normalisation factors
    norm_file = f'{logdir}/norm.pkl'
    if not os.path.isfile(norm_file):

        print('Calculating normalisation factors.')

        mean = np.mean(feat_array, axis=0)
        max_val = np.max(feat_array, axis=0)
        min_val = np.min(feat_array, axis=0)
        var = np.var(feat_array, axis=0)

        # Normalize by standard deviation
        norm = np.sqrt(var)

        norm_dict = {'mean': mean,
                     'norm': norm,
                     'min_val': min_val,
                     'max_val': max_val}

        joblib.dump(norm_dict, norm_file)

        print('Normalisation factors calculated.')

    else:
        print('Normalisation factors already stored.')


def load_norm(norm_file):
    norm_dict = joblib.load(norm_file)
    mean = norm_dict['mean']
    norm = norm_dict['norm']

    return mean, norm


def return_data(feat_array, logdir, normalize=True, randomize=True):

    # Shuffle tha data
    randomized_indices = list(range(len(feat_array)))
    if randomize:
        random.shuffle(randomized_indices)

    # If desired, load normalisation
    if normalize:
        norm_file = f'{logdir}/norm.pkl'
        mean, norm = load_norm(norm_file)

    # Loop through data
    for id_val in randomized_indices:

        # Load features and annotations and extract correct slices
        features = copy.copy(feat_array[id_val])

        # Normalise
        if normalize:
            features -= mean
            # Can occasionally have feature dimensions with zero variance, set those values to zero
            features = np.divide(features, norm, out=np.zeros_like(features), where=norm != 0)

        yield features


class DataReader(object):
    def __init__(self,
                 feat_array,
                 feature_normalization,
                 coord,
                 logdir,
                 queue_size=128):

        self.feat_array = feat_array
        self.normalize = feature_normalization
        self.num_data = feat_array.shape[0]
        self.dimension = feat_array.shape[1]
        self.coord = coord
        self.logdir = logdir
        self.threads = []

        print('Total amount of data: ', self.num_data)
        print("Input feature dimension: ", self.dimension)

        # Make sure normalization factors have been calculated
        if self.normalize:
            normalize(self.feat_array, self.logdir)

        self.feature_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=None)
        self.feature_queue = tf.compat.v1.PaddingFIFOQueue(queue_size,
                                                 ['float32'],
                                                 shapes=[[self.dimension]])
        self.feature_enqueue = self.feature_queue.enqueue([self.feature_placeholder])

    def dequeue_feature(self, num_elements):
        output = self.feature_queue.dequeue_many(num_elements)
        return output

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = return_data(self.feat_array,
                                   logdir=self.logdir,
                                   normalize=self.normalize)
            count = 0
            for feature in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                sess.run(self.feature_enqueue,
                         feed_dict={self.feature_placeholder: feature})

                count += 1

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads

    def get_epoch(self, batch_size, step):
        return (batch_size * step) / self.num_data


class Batcher(object):
    def __init__(self,
                 feat_array,
                 feature_normalization,
                 logdir,
                 shuffle=False):

        self.feat_array = feat_array
        self.normalize = feature_normalization
        self.logdir = logdir
        self.shuffle = shuffle
        self.randomized_indices = list(range(len(feat_array)))

        if self.shuffle:
            np.random.shuffle(self.randomized_indices)

        self.num_data = len(self.randomized_indices)
        print('Total amount of data: ', self.num_data)

        self.index = 0

        if self.normalize:
            self.mean, self.norm = load_norm(f'{self.logdir}/norm.pkl')

    def get_epoch(self, batch_size, step):
        return (batch_size * step) / self.num_data

    def next_batch(self, batch_size):

        feature_list = []

        data_iterator = return_data(self.feat_array,
                                    logdir=self.logdir,
                                    normalize=self.normalize,
                                    randomize=False)

        for k in range(batch_size):

            # Return features from generator, possibly recreating it if it's empty
            try:
                features = next(data_iterator)
            except:
                # Recreate the generator
                data_iterator = return_data(self.feat_array,
                                            logdir=self.logdir,
                                            normalize=self.normalize,
                                            randomize=False)
                features = next(data_iterator)

            feature_list.append(np.float32(np.expand_dims(features, axis=0)))

            self.index += 1
            if self.index == self.num_data:
                self.index = 0

                if self.shuffle:
                    np.random.shuffle(self.ids)

        feature_batch = np.concatenate(feature_list, axis=0)

        return feature_batch


def load_dataset_file(filename):

    print('Loading dataset.')

    dataset = joblib.load(filename)

    dimension = dataset['dimension']
    file_paths = dataset['file_paths']
    ids = dataset['ids']

    return ids, file_paths, dimension
