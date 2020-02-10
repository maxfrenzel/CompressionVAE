import argparse
import os
import random
import joblib
import numpy as np

def get_data_subset(ids, full_data):

    file_paths_full = full_data['file_paths']

    file_paths = dict()

    for id in ids:
        file_paths[id] = file_paths_full[id]

    datasubset = {
        'ids': ids,
        'file_paths': file_paths,
        'dimension': full_data['dimension']
    }

    return datasubset


def prepare_dataset(data_dir,
                    logdir,
                    train_ratio=0.9):

    # Get all paths of numpy files
    feature_files = []

    for dirName, subdirList, fileList in os.walk(data_dir, topdown=False):
        for fname in fileList:
            if os.path.splitext(fname)[1] in ['.npy']:
                feature_files.append('%s/%s' % (dirName, fname))

    print(f'Total number of feature vectors found: {len(feature_files)}. Building dataset.')

    # Build dataset
    ids = []
    file_paths = dict()

    for path in feature_files:
        # Find unique ID . Try filename first, if already exists add arbitrary extension
        id = os.path.splitext(os.path.basename(path))[0]
        while id in ids:
            id += 'x'

        file_paths[id] = path
        ids.append(id)

    # Get dimensionality (assume same for all)
    dimension = np.load(feature_files[0]).shape[0]
    print(f'Dimensionality of dataset: {dimension}.')

    dataset = {
        'ids': ids,
        'file_paths': file_paths,
        'dimension': dimension
    }

    # Train/valid split
    split_index = int(train_ratio * len(ids))

    for k in range(10):
        random.shuffle(ids)

    ids_train = ids[:split_index]
    ids_valid = ids[split_index:]

    print(f'Splitting {len(ids)} samples into {len(ids_train)} training and {len(ids_valid)} validation samples.')

    dataset_train = get_data_subset(ids_train, dataset)
    dataset_valid = get_data_subset(ids_valid, dataset)

    print('Saving dataset files.')

    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    joblib.dump(dataset_train, f'{logdir}/data_train.pkl')
    joblib.dump(dataset_valid, f'{logdir}/data_valid.pkl')

    print('Done.')

    return dimension
