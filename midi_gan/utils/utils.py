import os
import pickle
import tarfile
import urllib
from datetime import datetime

import numpy as np
import soundfile
from matplotlib import pyplot as plt
from nesmdb.convert import exprsco_to_wav
from scipy.ndimage import gaussian_filter


def save_samples(generated_data, rate, playback_nsamps, outputs_dir, epoch):
    os.mkdir(outputs_dir + str(epoch))
    for i in range(generated_data.shape[0]):
        wav = exprsco_to_wav((rate, playback_nsamps, generated_data[i]))
        soundfile.write(outputs_dir + str(epoch) + '/' + str(i) + '.wav', wav, 44100)


def plot_save_losses(losses, labels, outputs_dir, name, sigma=3):
    plt.figure(figsize=(15, 4.5))
    for loss, label in zip(losses, labels):
        loss = gaussian_filter(loss, sigma=sigma)
        plt.plot(loss, label=label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputs_dir + name + '.png')
    plt.close()


def generate_run_dir(path, model_type):
    root_path = path + model_type + '/'
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    run_dir = root_path + current_datetime + '/'
    outputs_dir = run_dir + 'outputs/'
    model_dir = run_dir + 'models/'
    generated_datasets_dir = run_dir + 'generated_datasets/'

    os.mkdir(run_dir)
    os.mkdir(outputs_dir)
    os.mkdir(model_dir)
    os.mkdir(generated_datasets_dir)

    return run_dir, outputs_dir, model_dir, generated_datasets_dir


def load_midi_dataset(filepath, sequence_length):
    def parse_dataset(dataset, sequence_length):
        parsed_dataset = []
        for element in dataset:
            if element.shape[0] > sequence_length:
                while element.shape[0] > sequence_length:
                    parsed_dataset.append(element[:sequence_length])
                    element = element[sequence_length:]
        dataset = np.array(parsed_dataset).astype(np.float)
        return dataset

    dataset = np.load(filepath, allow_pickle=True)
    dataset = parse_dataset(dataset, sequence_length)

    dataset[:, :, :, 0] /= 108.0
    dataset[:, :, :, 1] /= 15.0
    dataset[:, :, :, 2] /= 3.0
    return dataset


def load_dataset(folder):
    dataset = []
    for filename in sorted(os.listdir(folder)):
        with open(folder + filename, 'rb') as f:
            rate, nsamps, exprsco = pickle.load(f)
            dataset.append(exprsco)
    return np.array(dataset)


def create_datasets():
    urllib.urlretrieve('http://deepyeti.ucsd.edu/cdonahue/nesmdb/nesmdb24_exprsco.tar.gz', 'nesmdb24_exprsco.tar.gz')

    with tarfile.open('nesmdb24_exprsco.tar.gz') as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path="../data/")

    train_folder = '../data/nesmdb24_exprsco/train/'
    validation_folder = '../data/nesmdb24_exprsco/valid/'
    test_folder = '../data/nesmdb24_exprsco/test/'

    train_dataset = load_dataset(train_folder)
    validation_dataset = load_dataset(validation_folder)
    test_dataset = load_dataset(test_folder)

    np.save('../data/nesmdb24_exprsco/train.npy', train_dataset)
    np.save('../data/nesmdb24_exprsco/valid.npy', validation_dataset)
    np.save('../data/nesmdb24_exprsco/test.npy', test_dataset)
