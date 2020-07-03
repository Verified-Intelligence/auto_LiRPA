import random
import json
from auto_LiRPA.utils import logger

def load_data_sst():
    data = []
    for split in ['train_all_nodes', 'train', 'dev', 'test']:
        with open('data/sst/{}.json'.format(split)) as file:
            data.append(json.loads(file.read()))
    return data

def load_data(dataset):    
    if dataset == "sst":
        return load_data_sst()
    else:
        raise NotImplementedError('Unknown dataset {}'.format(dataset))

def clean_data(data):
    return [example for example in data if example['candidates'] is not None]

def get_batches(data, batch_size):
    batches = []
    random.shuffle(data)
    for i in range((len(data) + batch_size - 1) // batch_size):
        batches.append(data[i * batch_size : (i + 1) * batch_size])
    return batches
