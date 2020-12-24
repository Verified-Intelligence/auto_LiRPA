"""Test classes for Transformer and LSTM on language tasks"""
import os
import argparse
import pickle
import torch
import numpy as np
from auto_LiRPA.utils import logger

parser = argparse.ArgumentParser()
parser.add_argument('--gen_ref', action='store_true', help='generate reference results')
parser.add_argument('--train', action='store_true', help='pre-train the models')
args, unknown = parser.parse_known_args()   

def prepare_data():
    os.system('cd ../examples/language;\
        wget http://download.huan-zhang.com/datasets/language/data_language.tar.gz;\
        tar xvf data_language.tar.gz')

cmd_transformer_train = 'cd ../examples/language; \
    DIR=model_transformer_test; \
    python train.py --hidden_size=16 --embedding_size=16 --intermediate_size=16 --max_sent_length=16 \
    --dir=$DIR --robust --method=IBP+backward_train \
    --num_epochs=2 --num_epochs_all_nodes=1 --eps_start=2 --train'
cmd_transformer_test = 'cd ../examples/language; \
    python train.py --hidden_size=16 --embedding_size=16 --intermediate_size=16 --max_sent_length=16 \
    --robust --method=IBP+backward --budget=1 --auto_test --eps=0.2 --load=../../tests/data/ckpt_transformer \
    --device=cpu'
cmd_lstm_train = 'cd ../examples/language; \
    DIR=model_lstm_test; \
    python train.py  --hidden_size=16 --embedding_size=16 --max_sent_length=16 \
    --dir=$DIR --model=lstm --lr=1e-3 --robust --method=IBP+backward_train --dropout=0.5 \
    --num_epochs=2 --num_epochs_all_nodes=1 --eps_start=2 --train'
cmd_lstm_test = 'cd ../examples/language; \
    python train.py --model=lstm --hidden_size=16 --embedding_size=16 --max_sent_length=16 \
    --robust --method=IBP+backward --budget=1 --auto_test --eps=0.2 --load=../../tests/data/ckpt_lstm \
    --device=cpu'
res_path = '../examples/language/res_test.pkl'

"""Pre-train a simple Transformer and LSTM respectively"""
def train():
    if os.path.exists("../examples/language/model_transformer_test"):
        os.system("rm -rf ../examples/language/model_transformer_test")
    if os.path.exists("../examples/language/model_lstm_test"):
        os.system("rm -rf ../examples/language/model_lstm_test")
    logger.info("Training a Transformer")
    os.system(cmd_transformer_train)
    os.system("cp ../examples/language/model_transformer_test/ckpt_2 data/ckpt_transformer")
    logger.info("Training an LSTM")
    os.system(cmd_lstm_train)
    os.system("cp ../examples/language/model_lstm_test/ckpt_2 data/ckpt_lstm")

def read_res():
    with open(res_path, 'rb') as file:
        return pickle.load(file)

def evaluate():
    logger.info('Evaluating the trained Transformer')
    os.system(cmd_transformer_test)
    res_transformer = read_res()
    logger.info('Evaluating the trained LSTM')
    os.system(cmd_lstm_test)
    res_lstm = read_res()
    os.system("rm {}".format(res_path))
    return res_transformer, res_lstm

def gen_ref():
    if args.train:
        train()
    res_transformer, res_lstm = evaluate()
    with open('data/language_test_data', 'wb') as file:
        pickle.dump((res_transformer, res_lstm), file)
    logger.info('Reference results saved')

def check():
    with open('data/language_test_data', 'rb') as file:
        res_transformer_ref, res_lstm_ref = pickle.load(file)
    res_transformer, res_lstm = evaluate()
    for res, res_ref in zip([res_transformer, res_lstm], [res_transformer_ref, res_lstm_ref]):
        for a, b in zip(res, res_ref):
            ta, tb = torch.tensor(a), torch.tensor(b)
            assert torch.max(torch.abs(ta - tb)) < 1e-5
            assert (torch.tensor(a) - torch.tensor(b)).pow(2).sum() < 1e-9

def test():
    if not os.path.exists('../examples/language/data'):
        prepare_data()
    if args.gen_ref:
        gen_ref()
    else:
        check()
    logger.info("test_Language done")

if __name__ == '__main__':
    test()
