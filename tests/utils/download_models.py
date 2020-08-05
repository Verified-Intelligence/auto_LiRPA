"""Download pre-trained models if you do not locally train them"""
import os

models = [
    ('http://web.cs.ucla.edu/~zshi/files/auto_LiRPA/tests/data/ckpt_transformer', 'data/ckpt_transformer'),
    ('http://web.cs.ucla.edu/~zshi/files/auto_LiRPA/tests/data/ckpt_lstm', 'data/ckpt_lstm'),
]

if __name__ == '__main__':
    for model in models:
        os.system('wget {} -O {}'.format(model[0], model[1]))
