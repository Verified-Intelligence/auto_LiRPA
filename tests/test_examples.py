"""Test all the examples before release"""
import pytest
import subprocess
import os
import sys
import shlex
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', type=str, default=None)
args = parser.parse_args()   

pytest_skip = pytest.mark.skip(
    reason="It should be tested on a GPU server and excluded from CI")

if not 'CACHE_DIR' in os.environ:
    cache_dir = os.path.join(os.getcwd(), '.cache')
else:
    cache_dir = os.environ['CACHE_DIR']
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

def download_data_language():
    url = "http://download.huan-zhang.com/datasets/language/data_language.tar.gz"
    if not os.path.exists('../examples/language/data/sst'):
        subprocess.run(shlex.split(f"wget {url}"), cwd="../examples/language")
        subprocess.run(shlex.split(f"tar xvf data_language.tar.gz"), 
            cwd="../examples/language")

@pytest_skip
def test_transformer():
    cmd = f"""python train.py --dir {cache_dir} --robust
        --method IBP+backward_train --train --num_epochs 2 --num_epochs_all_nodes 2
        --eps_start 2 --eps_length 1 --eps 0.1"""    
    print(cmd, file=sys.stderr)
    download_data_language()
    subprocess.run(shlex.split(cmd), cwd='../examples/language')

@pytest_skip
def test_lstm():
    cmd = f"""python train.py --dir {cache_dir}
        --model lstm --lr 1e-3 --dropout 0.5 --robust
        --method IBP+backward_train --train --num_epochs 2 --num_epochs_all_nodes 2
        --eps_start 2 --eps_length 1 --eps 0.1
        --hidden_size 2 --embedding_size 2 --intermediate_size 2 --max_sent_length 4"""
    print(cmd, file=sys.stderr)
    download_data_language()
    subprocess.run(shlex.split(cmd), cwd='../examples/language')

#FIXME this is broken
@pytest_skip
def test_lstm_seq():
    cmd = f"""python train.py --dir {cache_dir}
        --hidden_size 2 --num_epochs 2 --num_slices 4"""
    print(cmd, file=sys.stderr)
    subprocess.run(shlex.split(cmd), cwd='../examples/sequence')   

@pytest_skip
def test_simple_verification():
    cmd = "python simple_verification.py"
    print(cmd, file=sys.stderr)
    subprocess.run(shlex.split(cmd), cwd='../examples/vision')    

@pytest_skip
def test_simple_training():
    cmd = """python simple_training.py
        --num_epochs 5 --scheduler_opts start=2,length=2"""
    print(cmd, file=sys.stderr)
    subprocess.run(shlex.split(cmd), cwd='../examples/vision')

@pytest_skip
def test_cifar_training():
    cmd = """python cifar_training.py
        --batch_size 64 --model ResNeXt_cifar
        --num_epochs 5 --scheduler_opts start=2,length=2"""
    print(cmd, file=sys.stderr)
    subprocess.run(shlex.split(cmd), cwd='../examples/vision')

@pytest_skip
def test_weight_perturbation():
    cmd = """python weight_perturbation_training.py
        --norm 2 --bound_type CROWN-IBP
        --num_epochs 3 --scheduler_opts start=2,length=1 --eps 0.01"""
    print(cmd, file=sys.stderr)
    subprocess.run(shlex.split(cmd), cwd='../examples/vision')

@pytest_skip
def test_tinyimagenet():
    cmd = f"""python tinyimagenet_training.py
        --batch_size 32 --model wide_resnet_imagenet64
        --num_epochs 3 --scheduler_opts start=2,length=1 --eps {0.1/255}
        --in_planes 2 --widen_factor 2"""
    print(cmd, file=sys.stderr)
    if not os.path.exists('../examples/vision/data/tinyImageNet/tiny-imagenet-200'):
        subprocess.run(shlex.split("bash tinyimagenet_download.sh"), 
        cwd="../examples/vision/data/tinyImageNet")    
    subprocess.run(shlex.split(cmd), cwd='../examples/vision')

@pytest_skip
def test_imagenet():
    cmd = f"""python imagenet_training.py
        --batch_size 32 --model wide_resnet_imagenet64_1000class
        --num_epochs 3 --scheduler_opts start=2,length=1 --eps {0.1/255}
        --in_planes 2 --widen_factor 2"""
    print(cmd)
    if (not os.path.exists('../examples/vision/data/ImageNet64/train') or 
            not os.path.exists('../examples/vision/data/ImageNet64/test')):
        print('Error: ImageNet64 dataset is not ready.')
        return -1
    subprocess.run(shlex.split(cmd), cwd='../examples/vision')

@pytest_skip
def test_release():
    if args.test:
        # Only run a specified test
        eval(f'test_{args.test}')()
    else:
        # Run all tests
        test_simple_training()
        test_transformer()
        test_lstm()
        test_lstm_seq()
        test_simple_verification()
        test_cifar_training()
        test_weight_perturbation()
        test_tinyimagenet()

if __name__ == '__main__':
    test_release()
