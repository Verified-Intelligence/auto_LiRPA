from setuptools import setup, find_packages
from pathlib import Path

# Check PyTorch version
pytorch_version_l = '1.11.0'
pytorch_version_u = '2.3.0' # excluded
torchvision_version_l = '0.12.0'
torchvision_version_u = '0.18.0' # excluded
msg_install_pytorch = (f'It is recommended to manually install PyTorch '
                    f'(>={pytorch_version_l},<{pytorch_version_u}) suitable '
                    'for your system ahead: https://pytorch.org/get-started.\n')
try:
    import torch
    if torch.__version__ < pytorch_version_l:
        print(f'PyTorch version {torch.__version__} is too low. '
                        + msg_install_pytorch)
    if torch.__version__ >= pytorch_version_u:
        print(f'PyTorch version {torch.__version__} is too high. '
                        + msg_install_pytorch)
except ModuleNotFoundError:
    print(f'PyTorch is not installed. {msg_install_pytorch}')

with open('auto_LiRPA/__init__.py') as file:
    for line in file.readlines():
        if '__version__' in line:
            version = eval(line.strip().split()[-1])

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

print(f'Installing auto_LiRPA {version}')
setup(
    name='auto_LiRPA',
    version=version,
    description='A library for Automatic Linear Relaxation based Perturbation Analysis (LiRPA) on general computational graphs, with a focus on adversarial robustness verification and certification of deep neural networks.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Verified-Intelligence/auto_LiRPA',
    author='Kaidi Xu, Zhouxing Shi, Huan Zhang, Yihan Wang, Shiqi Wang, Linyi Li, Jinqi (Kathryn) Chen, Zhuolin Yang, Christopher Brix, Xiangru Zhong, Qirui Jin, Zhuowen Yuan',
    author_email='xu.kaid@husky.neu.edu, zhouxingshichn@gmail.com, huan@huan-zhang.com, wangyihan617@gmail.com, sw3215@columbia.edu, linyi2@illinois.edu, jinqic@cs.cmu.edu, zhuolin5@illinois.edu, brix@cs.rwth-aachen.de, xiangruzh0915@gmail.com, qiruijin@umich.edu, realzhuowen@gmail.com',
    packages=find_packages(),
    install_requires=[
        f'torch>={pytorch_version_l},<{pytorch_version_u}',
        f'torchvision>={torchvision_version_l},<{torchvision_version_u}',
        'numpy>=1.20',
        'packaging>=20.0',
        'pytest>=5.0',
        'pylint>=2.15',
        'pytest-order>=1.0.0',
        'appdirs>=1.4',
        'pyyaml>=5.0',
        'ninja>=1.10',
        'tqdm>=4.64',
    ],
    platforms=['any'],
    license='BSD',
)
