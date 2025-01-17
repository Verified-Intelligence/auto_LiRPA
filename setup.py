from setuptools import setup, find_packages
from pathlib import Path

# Check PyTorch version
pytorch_version_l = '2.0.0'
pytorch_version_u = '2.4.0' # excluded
torchvision_version_l = '0.12.0'
torchvision_version_u = '0.19.0' # excluded
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
    author='Huan Zhang, Zhouxing Shi, Xiangru Zhong, Jorge Chavez, Duo Zhou, Christopher Brix, Keyi Shen, Hongji Xu, Kaidi Xu, Hao Chen, Keyu Lu',
    author_email='huan@huan-zhang.com, zhouxingshichn@gmail.com, xiangruzh0915@gmail.com, jorgejc2@illinois.edu, duozhou2@illinois.edu, brix@cs.rwth-aachen.de, keyis2@illinois.edu, hx84@duke.edu, kx46@drexel.edu, haoc8@illinois.edu, keyulu2@illinois.edu',
    packages=find_packages(),
    install_requires=[
        f'torch>={pytorch_version_l},<{pytorch_version_u}',
        f'torchvision>={torchvision_version_l},<{torchvision_version_u}',
        'numpy>=1.20,<2.0',
        'packaging>=20.0',
        'pytest==8.1.1',
        'pylint>=2.15',
        'pytest-order>=1.0.0',
        'pytest-mock>=3.14',
        'appdirs>=1.4',
        'pyyaml>=5.0',
        'ninja>=1.10,<1.11.1.2',
        'tqdm>=4.64',
        'graphviz>=0.20.3'
    ],
    platforms=['any'],
    license='BSD',
)
