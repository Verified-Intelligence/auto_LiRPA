from setuptools import setup, find_packages
import sys

"""Check PyTorch version"""
pytorch_version_l = "1.8.0"
pytorch_version_u = "1.9.0" # excluded
msg_install_pytorch = (f"It is recommended to manually install PyTorch "
                    f"(>={pytorch_version_u},<{pytorch_version_u}) suitable "
                    "for your system ahead: https://pytorch.org/get-started.\n")
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

assert sys.version_info.major == 3, 'Python 3 is required'
if sys.version_info.minor < 8:
    # numpy 1.22 requires Python 3.8+
    numpy_requirement = 'numpy>=1.16,<=1.21'
else:
    numpy_requirement = 'numpy>=1.16'

print(f'Installing auto_LiRPA {version}')
setup(
    name='auto_LiRPA',
    version=version,
    description='A library for Automatic Linear Relaxation based Perturbation Analysis (LiRPA) on general computational graphs, with a focus on adversarial robustness verification and certification of deep neural networks.',
    url='https://github.com/KaidiXu/auto_LiRPA',
    author='Kaidi Xu, Zhouxing Shi, Huan Zhang, Yihan Wang, Shiqi Wang',
    author_email='xu.kaid@husky.neu.edu, zhouxingshichn@gmail.com, huan@huan-zhang.com, wangyihan617@gmail.com, sw3215@columbia.edu',
    packages=find_packages(),
    install_requires=[
        f'torch>={pytorch_version_l},<{pytorch_version_u}',
        'torchvision>=0.9,<0.10',
        numpy_requirement,
        'packaging>=20.0',
        'pytest>=5.0',
        'appdirs>=1.4',
        'pyyaml>=5.0',
    ],
    platforms=['any'],
    license='BSD',
)
