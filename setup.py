from setuptools import setup, find_packages

REQUIREMENTS = [
    'numpy',
    'torch',
    'torchvision'
]

setup(
    name = 'torchvision-yolov3',
    version = '0.6.0',
    description = 'A minimal PyTorch implementation of YOLOv3.',
    author = 'Bob Liu',
    maintainer = 'Danilo Peixoto',
    maintainer_email = 'danilopeixoto@outlook.com',
    license = 'BSD',
    url = 'https://github.com/danilopeixoto/torchvision-yolov3',
    download_url = 'https://github.com/danilopeixoto/torchvision-yolov3/archive/v0.6.0.tar.gz',
    packages = find_packages(),
    python_requires = '>=3.6',
    install_requires = REQUIREMENTS,
    zip_safe = False)
