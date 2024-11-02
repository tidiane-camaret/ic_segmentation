# setup.py
from setuptools import setup, find_packages

setup(
    name="your-package-name",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
        'numpy',
        'nibabel',
        # add other requirements
    ],
)