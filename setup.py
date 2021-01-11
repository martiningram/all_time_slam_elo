from os import getenv
from setuptools import setup
from setuptools import find_packages


setup(
    name="all-time-elo",
    version=getenv("VERSION", "LOCAL"),
    description="Elo",
    packages=find_packages(),
)
