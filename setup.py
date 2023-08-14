from setuptools import setup, find_packages

setup(
    name='time-warp-attend',
    description='Learning topological invariants of dynamical systems',
    version='0.1.0',
    packages=find_packages(),
    entry_points={'console_scripts': ['twa = twa.cli:cli']}
)
