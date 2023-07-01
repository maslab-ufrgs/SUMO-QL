from setuptools import setup

setup(
    name='SUMO-drivers',
    version='1.1',
    packages=['sumo_drivers', 'sumo_vg'],
    install_requires=[
        'numpy',
        'pandas',
        'gym',
        'ray[rllib]',
        'traci',
        'sumolib',
        'libsumo',
        'scikit-learn',
        'igraph',
        'matplotlib',
        'pymoo',
        'seaborn',
        'cairocffi'
    ],
    author='guidytz, hugobbi',
    author_email='guidytz@gmail.com, hgugobbi@gmail.com',
    url='https://github.com/maslab-ufrgs/SUMO-drivers',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    description='A python code to handle Multi-agent Reinforcement Learning, where the agents are drivers, using SUMO as a microscopic traffic simulation. This code also handles the creation of a graph that links different elements of the traffic network that have similar patterns.'
)
