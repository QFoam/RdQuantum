from setuptools import setup, find_packages

setup(
    name="rdquantum",
    version="2023.05.17",
    install_requires=["gymnasium==0.26.3", "ray[rllib]", "qutip==4.7.0", "tensorflow", "torch"],
    packages=find_packages(),
)
