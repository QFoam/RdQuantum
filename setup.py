from setuptools import setup, find_packages

setup(
    name="rdquantum",
    version="2023.04.22",
    install_requires=["gymnasium==0.28.1", "qutip==4.7.0"],
    packages=find_packages(),
)
