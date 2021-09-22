from setuptools import setup

with open("README.md", "r") as fh:
    ld = fh.read()

setup(
    name='ionmob',
    version='0.0.1',
    description='predict peptide ion-mobilities',
    packages=['ionmob.models',
              'ionmob.preprocess',
              'ionmob.alignment'
              ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv3+)",
        "Operating System :: OS Independent"
    ],
    long_description=ld,
    long_description_content_type="text/markdown",
    install_requires=[
        "tensorflow >=2.4",
        "pandas >=1.1",
        "scipy >=1.5"
    ],
)
