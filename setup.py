from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    ld = fh.read()

setup(
    name="ionmob",
    version="0.2.0",
    description="predict peptide collision-cross sections / ion-mobilities",
    packages=[
        "ionmob",
        "ionmob.models",
        "ionmob.preprocess",
        "ionmob.utilities",
        "tests",
    ],
    # package_data={
    #     "example_data": ["*.parquet"],
    #     "pretrained_models": ["*"],
    # },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    long_description=ld,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=[
        "tensorflow >=2.7",
        "pyopenms >= 1.0",
        "pandas >=1.1",
        "pyarrow >= 0.5",
        "scipy >=1.5",
        "scikit-learn >=1.0.0",
        "h5py >=3.0.0",
        "biopython >=1.5",
        "matplotlib >=3.0",
    ],
)
