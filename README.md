# ionmob
Prediction Framework for Peptide Ion-Mobilities

setup virtual environment and install ionmob

python3.8 -m venv your-env
source your-env/bin/activate

#install wheel for build
pip install wheel jupyter
#allows you to select your-env from jupyter
ipython kernel install --name "local-venv" --user

#build ionmob package
python setup.py bdist_wheel
#install ionmob package
pip install -e .

