set -e

export CC=gcc
export CXX=g++

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
# deactivate

# Use the miniconda installer for faster download / install of conda
# itself

chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda update --yes conda

# Configure the conda environment and put it in the path using the
# provided versions

conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
    numpy scipy scikit-learn  pandas joblib pyyaml

source activate testenv

# build output in the travis output when it succeeds.
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
python -c "import joblib; print('joblib %s' % joblib.__version__)"
python -c "import pandas; print('pandas %s' % pandas.__version__)"
python -c "import yaml; print('pyyaml %s' % yaml.__version__)"
