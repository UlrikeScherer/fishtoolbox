#!/bin/sh

source ~/miniconda3/etc/profile.d/conda.sh
echo "deactivating the base environment"
conda deactivate

echo "creating conda env from rapidsai"
conda create -n toolbox -c rapidsai -c conda-forge -c nvidia  \
    cudf=23.04 cuml=23.04 python=3.10 cudatoolkit=11.8 -y
echo "activating env"
conda activate toolbox

echo "installing additional conda deps"
conda install -y -c anaconda h5py==3.7.0
conda install -y -c conda-forge graph-tool==2.56 hdf5storage==0.1.19 moviepy==1.0.3 scikit-learn==1.2.2 scikit-image==0.20.0 statsmodels==0.14.0

echo "installing additional pip deps"
pip install -r requirements.txt

echo "installing fishproviz"
cd ../Fish-Tracking-Visualization/;
python setup.py install;

echo "installing motionmapperpy"
cd ../motionmapperpy;
python setup.py install;
cd ../fishtoolbox;
