# fishtoolbox

Contains a set of modules to analyze and visualize data from block1 and block2 of the experiment from 2021 September. 

## Dependencies
This repository is based on python and therefore requires conda and python-pip for installations.
The following repositories are project-dependencies that have to been build inside the underlying environment:
- [fishproviz](https://github.com/lukastaerk/Fish-Tracking-Visualization)

- [motionmapperpy fork](https://github.com/lukastaerk/motionmapperpy)

#### Data Flow
![Dataflow](./docs/dataflow.pdf)

### Using the HPC cluster
## Installation
1. Environment installations using Conda, including the python environment and c++ dependencies
    ```bash
    conda env create -n toolbox --file environment.yml
    conda activate toolbox
    ```
2. Python package installations using python-pip
    ```bash
    # conda environment should be activated
    python -m venv .venv # python virtual environment creation
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
3. Fishproviz project-dependencies installation
    ```bash
    # working directory should be equal to <path/to/fishtoolbox>
    # conda environment should be activated
    # python venv should be activated
    cd ..
    git clone git@github.com:lukastaerk/Fish-Tracking-Visualization.git
    cd Fish-Tracking-Visualization
    python setup.py install
    ```
4. Motionmapper project-dependencies installation
    ```bash
    # working directory should be equal to <path/to/fishtoolbox>
    # conda environment should be activated
    # python venv should be activated
    cd ..
    git clone git@github.com:lukastaerk/motionmapperpy.git
    cd motionmapperpy
    python setup.py install
    ```

## HPC Usage
1. Start on the GPU
`sbatch scripts/hpc-python.sh`
2. NOTEBOOK
- `conda activate rapids-22.04`
- Type `ifconfig` and get the `inet` entry for `eth0`, i.e. the IP address of the node
- `srun --pty --partition=ex_scioi_gpu --gres=gpu:1 --time=0-02:00 bash -i` to start a new shell with a GPU
- `ssh -L localhost:5000:localhost:5000 user.name@[IP address]` on your local machine
- `jupyter-lab --no-browser --port=5000`

## Start
- set the BLOCK variable to BLOCK1 or BLOCK2 in `config.py`
- set the projectPath variable to the path of a new folder in `config.py` this is where the data will be stored
- setup fishprovis with the correct paths and area configurations. 
- export the preprocessed data with `python3 -m data_factory.processing` 
- repeat for the other block

# Program Parts
## Parameters 
- `parameters = set_parameters()` to get the parameters that are used throughout the fishtoolbox 

## Data Factory 
### Processing 
- `load_trajectory_data_concat` load the x y coordinates, projections (the three features), time index, area
- `load_zVals_concat` load the umap data
- `load_clusters_concat` load cluster labels for individuals and day 
    paramerter.kmeans = 5 to specify the clustering that you want to load. 

## Plasticity  
There are three ways in this module to compute plasticity. 
- `compute_cluster_entropy` computes the cluster entropy for each individual and day. Using the watershed regions or kmeans clusters, by providing the function to load the corresponding clusters.
- `compute_coefficient_of_variation` computes the coefficient of variation for each individual and day.

## Table Export 
Records function to export averaged step length to a csv file and melted them into a long format table for statistical analysis (Repeatability).

# Repeatability 
## ![Repeatability](https://cran.r-project.org/web/packages/rptR/vignettes/rptR.html)
From means of features (step, angle, wall distance), e.g. batches of 60 data frames. Produce a long table, recording block number, id. 
## Sampling
The research question is how many samples are needed to get a good estimate of the repeatability.
Provided a table with means of a feature (step length) over a number of consecutive data frames, we can sample from this table a number of minutes for a number of days. Further we look at the effect when sampling the time of the day only once for all days versus sampling the time of the day for each day.


## Poltting 
### Caterpillar Plots
- ethnogram_of_clusters

# TODOs:
- check the new area files, see if there are significant updates for any of them, what is the difference, do we need an refined get_area_function(fishkey,day) ? 


