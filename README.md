# fishtoolbox

Contains a set of modules to analyze and visualize data from block1 and block2 of the experiment from 2021 September. 

### Dependencies
Install the following packages:
- [fishproviz](https://github.com/lukastaerk/Fish-Tracking-Visualization)
- [motionmapperpy fork](https://github.com/lukastaerk/motionmapperpy)

### Using the HPC cluster
1. Start on the GPU
`sbatch scripts/hpc-python.sh`
2. NOTEBOOK
- `conda activate rapids-22.04`
- Type `ifconfig` and get the `inet` entry for `eth0`, i.e. the IP address of the node
- `srun --pty --partition=ex_scioi_gpu --gres=gpu:1 --time=0-02:00 bash -i` to start a new shell with a GPU
- `ssh -L localhost:5000:localhost:5000 user.name@[IP address]` on your local machine
- `jupyter-lab --no-browser --port=5000`