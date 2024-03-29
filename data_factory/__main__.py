
import motionmapperpy as mmpy
import hdf5storage
import numpy as np
import os

from .utils import get_individuals_keys, set_parameters
from .plasticity import compute_cluster_entropy
from .processing import get_regions_for_fish_key, load_zVals_concat
from .plotting import plot_transition_rates 

def main_factory(parameters, n_clusters = [5,10]):
    fish_ids = get_individuals_keys(parameters)
    for n_c in n_clusters:
        parameters.kmeans = n_c
        wshed_path = '%s/%s/zVals_wShed_groups_%s.mat'%(parameters.projectPath, parameters.method,5)
        if not os.path.exists(wshed_path):
            startsigma=1.0
            mmpy.findWatershedRegions(parameters, minimum_regions=n_c, startsigma=startsigma, pThreshold=[0.33, 0.67], saveplot=True, endident = '*_pcaModes.mat')
        wshedfile = hdf5storage.loadmat(wshed_path)
        get_clusters_func_wshed = lambda fk, d: get_regions_for_fish_key(wshedfile,fk,d)
        for flag in [True, False]:
            compute_cluster_entropy(parameters,get_clusters_func_wshed,fish_ids,parameters.kmeans,
                    name=f"cluster_entropy_wshed",by_the_hour=flag)
        plot_transition_rates(get_regions_for_fish_key(wshedfile), 
                    filename=parameters.projectPath+"/overall_kmeans_%s.pdf"%n_c)
        plot_transition_rates(load_zVals_concat(parameters)["kmeans_clusters"]+1, 
                    filename=parameters.projectPath+"/overall_kmeans_%.pdf"%n_c, cluster_remap=None)


if __name__ == "__main__":
    parameters = set_parameters()
    main_factory(parameters)