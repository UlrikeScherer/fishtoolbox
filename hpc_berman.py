import motionmapperpy as mmpy
from data_factory.processing import compute_all_projections
from data_factory.processing import return_normalization_func
from data_factory.utils import set_parameters, get_individuals_keys
from fishproviz.utils import get_camera_pos_keys
import h5py, hdf5storage, pickle, glob
import time
import os


def factory_main():
    parameters = set_parameters()
    parameters.useGPU=-1 #0 for GPU, -1 for CPU
    parameters.training_numPoints = 5000    #% Number of points in mini-trainings.
    parameters.trainingSetSize = 72000  #% Total number of training set points to find. 
                                 #% Increase or decrease based on
                                 #% available RAM. For reference, 36k is a 
                                 #% good number with 64GB RAM.
    parameters.embedding_batchSize = 30000  #% Lower this if you get a memory error when 
                                            #% re-embedding points on a learned map.
    #from cuml import UMAP
    from umap import UMAP
    parameters.umap_module = UMAP
    
    mmpy.createProjectDirectory(parameters.projectPath)
    fish_keys = get_camera_pos_keys()
    # preprocessing should we be done with python -m preprocessing. 
    #compute_all_projections_filtered(parameters.projectPath,fish_keys,recompute=False)
    #normalize 
    indiviuals_ids = get_individuals_keys(parameters=parameters)
    print("Individuals ids: ", indiviuals_ids, "number of individuals: ", len(indiviuals_ids))
    parameters.normalize_func = return_normalization_func(parameters)
    print("Subsample from projections")
    mmpy.subsampled_tsne_from_projections(parameters, parameters.projectPath)
    print("Fit data / find embeddings")
    fit_data(parameters)
    print("Find Watershed...")
    startsigma = 4.2 if parameters.method == 'TSNE' else 1.0

    for k in parameters.kmeans_list:
        mmpy.findWatershedRegions(parameters, minimum_regions=k, startsigma=startsigma, pThreshold=[0.33, 0.67], saveplot=True, endident = '*_pcaModes.mat')
    print("Done!")
    
def fit_data(parameters):
    #tsne takes 19 mins
    tall = time.time()
    tfolder = parameters.projectPath+'/%s/'%parameters.method

    # Loading training data
    with h5py.File(tfolder + 'training_data.mat', 'r') as hfile:
        trainingSetData = hfile['trainingSetData'][:].T
    trainingSetData[trainingSetData==0] = 1e-12 # replace 0 with 1e-12

    # initialize the kmeans model on training data for different values of k
    for k in parameters.kmeans_list:
        if not os.path.exists(tfolder + '/kmeans_%i.pkl'%k):
            print('Initializing kmeans model with %i clusters'%k)
            mmpy.set_kmeans_model(k, tfolder, trainingSetData, parameters.useGPU)
    # Loading training embedding
    with h5py.File(tfolder+ 'training_embedding.mat', 'r') as hfile:
        trainingEmbedding= hfile['trainingEmbedding'][:].T

    if parameters.method == 'TSNE':
        zValstr = 'zVals' 
    else:
        zValstr = 'uVals'

    projectionFiles = glob.glob(parameters.projectPath+'/Projections/*pcaModes.mat')
    for i in range(len(projectionFiles)):
        print('Finding Embeddings')
        t1 = time.time()
        print('%i/%i : %s'%(i+1,len(projectionFiles), projectionFiles[i]))
        # Skip if embeddings already found.
        if os.path.exists(projectionFiles[i][:-4] +'_%s.mat'%(zValstr)):
            print('Already done. Skipping.\n')
            continue
        # load projections for a dataset
        projections = hdf5storage.loadmat(projectionFiles[i])['projections']
        print(projections.shape, trainingSetData.shape)

        clusters_dict = mmpy.findClusters(projections, parameters)
        for key, value in clusters_dict.items():
            hdf5storage.write(data = {"clusters":value, "k":int(key.split("_")[1])}, path = '/', truncate_existing = True,
                        filename = projectionFiles[i][:-4]+'_%s.mat'% (key), store_python_metadata = False, matlab_compatible = True)
        #del clusters_dict      

        # Find Embeddings
        zValues, outputStatistics = mmpy.findEmbeddings(projections,trainingSetData,trainingEmbedding,parameters)

        # Save embeddings
        hdf5storage.write(data = {'zValues':zValues}, path = '/', truncate_existing = True,
                        filename = projectionFiles[i][:-4]+'_%s.mat'%(zValstr), store_python_metadata = False,
                          matlab_compatible = True)
        
        

        # Save output statistics
        with open(projectionFiles[i][:-4] + '_%s_outputStatistics.pkl'%(zValstr), 'wb') as hfile:
            pickle.dump(outputStatistics, hfile)

        del zValues,projections,outputStatistics
        
    print('All Embeddings Saved in %i seconds!'%(time.time()-tall))

if __name__ == "__main__":
    tstart = time.time()
    factory_main()
    tend = time.time()
    print("Running time:", tend - tstart, "sec.")
    
