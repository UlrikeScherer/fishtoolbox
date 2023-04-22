import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .processing import load_trajectory_data
from .utils import get_individuals_keys, get_days, set_parameters, split_into_batches
from data_factory.plot_helpers import remove_spines
from config import BLOCK1, BLOCK2

N_HOURS = 8

def plot_covariance(cov_matrix, label="day",title="Correlation",symmetric_bounds=True,clip_percentile=99.5,**imshow_kwargs):
    fig, ax = plt.subplots(figsize=(8,7))
    mask= np.tri(cov_matrix.shape[0], k=-1).T
    vmin, vmax = None, None
    if symmetric_bounds:
        vmax = np.percentile(np.abs(cov_matrix), clip_percentile)
        vmin = -vmax
    cov_matrix = np.ma.array(cov_matrix, mask=mask) 
    cax = ax.imshow(
        cov_matrix,
        vmin=vmin, vmax=vmax,
        aspect="equal", interpolation='none', **imshow_kwargs)
    remove_spines(ax)
    ax.set_title(title)
    ax.set_xlabel(label)
    ax.set_ylabel(label)
    fig.colorbar(cax)
    return fig

def step_length_avg_by_day(parameters, fish_keys):
    main_df = pd.DataFrame({BLOCK1:get_days(parameters,prefix=BLOCK1), BLOCK2:get_days(parameters, prefix=BLOCK2)})
    for fk in fish_keys:
        block = BLOCK1 if BLOCK1 in fk else BLOCK2
        data = load_trajectory_data(parameters, fk)
        step_df = pd.DataFrame({
                fk:[data[i]["projections"][:,0].mean() for i in range(len(data))], 
                block:[data[i]["day"][0][0] for i in range(len(data))]
                               }).dropna()
        main_df = main_df.merge(step_df,on=[block,"DATAFRAMES"], how="left")
    return main_df

def step_length_avg(parameters, fish_keys, batch_dfs, max_rows_per_day=300):
    if batch_dfs:
        days_b1, days_b2 = get_days(parameters,prefix=BLOCK1), get_days(parameters,prefix=BLOCK2)
        DFsByDay = N_HOURS*60*60*5 
        hidx = np.concatenate([
            [(d1,d2,dfidx) for dfidx in random.sample(range(0,DFsByDay, batch_dfs), k=min(DFsByDay//batch_dfs, max_rows_per_day))] 
                               for (d1,d2) in zip(days_b1, days_b2)])
        main_df = pd.DataFrame(hidx, columns=[BLOCK1,BLOCK2,"DATAFRAMES"])
        main_df.sort_values(inplace=True, by=[BLOCK1,"DATAFRAMES"])
    for fk in fish_keys:
        block = BLOCK1 if BLOCK1 in fk else BLOCK2
        data = load_trajectory_data(parameters, fk)
        step_means = np.concatenate([list(
            map(
                lambda batch: (
                    np.mean(batch[1]),
                    di["day"][0][0],
                    map_to_batch_dfs(batch[0], batch_dfs)
                                ),
                zip(*split_into_batches(di["df_time_index"].flatten(),di["projections"][:,0], batch_size=batch_dfs))
            )
        ) for di in data])

        step_df = pd.DataFrame({
            fk:step_means[:,0], 
            block:step_means[:,1],
            "DATAFRAMES":step_means[:,2]
                            }).dropna()
        main_df = main_df.merge(step_df,on=[block,"DATAFRAMES"], how="left")
    return main_df

def map_to_batch_dfs(batch, batch_dfs):
    if len(batch)==0: return np.nan
    else: return int((batch[0]//(batch_dfs))*batch_dfs - (6*60*60*5)) # experiment starts at 6am

if __name__ == "__main__":
    parameters = set_parameters()
    FISH_KEYS = get_individuals_keys(parameters)
    
    # Example usage
    batch_dfs= 60*5
    max_rows_per_day = 300
    matrix = step_length_avg(parameters, FISH_KEYS, batch_dfs=batch_dfs, max_rows_per_day=max_rows_per_day)
    matrix.to_csv(parameters.projectPath+"/avg_step_by_%ddfs.csv"%batch_dfs)
    #matrix = pd.read_csv(parameters.projectPath+"/avg_step_by_day.csv")
    #cov_matrix = matrix[FISH_KEYS].T.corr(numeric_only=False)
    #fig= plot_covariance(cov_matrix, label="day",title="Correlation",symmetric_bounds=True,clip_percentile=99.5,cmap="RdBu_r")
    #fig.savefig(parameters.projectPath+"/correlation_avg_step_by_day_block1.pdf")
