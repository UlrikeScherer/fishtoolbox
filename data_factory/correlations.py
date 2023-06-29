"""
Plotting functions for correlation matrices, showing how the step length of the fish are correlated over time.
"""

import matplotlib.pyplot as plt
import numpy as np

from data_factory.table_export import step_length_avg
from .utils import get_individuals_keys, set_parameters
from data_factory.plot_helpers import remove_spines


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
