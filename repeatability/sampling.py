import numpy as np
import pandas as pd

from data_factory.utils import set_parameters
from .repeatability import get_melted_table, repeatability_lmm, repeatability_t
import matplotlib.pyplot as plt

def binning_days(n, n_days=28):
    """
    n: number of bins
    n_days: number of days in the experiment
    return: array of n+1 elements for boundaries of bins
    """
    r = n_days%n
    sep = np.array([0]+[n_days//n]*n)
    select_r = np.random.choice(range(1,n+1), r, replace=False)
    sep[select_r]+=1
    return np.cumsum(sep)

def sample_days_mins(table, n_sep, sametime=False):
    sep = binning_days(n_sep)
    sample_table = list()
    if sametime:
        smin = table["minutes"].sample().values[0]
    for i in range(len(sep)-1):
        s1,s2 = sep[i],sep[i+1]
        m = table.query("day >= @s1 & day < @s2")
        sday = m["day"].sample().values[0]
        if not sametime:
            smin = m["minutes"].sample().values[0]
        sample_table.append(table.query("day == @sday & minutes == @smin"))
    return pd.concat(sample_table)

def plot_repeatability_sambling(data, n_sample_list, true_rep, title=""):
    reps = pd.DataFrame(np.array(data).T, columns=n_sample_list)
    fig,ax = plt.subplots()
    reps.boxplot(ax=ax, grid=False, showfliers=False)
    for i in range(len(n_sample_list)):
        y = reps.iloc[:,i]
        x = np.random.normal(i+1, 0.04, size=len(y))
        ax.plot(x, y, 'r.', alpha=0.2)
    ax.set_xticklabels(n_sample_list)
    ax.plot([1,len(n_sample_list)], [true_rep,true_rep],"--", label="true repeatability %.2f"%true_rep, color="k")
    ax.set_xlabel("#samples")
    ax.set_ylabel("repeatability")
    ax.legend()
    ax.set_title(title)
    plt.close()
    return fig

def repeatability_sampling(table, n_iter, n_sample_list, sametime=False):
    return np.array([[repeatability_t(sample_days_mins(table, n_sep, sametime=sametime)) for i in range(n_iter)] 
                for n_sep in n_sample_list])

# script to run the sampling
if __name__ == "__main__":
    import os
    parameters = set_parameters()
    nminutes = [60]#[60, 30, 10, 1]
    n_sample_list = [2,4,8,12,16,20,24,28]
    sametime = [False, True]
    n_iter= 10
    rep_sam_dir = parameters.projectPath+"/repeatability_sampling"
    os.makedirs(rep_sam_dir, exist_ok=True)
    for nmin in nminutes:
        matrix = pd.read_csv(parameters.projectPath+"/avg_step_by_%dmin.csv"%nmin,index_col=0)
        meld_matrix = get_melted_table(matrix)
        true_rep = repeatability_t(meld_matrix)
        for st in sametime:
            title = "same time: %s, sample duration %d min"%(st,nmin)
            res = repeatability_sampling(meld_matrix, n_iter,n_sample_list, sametime=st)
            fig = plot_repeatability_sambling(res,n_sample_list, true_rep, title=title)
            fig.savefig(rep_sam_dir+"/repeatablity_sampling_%dmin_%dtime"%(nmin, st))
