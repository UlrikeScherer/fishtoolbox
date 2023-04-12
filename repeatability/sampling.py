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

def sample_days_mins(table, n_sep, sametime=False, timeinterval=60):
    sep = binning_days(n_sep)
    sample_table = list()
    min_table = table.query("minutes>= @timeinterval")["minutes"]
    if sametime:
        smin = min_table.sample().values[0]
    for i in range(len(sep)-1):
        s1,s2 = sep[i],sep[i+1]
        m = table.query("day >= @s1 & day < @s2")
        sday = m["day"].sample().values[0]
        if not sametime:
            smin = min_table.sample().values[0]
        sample_table.append(table.query("day == @sday & (minutes <=@smin and minutes > @smin-@timeinterval)"))
    return pd.concat(sample_table)

def plot_repeatability_sambling(data, n_sample_list, true_vals, title=""):
    reps = pd.DataFrame(np.array(data[:,:,0]).T, columns=n_sample_list)
    V_g = pd.DataFrame(np.array(data[:,:,1]).T, columns=n_sample_list)
    V_r = pd.DataFrame(np.array(data[:,:,2]).T, columns=n_sample_list)
    data2plot = [reps, V_g, V_r]
    titles = ["repeatability", "V_g", "V_r"]
    fig,axes = plt.subplots(3,1, figsize=(7,7), sharex=True)

    for d, ax, t, tv in zip(data2plot, axes, titles, true_vals):
        d.boxplot(ax=ax, grid=False, showfliers=False)
        ax.plot([1,len(n_sample_list)], [tv,tv],"--", label="true %s %.2f"%(t,tv), color="k")
        ax.set_xlabel("#samples")
        ax.set_ylabel(t)
        ax.legend()
        ax.set_title(t)
    for i in range(len(n_sample_list)):
        y = reps.iloc[:,i]
        x = np.random.normal(i+1, 0.04, size=len(y))
        axes[0].plot(x, y, 'r.', alpha=0.2)
    #axes[0].set_xticklabels(n_sample_list)
    
    plt.close()
    return fig

def repeatability_sampling(table, n_iter, n_sample_list, sametime=False, timeinterval=60):
    return np.array([[repeatability_lmm(sample_days_mins(table, n_sep, sametime=sametime, timeinterval=timeinterval)) for i in range(n_iter)] 
                for n_sep in n_sample_list])

# queries the table for by time interval and returns the repeatability for each time interval
def repeatability_over_time(table, interval=60, func=repeatability_lmm):
    return np.array([[func(table.query("minutes >= @i & minutes < @j")) for (i,j) in zip(range(0,60*8,interval)
                                                                                                      ,range(interval,60*8+interval,interval))]])

# script to run the sampling
if __name__ == "__main__":
    import os
    parameters = set_parameters()
    ndfs=60*60*5
    nminutes =[60]# [60, 30, 10, 5]
    n_sample_list = [2,4,8,12,16,20,24,28]
    sametime = [False, True]
    n_iter= 30
    rep_sam_dir = parameters.projectPath+"/repeatability_sampling"
    os.makedirs(rep_sam_dir, exist_ok=True)
    for nmin in nminutes:
        matrix = pd.read_csv(parameters.projectPath+"/avg_step_by_%ddfs.csv"%ndfs,index_col=0)
        meld_matrix = get_melted_table(matrix)
        true_rep, V_g, V_r = repeatability_lmm(meld_matrix)
        for st in sametime:
            title = "same time: %s, sample duration %d min"%(st,nmin)
            res = repeatability_sampling(meld_matrix, n_iter,n_sample_list, sametime=st, timeinterval=nmin)
            fig = plot_repeatability_sambling(res,n_sample_list, true_rep, title=title)
            fig.savefig(rep_sam_dir+"/repeatablity_sampling_%dmin_%dtime"%(nmin, st))
