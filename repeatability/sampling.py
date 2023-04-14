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

def plot_repeatability_sambling(datadf, n_sample_list, true_vals, ylabels=["repeatability", "group level variance", "residual variance"], title=""):
    fig,axes = plt.subplots(len(ylabels),1, figsize=(7,7), sharex=True, squeeze=False)
    for ax, t, tv in zip(axes, ylabels, true_vals):
        ax = ax[0]
        datadf.boxplot(ax=ax, grid=False, showfliers=False, column=t, by="n_samples")
        ax.plot([1,len(n_sample_list)], [tv,tv],"--", label="true %s %.2f"%(t,tv), color="k")
        ax.set_xlabel("#samples")
        ax.set_ylabel(t)
        ax.set_title("")
        ax.legend()
        for i in range(len(n_sample_list)):
            s = n_sample_list[i]
            y = datadf.query("n_samples == @s")[t]
            x = np.random.normal(i+1, 0.04, size=len(y))
            ax.plot(x, y, 'r.', alpha=0.2)
    axes[0][0].set_title(title)
    
    #axes[0].set_xticklabels(n_sample_list)
    
    plt.close()
    return fig

def repeatability_sampling(table, n_iter, n_sample_list, sametime=False, timeinterval=60):
    reps = np.array([[repeatability_lmm(sample_days_mins(table, n_sep, sametime=sametime, timeinterval=timeinterval)) for i in range(n_iter)] 
                for n_sep in n_sample_list])
    reps_vstack = np.vstack(reps) 
    df = pd.DataFrame(reps_vstack, columns=["repeatability", "group level variance", "residual variance"])
    df["n_samples"] = np.repeat(n_sample_list, n_iter)
    df["iter"] = np.tile(range(n_iter), len(n_sample_list))
    return df

# queries the table for by time interval and returns the repeatability for each time interval
def repeatability_over_time(table, interval=60, func=repeatability_lmm):
    return np.array([[func(table.query("minutes >= @i & minutes < @j")) for (i,j) in zip(range(0,60*8,interval)
                                                                                                      ,range(interval,60*8+interval,interval))]])

# script to run the sampling
if __name__ == "__main__":
    import os
    parameters = set_parameters()
    ndfs=60*5*10
    nminutes = [10] #[60, 30, 10, 5]
    n_sample_list = [2,4,8,12,16,20,24,28]
    sametime = [False, True]
    n_iter= 30
    rep_sam_dir = parameters.projectPath+"/repeatability_sampling"
    os.makedirs(rep_sam_dir, exist_ok=True)
    for nmin in nminutes:
        matrix = pd.read_csv(parameters.projectPath+"/avg_step_by_%ddfs.csv"%ndfs,index_col=0)
        meld_matrix = get_melted_table(matrix)
        true_values = repeatability_lmm(meld_matrix)
        for st in sametime:
            title = "same time: %s, sample duration %d min"%(st,nmin)
            file_name = "repeatability_sampling_%dmin_%dtime_%dndfs.csv"%(nmin, st, ndfs)
            if os.path.exists(f"{rep_sam_dir}/{file_name}"):
                res = pd.read_csv(f"{rep_sam_dir}/{file_name}", index_col=0)
            else:
                res = repeatability_sampling(meld_matrix, n_iter,n_sample_list, sametime=st, timeinterval=nmin)
                res.to_csv(f"{rep_sam_dir}/{file_name}")
            fig = plot_repeatability_sambling(res,n_sample_list, true_values, title=title)
            fig.savefig(f"{rep_sam_dir}/{file_name}.pdf")
