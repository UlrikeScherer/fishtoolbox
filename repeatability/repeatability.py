import glob
import os
from config import BLOCK1, BLOCK2
import numpy as np
from data_factory.table_export import df_table, get_melted_table
from data_factory.utils import get_days, set_parameters, get_individuals_keys
from data_factory.processing import load_trajectory_data
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


def repeatability_lmm(data, groups="id_cat", formula="step ~ (1 | id_cat) + block_cat"): # "step ~ (1 | id_cat) + block_cat" 
    """
    Repeatability of the step length
    :param data: pandas dataframe with columns: step, id_cat, block_cat
    :param groups: column name of the groups
    :param formula: formula for the mixed model
    :return: repeatability, group level variance, residual variance
    """
    # block_cat only if there are at least two blocks
    lmm = smf.mixedlm(formula, data, groups=data[groups])
    lmm_fit = lmm.fit(method="nm")
    V_g = lmm_fit.cov_re.iloc[0, 0] # similar to np.var([lmm_fit.random_effects[i][0] for i in lmm_fit.random_effects.keys()])
    V_r = lmm_fit.scale # similar to np.var(lmm_fit.resid)
    repeatability = V_g/(V_g+V_r)
    return repeatability, V_g, V_r

def repeatability_t(data_t):
    step_std =np.array([data_t.query("id_cat == @i & step >= 0")["step"].var() for i in range(data_t["id_cat"].max()+1)])
    step_mean = np.array([data_t.query("id_cat == @i & step >=0")["step"].mean() for i in range(data_t["id_cat"].max()+1)])
    V_w = step_std[~np.isnan(step_std)].mean() 
    V_g = step_mean[~np.isnan(step_std)].var() 
    R=V_g/(V_g+V_w)
    print("V_g",V_g,"V_w",V_w)
    return R, V_w, V_g

def repeatability(step_list):
    weights = np.array(list(map(len,step_list)))
    weights = weights/weights.sum()
    step_std =np.array(list(map(np.var,step_list)))
    step_mean = np.array(list(map(np.mean,step_list)))
    V_w = step_std.mean() #weighted_avg_and_std(step_std, weights)[0]
    V_g = step_mean.var() #weighted_avg_and_std(step_mean, weights)[1]
    R=V_g/(V_g+V_w)
    return R, V_w, V_g

def run_repeatability():
    parameters = set_parameters()
    days_b1 = get_days(parameters, prefix=BLOCK1)
    days_b2 = get_days(parameters, prefix=BLOCK2)
    fish_keys = get_individuals_keys(parameters)
    start = 6 * 5 * 60**2
    end = 14 * 5 * 60**2
    step = 60**2 *5 # 1 hour
    rr = np.zeros((max(len(days_b1), len(days_b2)), 8+1))
    for i,(d1,d2) in list(enumerate(zip(days_b1, days_b2))):
        data1 = load_trajectory_data(parameters=parameters,fk="",day=d1)
        data2 = load_trajectory_data(parameters=parameters,fk="",day=d2)
        datadf = pd.concat((df_table(data1, BLOCK1, fish_keys), df_table(data2, BLOCK2, fish_keys)))
        for j,s in enumerate(range(start, end, step)):
            flt_time = (datadf["time"] > s) & (datadf["time"] < s+step)
            rr[i,j] = repeatability_lmm(datadf[flt_time])
        rr[i,8] = repeatability_lmm(datadf)
        #data = data1+data2
        #step_list = [data[i]["projections"][:,0] for i in range(len(data))]
        #rr[i] = repeatability(step_list)
    pd.DataFrame(rr, columns=["h%s"%r for r in range(0,9)]).to_csv(parameters.projectPath+"/repeatability_lmm_by_h.csv",sep=";")
    return rr

def get_repeatability_dxdy(parameters, data_avg_step):
    result = get_melted_table(data_avg_step)
    days_b1 = get_days(parameters=parameters,prefix=BLOCK1)
    days_b2 = get_days(parameters=parameters,prefix=BLOCK2)
    n_days = len(days_b1)
    rep_M = np.zeros((n_days, n_days))
    for i in range(n_days):
        b1d1,b2d1 = days_b1[i],days_b2[i]
        for j in range(i,n_days):
            b1d2, b2d2 = days_b1[j],days_b2[j]
            copy_results = result.query(
                "block1 == @b1d1 or block1==@b1d2 or block2 == @b2d1 or block2==@b2d2"
            ).copy().dropna()
            rep_M[i,j] = repeatability_lmm(copy_results, groups="id_cat")
    return rep_M

def repeatability_dependence_on_averaging(parameters):
    """
    first compute average step length files for different numbers of dfs
    then this function computes the repeatability for each of these files
    """
    files = glob.glob(parameters.projectPath+"/avg_step_by_*dfs.csv")
    if len(files) == 0:
        raise Exception("No files found")
    else:
        print("Found %s files" % len(files))
    reps = {}
    for file in files:
        ndfs = (int(os.path.basename(file).split("_")[-1].split("dfs")[0]))
        matrix = pd.read_csv(file,index_col=0)
        meld_matrix = get_melted_table(matrix)
        reps[ndfs]=repeatability_lmm(meld_matrix)
    return reps

def plot_repeatability_dependence_on_averaging(parameters, reps):
    fig, ax = plt.subplots()
    reps = dict(sorted(reps.items()))
    data = np.array(list(reps.values()))
    ax.plot(reps.keys(), data[:,0], label="repeatability")
    ax.plot(reps.keys(), data[:,1], label="group-level variance")
    ax.plot(reps.keys(), data[:,2], label="residual variance")
    ax.set_xscale('log')
    ax.set_ylabel("repeatability from 4 weeks")
    ax.set_xlabel("step length averaged over num of dataframes")
    ax.legend()
    fig.savefig(parameters.projectPath+"/repeatability_depending_on_averaging_cardinality.pdf")
    return fig

def plot_repeatability(df_r):
    fig, ax = plt.subplots(figsize=(7,3.5))
    x = np.arange(8)
    data = df_r.to_numpy()[:,1:]
    for i in range(data.shape[0]):
        ax.scatter((x+(i*8)), data[i,:8])
    ax.plot(range(4,data.shape[0]*8,8), data[:,8], label="daily")
    ax.set_ylabel("repeatability")
    ax.set_xlabel("hours")
    ax.legend()
    return fig

if __name__ == "__main__":
    rr = run_repeatability()
    print(rr)