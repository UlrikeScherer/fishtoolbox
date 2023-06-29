"""
Functions for exporting data to csv tables and melting them to long format.
"""
import random
import numpy as np
import pandas as pd
from .processing import load_trajectory_data
from .utils import get_days, get_individuals_keys, set_parameters, split_into_batches
from config import BLOCK1, BLOCK2

# the column DATAFRAMES indicates the time of the day in dataframes 1/5 of a second.
DATAFRAMES = "DATAFRAMES"

def step_length_avg_by_day(parameters, fish_keys):
    """
    Compute the average step length for each fish, for each day.
    Returns a dataframe with columns [BLOCK1, BLOCK2, DATAFRAMES, fish_keys...]
    """
    main_df = pd.DataFrame({BLOCK1:get_days(parameters,prefix=BLOCK1), BLOCK2:get_days(parameters, prefix=BLOCK2)})
    for fk in fish_keys:
        block = BLOCK1 if BLOCK1 in fk else BLOCK2
        data = load_trajectory_data(parameters, fk)
        step_df = pd.DataFrame({
                fk:[data[i]["projections"][:,0].mean() for i in range(len(data))], 
                block:[data[i]["day"][0][0] for i in range(len(data))]
                               }).dropna()
        main_df = main_df.merge(step_df,on=[block], how="left")
    return main_df

def step_length_avg(parameters, fish_keys, batch_dfs, num_samples_per_day=np.inf, N_HOURS=8):
    """
    Compute the average step length for each fish, for each day and batch size. If num_samples_per_day is not np.inf, then we will sample that many intervals for each day.
    N_HOURS is the number of tracked hours in a day, in the original experiment this is 8.
    Returns a dataframe with columns [BLOCK1, BLOCK2, DATAFRAMES, fish_keys...]
    """
    days_b1, days_b2 = get_days(parameters,prefix=BLOCK1), get_days(parameters,prefix=BLOCK2)
    DFsByDay = N_HOURS*60*60*5 
    hidx = np.concatenate([
        [(d1,d2,dfidx) for dfidx in random.sample(range(0,DFsByDay, batch_dfs), k=min(DFsByDay//batch_dfs, num_samples_per_day))] 
                            for (d1,d2) in zip(days_b1, days_b2)])
    main_df = pd.DataFrame(hidx, columns=[BLOCK1,BLOCK2,DATAFRAMES])
    main_df[DATAFRAMES] = main_df[DATAFRAMES].astype(int)
    main_df.sort_values(inplace=True, by=[BLOCK1,DATAFRAMES])
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
            DATAFRAMES:step_means[:,2]
                            }).dropna()
        step_df[DATAFRAMES] = step_df[DATAFRAMES].astype(int)
        main_df = main_df.merge(step_df,on=[block,DATAFRAMES], how="left")
    return main_df

def map_to_batch_dfs(batch: int, num_dfs_per_batch: int):
    """
    Maps a batch to the first dataframe of the batch
    batch: list of dataframes (int)
    num_dfs_per_batch: number of dataframes per batch (int)
    """
    if len(batch)==0: return np.nan
    else: return int((batch[0]//(num_dfs_per_batch))*num_dfs_per_batch - (6*60*60*5)) # experiment starts at 6am

def df_table(data,block, fish_keys):
    """
    data: list of dictionaries with keys "projections" and "df_time_index" (from load_trajectory_data)
    block: string
    fish_keys: list of strings
    returns a dataframe with columns [step, time, fish_key]
    """
    return pd.concat([pd.DataFrame({"step":d["projections"][:,0],"time":d["df_time_index"].flatten()}) for d in data],keys=[fish_keys.index(block+"_"+d["fish_key"][0][0]) for d in data]).reset_index()

def get_melted_table(df):
    """
    df: dataframe with columns [BLOCK1, BLOCK2, DATAFRAMES, fish_keys...]
    returns a melted dataframe with columns [BLOCK1, BLOCK2, DATAFRAMES, id, step, block, cam_id, pos, id_cat, block_cat, day, minutes]
    """
    melted = pd.melt(df, id_vars=[BLOCK1,BLOCK2,DATAFRAMES],value_vars=df.columns[3:], var_name="id", value_name="step")
    split_cols = melted['id'].str.split('_', expand=True)
    split_cols.columns = ['block', 'cam_id', 'pos']
    split_cols['id_cat'] = melted['id'].astype('category').cat.codes
    #combine the melted dataframe with the split columns
    result = pd.concat([melted, split_cols], axis=1)
    result['block_cat'] = result['block'].astype('category').cat.codes
    result["day"]= result["block1"].astype("category").cat.codes
    result["minutes"] = result[DATAFRAMES] // (60*5)
    return result.dropna()


if __name__ == "__main__":
    parameters = set_parameters()
    FISH_KEYS = get_individuals_keys(parameters)
    df = step_length_avg_by_day(parameters, FISH_KEYS)
    df.to_csv(parameters.projectPath+"/step_length_avg_by_day.csv", index=False)
    batch_dfs = 60*5 # one minute
    df = step_length_avg(parameters, FISH_KEYS, batch_dfs)
    df.to_csv(parameters.projectPath+"/step_length_avg_by_minute.csv", index=False)
