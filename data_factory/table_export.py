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

def load_entropy_data(
    parameters, 
    time_range = ['daily', 'hourly'],
    cluster_sizes = ['005', '007', '010', '020', '050']
):
    entropy_data_dict = {}
    for time in time_range:
        for size in cluster_sizes:
            file_name = parameters.projectPath+f'/plasticity/cluster_entropy_wshed/cluster_entropy_wshed_{time}_{size}.csv'
            entropy_data_dict[f'{time}_{size}'] = pd.read_csv(file_name)
    return entropy_data_dict


def load_coefficient_of_variation_data(
    parameters,
    time_range = ['daily', 'hourly'],
    accuracies = ['050'],
    principal_components = ['distance to wall', 'turning angle', 'step length']
):
    cv_data_dict = {}
    for time in time_range:
        for accuracy in accuracies:
            for mode in principal_components:
                file_name = parameters.projectPath+f'/plasticity/cv/cv_{mode}_{time}_ndf_{accuracy}.csv'
                if mode == 'distance to wall':
                    mode_shrtnd = 'd2w'
                elif mode == 'turning angle':
                    mode_shrtnd = 'angle'
                elif mode == 'step length':
                    mode_shrtnd = 'step'
                cv_data_dict[f'{time}_{accuracy}_{mode_shrtnd}'] = pd.read_csv(file_name)
    return cv_data_dict


def build_singular_ids_table(
    input_data_dict,
    fish_keys,
    time_constraint,
    measure = 'entropy',
    modes = ['005', '007', '010', '020', '050'],
    cov_accuracy = '050',
):
    output_id_dict = {}
    for id in fish_keys:
        # cv values
        intermediate_dict = {}
        for mode in modes:
            if (measure == 'coefficient of variation') or (measure == 'cov'):
                identifier_str = f'{time_constraint}_{cov_accuracy}_{mode}'
                column_identifier_str = f'{mode}'
            elif measure == 'entropy':
                identifier_str = f'{time_constraint}_{mode}'
                column_identifier_str = f'entropy_{mode}'
            mode_df = input_data_dict[f'{identifier_str}'][[id, 'Unnamed: 0']]
            mode_df = mode_df.rename(columns={'Unnamed: 0': 'timestep', f'{id}': f'{column_identifier_str}'})
            mode_df['id'] = id
            intermediate_dict[f'{mode}'] = mode_df
        output_id_dict[f'{id}'] = intermediate_dict
    return output_id_dict


def merge_cov_and_entropy_dicts_to_one_df(
    cov_dict,
    entropy_dict,
    fish_keys,
    time_constraint = 'daily',
    cov_modes = ['d2w', 'angle', 'step'],
    cov_accuracy = '050',
    entropy_modes = ['005', '007', '010', '020', '050'],
    output_file_name = None
):
    
    cv_dfs_list = []
    entropy_dfs_list = []
    
    for id in fish_keys:
        # cov values
        cv_dict = cov_dict[id]
        cv_df = pd.merge(
            pd.merge(
                cv_dict[cov_modes[0]], cv_dict[cov_modes[1]], 
                on=['timestep', 'id']
            ), cv_dict[cov_modes[2]],
            on=['timestep', 'id']
        )
        cv_dfs_list.append(cv_df)
        
        # entropy values
        entropy_specified_dict = entropy_dict[id]
        if len(entropy_modes) == 5:
            entropy_df = pd.merge(
                pd.merge(
                    pd.merge(
                        pd.merge(
                            entropy_specified_dict[entropy_modes[0]], entropy_specified_dict[entropy_modes[1]],
                            on=['timestep', 'id']
                        ), entropy_specified_dict[entropy_modes[2]],
                        on=['timestep', 'id']
                    ), entropy_specified_dict[entropy_modes[3]],
                    on=['timestep', 'id']
                ), entropy_specified_dict[entropy_modes[4]],
                on=['timestep', 'id']
            )
        elif len(entropy_modes) == 4:
            entropy_df = pd.merge(
                pd.merge(
                    pd.merge(
                        entropy_specified_dict[entropy_modes[0]], entropy_specified_dict[entropy_modes[1]],
                        on=['timestep', 'id']
                    ), entropy_specified_dict[entropy_modes[2]],
                    on=['timestep', 'id']
                ), entropy_specified_dict[entropy_modes[3]],
                on=['timestep', 'id']
            )
        elif len(entropy_modes) == 3:
            entropy_df = pd.merge(
                pd.merge(
                    entropy_specified_dict[entropy_modes[0]], entropy_specified_dict[entropy_modes[1]],
                    on=['timestep', 'id']
                ), entropy_specified_dict[entropy_modes[3]],
                on=['timestep', 'id']
            )
        elif len(entropy_modes) == 2:
            entropy_df = pd.merge(
                entropy_specified_dict[entropy_modes[0]], entropy_specified_dict[entropy_modes[1]],
                on=['timestep', 'id']
            )
        entropy_dfs_list.append(entropy_df)

    cv_all_df = pd.concat(cv_dfs_list)
    entropy_all_df = pd.concat(entropy_dfs_list)

    cols = ['timestep', 'id']
    all_df = pd.merge(cv_all_df, entropy_all_df, on=cols)
    if len(entropy_modes) == 5:
        all_df_reordered = all_df[['timestep', 'id', cov_modes[0], cov_modes[1], cov_modes[2], f'entropy_{entropy_modes[0]}', f'entropy_{entropy_modes[1]}', f'entropy_{entropy_modes[2]}', f'entropy_{entropy_modes[3]}', f'entropy_{entropy_modes[4]}']]
    elif len(entropy_modes) == 4:
        all_df_reordered = all_df[['timestep', 'id', cov_modes[0], cov_modes[1], cov_modes[2], f'entropy_{entropy_modes[0]}', f'entropy_{entropy_modes[1]}', f'entropy_{entropy_modes[2]}', f'entropy_{entropy_modes[3]}']]
    elif len(entropy_modes) == 3:
        all_df_reordered = all_df[['timestep', 'id', cov_modes[0], cov_modes[1], cov_modes[2], f'entropy_{entropy_modes[0]}', f'entropy_{entropy_modes[1]}', f'entropy_{entropy_modes[2]}']]
    elif len(entropy_modes) == 2:
        all_df_reordered = all_df[['timestep', 'id', cov_modes[0], cov_modes[1], cov_modes[2], f'entropy_{entropy_modes[0]}', f'entropy_{entropy_modes[1]}']]
    if output_file_name is not None: 
        all_df_reordered.to_csv(output_file_name)
    return all_df_reordered


def build_and_unify_cov_and_entropy_tables(
    parameters,
    fish_keys,
    time_range_extraction= ['daily', 'hourly'],
    time_constraint = 'daily',
    cov_principal_components = ['distance to wall', 'turning angle', 'step length'],
    cov_accuracies = ['050'],
    cluster_sizes = ['005', '007', '010', '020', '050'],
    output_file_name = None
):

    cov_data_dict = load_coefficient_of_variation_data(
        parameters,
        time_range = time_range_extraction,
        accuracies = cov_accuracies,
        principal_components = cov_principal_components
    )

    entropy_data_dict = load_entropy_data(
        parameters, 
        time_range = time_range_extraction,
        cluster_sizes = cluster_sizes
    )

    cov_principal_components = [sub.replace('distance to wall', 'd2w') for sub in cov_principal_components]
    cov_principal_components = [sub.replace('turning angle', 'angle') for sub in cov_principal_components]
    cov_principal_components = [sub.replace('step length', 'step') for sub in cov_principal_components]

    cov_id_features_dict = build_singular_ids_table(
        cov_data_dict,
        fish_keys,
        time_constraint,
        measure = 'cov',
        modes = cov_principal_components,
        cov_accuracy = cov_accuracies[0],
    )

    entropy_id_features_dict = build_singular_ids_table(
        entropy_data_dict,
        fish_keys,
        time_constraint,
        measure = 'entropy',
        modes = cluster_sizes,
    )

    output_df = merge_cov_and_entropy_dicts_to_one_df(
        cov_id_features_dict,
        entropy_id_features_dict,
        fish_keys,
        time_constraint = time_constraint,
        cov_modes = cov_principal_components,
        cov_accuracy = cov_accuracies[0],
        entropy_modes = cluster_sizes,
        output_file_name = output_file_name
    )
    return output_df

if __name__ == "__main__":
    parameters = set_parameters()
    FISH_KEYS = get_individuals_keys(parameters)
    df = step_length_avg_by_day(parameters, FISH_KEYS)
    df.to_csv(parameters.projectPath+"/step_length_avg_by_day.csv", index=False)
    batch_dfs = 60*5 # one minute
    df = step_length_avg(parameters, FISH_KEYS, batch_dfs)
    df.to_csv(parameters.projectPath+"/step_length_avg_by_minute.csv", index=False)
