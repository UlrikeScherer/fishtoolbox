
import os
from collections import Counter
import random
from random import sample
from time import gmtime, strftime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.pyplot import cm
import hdf5storage
import pandas as pd
import datashader as ds
from datashader.mpl_ext import dsshow
import motionmapperpy as mmpy
from clustering.clustering import boxplot_characteristics_of_cluster
from config import BLOCK
from data_factory.plot_helpers import remove_spines
from .processing import get_regions_for_fish_key, load_zVals_concat, load_clusters_concat, load_summerized_data, return_normalization_func
from .utils import pointsInCircum, get_individuals_keys, get_days
from clustering.transitions_cluster import transition_rates, draw_transition_graph

def plot_feature_distribution(input_data, name="distributions_step_angle_dist", n_dfs=[1], names = ["step length", "turning angle", "distance to the wall"]):
    n_rows, n_cols = len(n_dfs),len(names)
    fig = plt.figure(figsize=(n_cols*4,n_rows*4))
    axes = fig.subplots(n_rows,n_cols, sharex="col", squeeze=False)
    for axes_i, n_df in zip(axes,n_dfs):
        for i, ax in enumerate(axes_i):
            if i==0: 
                ax.set_ylabel("counts")#'%0.1f sec'%(n_df/5))
            # Draw the plot
            data_in = input_data[:,i]
            new_len = data_in.size//n_df
            data_in = data_in[:n_df*new_len].reshape((new_len, n_df)).mean(axis=1)
            ax.hist(data_in, bins = 50,density=False,stacked=False, range=(0,8) if i==0 else None,
                     color = '#6161b0', edgecolor='#6161b0', log=True)
            # Title and labels
            ax.set_title("$\mu=%.02f$, $\sigma=%.02f$"%(data_in.mean(), data_in.std()), size = 12)
            ax.set_xlabel(names[i], size = 12)
            remove_spines(ax)
    #plt.tight_layout()
    fig.savefig(f"{name}.pdf")
    return fig

def plot_lines_for_cluster2(
    positions,
    projections,
    area,
    clusters,
    n_clusters,
    limit=20,
    fig_name="cluster_characteristics.pdf",
):
    nrows = 2
    fig, axs = plt.subplots(
        nrows=nrows, ncols=n_clusters, figsize=(n_clusters * 4, nrows * 4), sharey="row"
    )
    projections_norm = projections / np.std(np.abs(projections),axis=0)
    for cluster_id in range(1,n_clusters+1):
        ax_b = axs[1, cluster_id-1]
        ax = axs[0, cluster_id-1]
        boxplot_characteristics_of_cluster(projections_norm[clusters == cluster_id], ax_b, metric_names=["speed", " turning angle", "wall distance"])
        samples_c_idx = np.where(clusters == cluster_id)[0]
        cluster_share = samples_c_idx.shape[0] / projections.shape[0]
        select = sample(range(len(samples_c_idx)), k=limit)
        plot_lines_select(
            positions,
            samples_c_idx[select],
            area, 
            ax=ax,
            title="cluster: %d,      share: %.3f" % (cluster_id, cluster_share),
        )
        ax_b.yaxis.set_tick_params(which="both", labelbottom=True)

    fig.savefig(fig_name, bbox_inches="tight")
    plt.close(fig)
    
def plot_lines_select(positions, samples_idx, area, ax, title):
    ax.set_title(title)
    plot_area(area, ax)
    for idx in samples_idx:
        s,t = max(idx-50,0),min(idx+50, len(positions)-1)
        ax.plot(positions[s:t, 0], positions[s:t, 1])

def plot_area(area_box, ax):
    area_box = np.concatenate((area_box, [area_box[0]]))
    ax.plot(*area_box.T)
    
    
def ethogram_of_clusters(parameters, clusters, start_time=0, end_time=8*(60**2)*5, fish_key="", day="",rows=4, write_fig=False, name_append=""):
    # f2min = (60**2)*5 # conversion factor for hours of end_time to the respective data_points 
    wregs = clusters[start_time:end_time]
    len_half = wregs.shape[0]//rows
    step = len_half//10
    ethogram = np.zeros((wregs.max(), len(wregs)))

    for wreg in range(1, wregs.max()+1):
        ethogram[wreg-1, np.where(wregs==wreg)[0]] = 1.0

    ethogram = np.split(ethogram.T, np.array([len_half*i for i in range(1,rows)]))

    fig, axes = plt.subplots(rows, 1, figsize=(30,3*rows))
    if rows == 1:
        axes = np.array([axes])
    axes[0].set_title(f"{fish_key} {day}")
    for k, (e, ax) in enumerate(zip(ethogram, axes.flatten())):
        ax.imshow(e.T, aspect='auto', cmap=mmpy.gencmap())
        ax.set_yticks([i for i in range(0, wregs.max(), 1)])
        ax.set_yticklabels(['Region %i'%(j+1) for j in range(0, wregs.max(), 1)])
        xticklocs = [i for i in range(0,len_half, step)]
        ax.set_xticks(xticklocs)
        ax.set_xticklabels([strftime("%H:%M", gmtime((j+(k*len_half))//5)) for j in xticklocs])

    ax.set_xlabel('Time (H:M)')
    if write_fig:
        path_e = f"{parameters.projectPath}/{BLOCK}_ethograms"
        if not os.path.exists(path_e):
            os.mkdir(path_e)
        fig.savefig(f"{path_e}/ethogram_{name_append}{fish_key}_{day}.pdf")
    return fig

def plot_transition_rates(clusters, filename, cluster_remap=[(0,1)]):
    if cluster_remap:
        for c1,c2 in cluster_remap:
            clusters[np.where(clusters==c1)]=c2
    t = transition_rates(clusters)
    n_clusters = t.index.size
    G = draw_transition_graph(t, n_clusters, pointsInCircum(1,n_clusters), 
                              output=filename, 
                              cmap=get_color_map(n_clusters))
    return G

def create_ethogram_for_individual_colormap_changes(
        parameters, 
        high_entropy_ind_dict: dict, 
        low_entropy_ind_dict: dict, 
        start_time: int, 
        end_time: int, 
        cluster_number: int, 
        fig_parent_dir: str
    ):
    ''' 
    Creation of an ethogrm with individual colormap changes, 
    the maximum number of cluster occurrences for one individual
    sets the maximum color-value in a customized colormap

    high_entropy_ind_dict = {'block1_23520289_front': {'day4': '20210914_060000', 'day5': '20210915_060000'}, 
                    'block1_23520270_front': {'day4': '20210914_060000', 'day5': '20210915_060000'}, 
                    'block2_23520278_front': {'day4': '20211105_060000', 'day5': '20211106_060000'}}
    low_entropy_ind_dict = { 'block2_23484201_back': {'day4': '20211105_060000', 'day5': '20211106_060000'},
                    'block2_23520266_back': {'day4': '20211105_060000', 'day5': '20211106_060000'},
                    'block1_23520264_back': {'day4': '20210914_060000', 'day5': '20210915_060000'}}
    '''
    ind_xy = {}
    occ_list = []
    # for all individuals in dicts: get xy values, get highest occurrence per y-list
    for ind in list(low_entropy_ind_dict.keys()) + list(high_entropy_ind_dict.keys()):
        print(ind)
        x,y, wregs = get_xy_for_individual_per_timerange(parameters, ind, start_time = start_time, end_time = end_time, cluster_number = cluster_number)
        highest_occ = get_highest_occ_p_y_list(y)
        ind_xy[str(ind)] = x,y
        occ_list.append(highest_occ)

    overall_highest_occ = max(occ_list)
    adjusted_colormap = get_discr_colormap_for_max_occ(overall_highest_occ)
    ethogram_dict = {}
    # for all ind: generate ethogram with color gradient
    for ind in list(low_entropy_ind_dict.keys()) + list(high_entropy_ind_dict.keys()):
        print(f'creating ethogram for individual {ind}')
        fig_name = f'ethogram_entropy_{ind}.pdf'
        y_occ_dict = {key: 0 for key in range(-1, 21, 1)}

        x = ind_xy[str(ind)][0]
        y = ind_xy[str(ind)][1]
        # Create a line plot with each data point colored differently
        fig, ax = plt.subplots(figsize=(400, 30))
        ax.tick_params(axis='both', labelsize='large')
        for i in range(0,len(x)):
            y_occ_dict[y[i]] += 1
            ax.plot(x[i], y[i], '|', markersize = 70, color=adjusted_colormap[y_occ_dict[y[i]]], label=f'Data Point {i+1}')
        max_region_number = wregs.max() + 1
        x_limit = end_time - start_time
        x_tick_label_count = 18000
        xticklocs = [i for i in range(0,len(x), x_tick_label_count)]
        xticklabels = [int(i/x_tick_label_count) for i in range(0,len(x), x_tick_label_count)]
        yticklocs = [i for i in range(1, max_region_number, 1)]
        yticklabels = [f'Region {i}' for i in range(1, max_region_number, 1)]
        ax.set_xticks(xticklocs, xticklabels, fontsize = 70)
        ax.set_yticks(yticklocs, yticklabels, fontsize= 70)
        ax.set_ylim(0, max_region_number) 
        ax.set_xlim(0, x_limit)
        ax.invert_yaxis()
        fig_path = os.path.join(
            fig_parent_dir,
            fig_name
        )
        fig.savefig(fig_path)
        print(f'\tfigure saved in: {fig_path}')
        ethogram_dict[str(ind)] = fig,ax
    return ethogram_dict

def get_xy_for_individual_per_timerange(
        parameters, 
        individual: str,
        start_time: int, 
        end_time: int, 
        cluster_number: int
        ):
    ''' calculates the respective x and y values for a specific individual,
    which correlates to the respective string representation of its fish_key.
    Returns the x and y values, as well as the wreg-array.
    '''
    individual = individual
    clusters = load_clusters_concat(parameters, individual, k= cluster_number)
    start_time = start_time
    end_time = end_time
    rows = 1
    wregs = clusters[start_time:end_time]
    len_half = wregs.shape[0]//rows
    ethogram = np.zeros((wregs.max(), len(wregs)))
    for wreg in range(1, wregs.max()+1):
        ethogram[wreg-1, np.where(wregs==wreg)[0]] = 1.0
    ethogram = np.split(ethogram.T, np.array([len_half*i for i in range(1,rows)]))
    x = []
    y = []
    for timestep, row in enumerate(list(ethogram[0])):
        # Find the index of the 1-hot encoding (value of 1)
        exists = np.any(row == 1)
        if exists:
            index = np.where(row == 1)[0][0]
            x.append(timestep)
            y.append(index+1)
        else:
            index = -1
    return x,y, wregs


def get_highest_occ_p_y_list(y: list)->int:
    ''' returns the highest number of cluster-occurrences per individual
    overloaded by the y-list of an individual
    '''
    element_counts = Counter(y)
    most_common = element_counts.most_common(1)[0]  
    most_common_element, most_common_count = most_common
    print(f"Most common element: {most_common_element}, Count: {most_common_count}")
    return most_common_count


def get_discr_colormap_for_max_occ(max_occ: int):
    ''' returns an adjusted colormap with color changes based on 
    an interval of `n` discrete values from 0 to 1, 
    `n` relates directly with `max_occ + 1`
    '''
    colors = plt.cm.viridis(np.linspace(0, 1, max_occ + 1))
    return colors


def load_watershed_file(wshed_path) -> dict:
    wshed_dict = hdf5storage.loadmat(wshed_path)
    return wshed_dict


def get_umap_density_figure(
        umap_embedding, 
        extent_factor, 
        axis_limit_tuple = ([-100, 100], [-100, 100]),
        overloaded_figure = None, 
        include_axis_visualization = False, 
        cmap = 'default',
        save_pdf_path: os.path = None,
        plot_figure = False
    ) -> plt.figure:
    '''
    creates a density map from the umap embedding, stored in the watershed file
    '''
    if overloaded_figure:
        fig, ax = overloaded_figure
    else: 
        fig, ax = plt.subplots()

    if cmap == 'default':
        cmap = mmpy.gencmap()
    
    # TODO: make cmap generation agnostic
    # Create a sample data range from 0 to 1
    data_range = np.linspace(0, 1, 256)

    # adjustment of the original colormap within the predefined bounds
    my_YlOrBr = plt.cm.YlOrBr(data_range)

    # Define the subset of colors you want
    start_index = 0  
    end_index = 100  

    # Extract the subset of colors
    subset_colors = my_YlOrBr[start_index:end_index]

    # Create a custom colormap using the subset of colors
    custom_cmap = mcolors.ListedColormap(subset_colors)
    custom_cmap.set_under('white')

    ax.imshow(
        X= umap_embedding,
        extent=(-extent_factor, extent_factor, -extent_factor, extent_factor), 
        origin='lower', 
        cmap=custom_cmap,
        vmin = 0.000005,
        vmax = 0.01290888073184956
    )
    ax.set_xlim(axis_limit_tuple[0])
    ax.set_ylim(axis_limit_tuple[1])
    if include_axis_visualization:
        ax.axis('on')
    else:
        ax.axis('off')
    if save_pdf_path is not None:
        fig.savefig(
            fname = save_pdf_path, 
            format = 'pdf'
        )
    if plot_figure:
        fig.show()
    return fig,ax


def get_watershed_boundaries_figure(
        boundaries_embedding, 
        extent_factor, 
        original_figure_width= 611, 
        axis_limit_tuple = ([-100, 100], [-100, 100]),
        overloaded_figure=None, 
        include_axis_visualization = False, 
        plot_figure = False
    ) -> plt.figure:
    '''
    creates a figure for the watershed boundaries by norming the its own figure sizes with 
    the figure sizes of the umap-trajectories. Please note the equation for the norming written below: \\
    new_boundary  = old_boundary / (old_resolution/ 2*extent_factor) - extent_factor \\
              = old_boundary / (old_resolution/ 2* half_extent_figure_size) - centering \\
              = old_boundary / scaling_factor - centering
    '''
    bounds_aug_x_new = ((boundaries_embedding[0][0]
                     / (original_figure_width/(2*extent_factor))) 
                     - extent_factor
                    )
    bounds_aug_y_new = ((boundaries_embedding[0][1]
                     / (original_figure_width/(2*extent_factor))) 
                     - extent_factor
                    )
    
    if overloaded_figure:
        fig, ax = overloaded_figure
    else: 
        fig, ax = plt.subplots()

    ax.scatter(
        x=bounds_aug_x_new, 
        y=bounds_aug_y_new, 
        color='gray', 
        s=0.05
    )
    ax.set_xlim(axis_limit_tuple[0])
    ax.set_ylim(axis_limit_tuple[1])
    if include_axis_visualization:
        ax.axis('on')
    else:
        ax.axis('off')
    if plot_figure:
        fig.show()
    return fig,ax


def get_watershed_clusters_figure(
        cluster_embeddings, 
        extent_factor, 
        original_figure_width = 611, 
        axis_limit_tuple = ([-100, 100], [-100, 100]),
        overloaded_figure=None, 
        include_axis_visualization= False, 
        cmap = 'default',
        save_pdf_path: os.path = None,
        plot_figure = False
    ) -> plt.figure:
    '''
    creates a figure with the watershed clusters and its respective cluster IDs, 
    normed for the figure size of the umap-trajectories
    '''
    if overloaded_figure:
        fig, ax = overloaded_figure
    else: 
        fig, ax = plt.subplots()

    # use np.nan for 0-values for clear distinction between embedding points and background
    cluster_embeddings = cluster_embeddings.astype(np.float64)
    cluster_embeddings[cluster_embeddings == 0] = np.nan

    if cmap == 'default':
        cmap = mmpy.gencmap()
    ax.imshow(
        X=cluster_embeddings, 
        extent=(-extent_factor, extent_factor, -extent_factor, extent_factor), 
        origin='lower', 
        cmap=cmap
    )

    for i in np.unique(cluster_embeddings)[1:]:
        fontsize = 8
        xinds, yinds = np.where(cluster_embeddings == i)
        xinds_new = ((xinds
                    / (original_figure_width/(2*extent_factor))) 
                    - extent_factor
        )
        yinds_new = ((yinds
                    / (original_figure_width/(2*extent_factor))) 
                    - extent_factor
        )
        
        x_text_location = np.mean(yinds_new)
        y_text_location = np.mean(xinds_new)
        # prevent cluster id text to be out of bounds of axis limits
        if not (x_text_location < axis_limit_tuple[0][0] 
            or x_text_location > axis_limit_tuple[0][1]
            or y_text_location < axis_limit_tuple[1][0]
            or y_text_location > axis_limit_tuple[1][1]
            or np.isnan(i)
        ):  
            ax.text(x_text_location, y_text_location, str(i.astype(int)), fontsize=fontsize, fontweight='bold')
    ax.set_xlim(axis_limit_tuple[0])
    ax.set_ylim(axis_limit_tuple[1])
    if include_axis_visualization:
        ax.axis('on')
    else:
        ax.axis('off')
    if save_pdf_path is not None:
        fig.savefig(
            fname = save_pdf_path, 
            format = 'pdf'
        )
    if plot_figure:
        fig.show()
    return fig, ax


def get_umap_trajectories_figure(
        parameters, 
        fish_key, 
        day, 
        figure_color= 'red',
        data_restriction = None,
        data_restriction_additional = None,
        axis_limit_tuple = ([-100, 100], [-100, 100]),
        alpha_transparency = None,
        overloaded_figure = None, 
        include_axis_visualization = False, 
        plot_figure = False
    ) -> plt.figure:
    '''
    creates a figure of the umap trajectories for a specific `fish_key` and `day`. 
    restricting the number of trajectories is possible by setting a numerical limit 
    for `data_restriction_limit`, if there is None specified, then all trajectories 
    for the respective day are plotted. 

    '''

    zVals = load_zVals_concat(
        parameters= parameters,
        fk= fish_key,
        day= day
    )['embeddings']
    if data_restriction is not None:
        if 'limit' in data_restriction:
            zVals = zVals[0:data_restriction['limit']]
        elif 'nth_value' in data_restriction:
            zVals = zVals[0::data_restriction['nth_value']]
        elif 'interval' in data_restriction:
            zVals = zVals[data_restriction['interval'][0]:data_restriction['interval'][1]]
        if data_restriction_additional:
            if 'limit' in data_restriction_additional:
                zVals = zVals[0:data_restriction_additional['limit']]
            elif 'nth_value' in data_restriction_additional:
                zVals = zVals[0::data_restriction_additional['nth_value']]
            elif 'interval' in data_restriction_additional:
                zVals = zVals[data_restriction_additional['interval'][0]:data_restriction_additional['interval'][1]]
    
    if overloaded_figure:
        fig, ax = overloaded_figure
    else: 
        fig, ax = plt.subplots()
        
    ax.plot(
        zVals[:,0], zVals[:,1],
        color=figure_color,
        alpha = alpha_transparency,
        solid_capstyle = "butt",
        linewidth = 1
    )

    ax.set_xlim(axis_limit_tuple[0])
    ax.set_ylim(axis_limit_tuple[1])
    if include_axis_visualization:
        ax.axis('on')
    else:
        ax.axis('off')
    if plot_figure:
        fig.show()
    return fig, ax


def get_umap_scatter_figure_per_fk_day(
        parameters, 
        fish_key, 
        day, 
        point_size = 15,
        alpha_transparency = 0.5,
        figure_color= 'red',
        data_restriction = None, 
        axis_limit_tuple = ([-100, 100], [-100, 100]),
        overloaded_figure=None, 
        include_axis_visualization = False, 
        plot_figure = False
    ) -> plt.figure:
    ''' scatter umap-trajectories data points for one fishkey for one day
        by following a `data_restriction` rule, the following rules 
        are supported and have to be formulated in a dictionary
        >>> data_restriction: {'limit': 200} or {'nth_value': 50} or None
        the arguments `point_size` and `alpha_transparency` directly
        affect the visual appearance
        returns a figure and axis, and the respective x and y values each as a list
    '''
    zVals = load_zVals_concat(
        parameters= parameters,
        fk= fish_key,
        day= day
    )['embeddings']
    
    if data_restriction is not None:
        if 'limit' in data_restriction:
            zVals = zVals[0:data_restriction['limit']]
        elif 'nth_value' in data_restriction:
            zVals = zVals[0::data_restriction['nth_value']]

    
    if overloaded_figure:
        fig, ax = overloaded_figure
    else: 
        fig, ax = plt.subplots()
        
    x, y = zVals[:,0], zVals[:,1] 
    ax.scatter(
        x = zVals[:,0], 
        y = zVals[:,1], 
        s = point_size,
        alpha = alpha_transparency,
        c = figure_color
    )
    ax.set_xlim(axis_limit_tuple[0])
    ax.set_ylim(axis_limit_tuple[1])
    if include_axis_visualization:
        ax.axis('on')
    else:
        ax.axis('off')
    if plot_figure:
        fig.show()
    return fig, ax, x,y


def umap_scatter_figure_for_all_individuals_sorted(
        parameters, 
        distinct_individuals = False,
        single_plot_for_every_individual = False,
        point_size = 15,
        alpha_transparency = 0.5,
        figure_color= 'red',
        data_restriction = None,
        data_restriction_days = None, 
        elements_restriction = None,
        axis_limit_tuple = ([-100, 100], [-100, 100]),
        overloaded_figure=None, 
        include_axis_visualization = False, 
        scatter_with_density = False,
        save_pdf_path: os.path = None,
        plot_figure = False
    ) -> plt.figure:
    '''
    scatter umap-trajectory points for the whole dataset of all fish_keys and day
    by following a `data_restriction` rule, the following rules 
    are supported and have to be formulated in a dictionary
    >>> data_restriction: {'limit': 200} or {'nth_value': 50}
    If the number of plotted days should be limited to a specific time-range, this can be
    specified using the parameter (e.g. for the second week, day 6-10)
    >>> data_restriction_days: ['20210916_060000', '20210917_060000', '20210918_060000', '20210919_060000',
    '20210920_060000','20211102_060000', '20211103_060000', '20211104_060000', '20211105_060000', '20211106_060000']
    the arguments `point_size` and `alpha_transparency` directly
    affect the visual appearance.
    By using an `elements_restriction`, it is possible to only use a certain number 
    of samples instead of the whole dataset.
    By using distinct_individuals, every present individual will be plotted with an own color, 
    resulting in a scatter plot with possibilities to differentiate individuals.
    '''

    if single_plot_for_every_individual:
        individual_plot_dict = {}

    if overloaded_figure:
        fig, ax = overloaded_figure
    else: 
        fig, ax = plt.subplots()
        
    fishkey_list = get_individuals_keys(parameters= parameters)
    days_list = get_days(parameters= parameters)

    elements_counter = 0
    if distinct_individuals:
        color = iter(cm.tab20c(np.linspace(0, 1, len(fishkey_list))))
    for fk in fishkey_list:
        if distinct_individuals:
            figure_color = next(color)
        if single_plot_for_every_individual:
            fig, ax = plt.subplots()
        if scatter_with_density:
            x_list, y_list = [], []

        for day in days_list:
            if data_restriction_days is not None:
                if day not in data_restriction_days:
                    continue
            zVal_path = parameters.projectPath+f'/Projections/{fk}_{day}_pcaModes_uVals.mat'
            if os.path.exists(zVal_path):
                elements_counter += 1
                if (elements_restriction is not None) and (elements_counter > elements_restriction):
                    return (fig, ax)
                if scatter_with_density:
                    _fig, _ax, xVals, yVals = get_umap_scatter_figure_per_fk_day(
                        parameters= parameters,
                        fish_key= fk,
                        day= day,
                        point_size= point_size,
                        alpha_transparency= alpha_transparency,
                        figure_color= figure_color,
                        data_restriction= data_restriction,
                    )
                    x_list.append(xVals)
                    y_list.append(yVals)

                else: 
                    fig, ax, _x, _y= get_umap_scatter_figure_per_fk_day(
                        parameters= parameters,
                        fish_key= fk,
                        day= day,
                        point_size= point_size,
                        alpha_transparency= alpha_transparency,
                        figure_color= figure_color,
                        data_restriction= data_restriction,
                        overloaded_figure = (fig, ax)
                    )
        if scatter_with_density:
            fig, ax = plt.subplots()
            df = pd.DataFrame(
                dict(
                    x=x_list, 
                    y=y_list
                )
            )
            dsartist = dsshow(
                df,
                ds.Point("x", "y"),
                ds.count(),
                vmin=0,
                vmax=35,
                norm="linear",
                aspect="auto",
                ax=ax,
            )

        if single_plot_for_every_individual:
            individual_plot_dict[str(fk)] = [fig, ax]
    ax.set_xlim(axis_limit_tuple[0])
    ax.set_ylim(axis_limit_tuple[1])
    if include_axis_visualization:
        ax.axis('on')
    else:
        ax.axis('off')
    if save_pdf_path is not None:
        fig.savefig(
            fname = save_pdf_path, 
            format = 'pdf'
        )
    if plot_figure:
        fig.show()
    if single_plot_for_every_individual:
        return individual_plot_dict
    return fig, ax


def umap_scatter_figure_for_total_population(
        parameters, 
        point_size = 0.0001,
        alpha_transparency = 1,
        data_restriction = None,
        cmap = plt.cm.gist_ncar,
        axis_limit_tuple = ([-50, 75], [-50, 75]),
        overloaded_figure=None, 
        include_axis_visualization = False, 
        save_pdf_path: os.path = None,
        plot_figure = False
    ) -> plt.figure:
    '''
    creates a scatter umap plot for the whole population with a randomized order 
    of the sorted fishkeys. Results in a more heterogenous distribution of datapoints
    without correlations of scatter-plotted region and the individuals entropy-value
    color identifyable with its color-coding. 
    Downsampling the number of datapoints is highly recommended to enhance runtime 
    '''
    print('run calculations for subsampling value: ', data_restriction['nth_value'])
    print('get all embeddings')
    fk_content_dict = {}
    fishkey_list = get_individuals_keys(parameters= parameters)
    days_list = get_days(parameters= parameters)
    random.shuffle(fishkey_list)
    for fk in tqdm(fishkey_list):
        zVals_combined_list = []
        for day in days_list:
            zVal_path = parameters.projectPath+f'/Projections/{fk}_{day}_pcaModes_uVals.mat'
            if os.path.exists(zVal_path):
                zVals = load_zVals_concat(
                    parameters= parameters,
                    fk= fk,
                    day= day
                )['embeddings']
                if data_restriction is not None:
                    if 'limit' in data_restriction:
                        zVals = zVals[0:data_restriction['limit']]
                    elif 'nth_value' in data_restriction:
                        zVals = zVals[0::data_restriction['nth_value']]
                x, y = zVals[:,0], zVals[:,1] 
                zVals_combined_list.extend(zVals)
        fk_content_dict[str(fk)] = zVals_combined_list
    combined_list = []
    print('combine lists')
    for k,v in fk_content_dict.items():
        combined_list.extend(v)
    x = list(enumerate(combined_list))
    # generate dict of index assignments
    fk_index_asgnm_dict = {}
    current_index = 0
    print('assign indices to all fks')
    for idx, fk in tqdm(enumerate(fishkey_list)): 
        fk_list_ending_idx = current_index + len(fk_content_dict[str(fk)])
        fk_index_asgnm_dict[str(fk)] = list(map(lambda x: x, range(current_index, fk_list_ending_idx)))
        current_index = fk_list_ending_idx

    print('shuffle combined list')
    random.shuffle(x)
    indices, l = zip(*x)

    # assign to every index an individual
    print('assign fk to shuffeled list')
    shuffled_individual_assignment_list = []
    for idx, el in tqdm(x):
        individual = [k for k, v in fk_index_asgnm_dict.items() if idx in v]
        shuffled_individual_assignment_list.append(individual)
        # color = cmap(gen(assigned_list))
        # color_list.append(color)
    # >color_idx_list = [0.1, 0.4, 0.8,...]
    print('assign index to fk in shuffeled list')
    fk_to_index_dict = {}
    for idx, el in tqdm(enumerate(fk_content_dict.keys())):
        fk_to_index_dict[str(el)] = idx
    # vector of individual-identifier, directly corresponding with element key
    print('generate indices list for colorings')
    fk_ind_list = []
    for el in tqdm(enumerate(shuffled_individual_assignment_list)):
        curr_el = el[1][0]
        fk_ind_list.append(fk_to_index_dict[str(curr_el)])

    print('figure generation')
    custom_cmap = adjust_cmap(cmap, len(fishkey_list))
    fig, ax = plt.subplots()
    ax.scatter(
        x = [item[1][0] for item in x], 
        y = [item[1][1] for item in x], 
        s = point_size,
        alpha = alpha_transparency,
        c = fk_ind_list,
        cmap= custom_cmap
    )
    ax.set_xlim(axis_limit_tuple[0])
    ax.set_ylim(axis_limit_tuple[1])
    
    if include_axis_visualization:
        ax.axis('on')
    else:
        ax.axis('off')
    if save_pdf_path is not None:
        fig.savefig(
            fname = save_pdf_path, 
            format = 'pdf'
        )
    if plot_figure:
        fig.show()
    return fig, ax


def adjust_cmap(original_cmap, max_elements):
    data_range = np.linspace(0, 1, max_elements)
    my_cmap = original_cmap(data_range)
    start_index = 0  
    end_index = max_elements  
    subset_colors = my_cmap[start_index:end_index]
    custom_cmap = mcolors.ListedColormap(subset_colors)
    custom_cmap.set_under('white')
    return custom_cmap


def plot_umap_trajectories_and_watershed_characteristics(
        parameters,
        wshed_path,
        fish_key,
        day,
        mode = 'clusters', 
        include_boundaries = True,
        trajectory_color = 'red',
        data_restriction= None,
        data_restriction_additional = None,
        axis_limit_tuple = ([-100, 100], [-100, 100]),
        include_axis_visualization= False, 
        cmap = 'default',
        trajectory_alpha_transparency = None,
        save_pdf_path: os.path = None,
        plot_figure= False
    ) -> plt.figure:
    ''' 
    plots umap-trajectories of a `fish_key` for a specific `day` and,
    depending on the mode, with either `clusters` a colorized overview of different 
    clusters with cluster-ids or a combination of the umap-density and the 
    watershed-cluster boundaries. 
    the number of trajectories can be limited using the `data_restriction_limit` flag.
    '''
    wshed_dict = load_watershed_file(wshed_path)
    
    fig, ax = plt.subplots()
    if mode == 'clusters':
        get_watershed_clusters_figure(cluster_embeddings = wshed_dict['LL'], 
            extent_factor = wshed_dict['xx'][0][-1], 
            original_figure_width = wshed_dict['density'].shape[0], 
            axis_limit_tuple = axis_limit_tuple,
            overloaded_figure = (fig, ax), 
            include_axis_visualization = False, 
            cmap= cmap,
            plot_figure = False
        )
    else:
        get_umap_density_figure(
            umap_embedding = wshed_dict['density'],
            extent_factor = wshed_dict['xx'][0][-1],
            axis_limit_tuple = axis_limit_tuple,
            overloaded_figure = (fig,ax),
            include_axis_visualization = False,
            cmap=cmap,
            plot_figure = False
        )

        if include_boundaries:
            get_watershed_boundaries_figure(
                boundaries_embedding= wshed_dict['wbounds'],
                extent_factor= wshed_dict['xx'][0][-1],
                original_figure_width= wshed_dict['density'].shape[0],
                axis_limit_tuple = axis_limit_tuple,
                overloaded_figure= (fig,ax),
                include_axis_visualization = False,
                plot_figure = False
            )

    get_umap_trajectories_figure(
        parameters = parameters,
        fish_key = fish_key,
        day = day,
        data_restriction= data_restriction,
        data_restriction_additional= data_restriction_additional,
        axis_limit_tuple = axis_limit_tuple,
        alpha_transparency = trajectory_alpha_transparency,
        figure_color= trajectory_color,
        overloaded_figure = (fig,ax),
        include_axis_visualization = False, 
        plot_figure = False
    )

    if include_axis_visualization:
        ax.axis('on')
    else:
        ax.axis('off')
    
    if save_pdf_path is not None:
        fig.savefig(
            fname = save_pdf_path, 
            format = 'pdf'
        )
    if plot_figure:
        fig.show()
    return fig, ax


def plot_basic_features_and_frequency_activations(
    parameters,
    projection_file,
    interval = [54000, 55500],
    normalize_projections = False,
    original_features_save_pdf_path = None,
    sampling_frequency = 5,
    number_of_periods = 25,
    min_frequency = 0.01,
    max_frequency = 2.5,
    numProcessors=10, 
    useGPU = -1,
    frequency_activations_save_pdf_path = None,
    ):
    '''
    Plots the three basic features (step length, turning angle, distance to the wall)
    and their respective activations resulting from a morlet transformation for the wavelet
    features. The `number_of_periods` should be equal 25 with a `sampling_frequency` of 5 
    and a `min_frequency` of 0.01 and `max_frequency` of 2.5 to match previous computations.
    The activations can be plotted after normalization if `normalize_projections` is set to True.
    '''
    parameters.projectionDirectory = parameters.projectPath + '/Projections'
    projections = np.array(hdf5storage.loadmat(projection_file, variable_names=['projections'])['projections'])

    if normalize_projections:
        parameters.normalize_func = return_normalization_func(parameters)
        projections = parameters.normalize_func(projections)


    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (20, 10))

    x1 = projections[:,0][interval[0]:interval[1]]
    x2 = projections[:,1][interval[0]:interval[1]]
    x3 = projections[:,2][interval[0]:interval[1]]

    y = np.arange(x1.shape[0])

    x_ticks_numbers = int((interval[1] - interval[0])/300 + 1)
    x_ticks_location_list = list(map(lambda x: x * 300, range(0, x_ticks_numbers)))
    x_ticks_label_list = list(map(lambda x: x * 60, range(0, x_ticks_numbers)))

    ax1.plot(y, x1, c='black')
    ax1.set_title('PC_1 over time')
    ax2.plot(y, x2, c='black')
    ax2.set_title('PC_2 over time')
    ax3.plot(y, x3, c='black')
    ax3.set_title('PC_3 over time')

    ax1.set_xlim(0,interval[1]-interval[0])
    ax2.set_xlim(0,interval[1]-interval[0])
    ax3.set_xlim(0,interval[1]-interval[0])

    ax1.set_xticks(x_ticks_location_list, x_ticks_label_list)
    ax2.set_xticks(x_ticks_location_list, x_ticks_label_list)
    ax3.set_xticks(x_ticks_location_list, x_ticks_label_list)

    if original_features_save_pdf_path is not None:
        fig.savefig(original_features_save_pdf_path)

    # wavelet transformation

    amplitudes, f = mmpy.findWavelets(
        projections = projections, 
        pcaModes = 3, 
        omega0 = 5, 
        numPeriods = number_of_periods, 
        samplingFreq = sampling_frequency, 
        maxF = max_frequency, 
        minF = min_frequency, 
        numProcessors = numProcessors, 
        useGPU = useGPU
    )

    fig_2, (ax1_2, ax2_2, ax3_3) = plt.subplots(3, 1, figsize = (20, 10))

    x1_wavelet = amplitudes[:,:25][interval[0]:interval[1]]
    x2_wavelet = amplitudes[:,25:50][interval[0]:interval[1]]
    x3_wavelet = amplitudes[:,50:75][interval[0]:interval[1]]

    im1 = ax1_2.imshow(x1_wavelet.T, aspect='auto')
    ax1_2.set_yticks(np.arange(x1_wavelet.T.shape[0]))
    ax1_2.invert_yaxis()
    ax1_2.set_title('Wavelet Transformations for PC_1')
    im2 = ax2_2.imshow(x2_wavelet.T, aspect='auto')
    ax2_2.set_yticks(np.arange(x2_wavelet.T.shape[0]))
    ax2_2.invert_yaxis()
    ax2_2.set_title('Wavelet Transformations for PC_2')
    im3 = ax3_3.imshow(x3_wavelet.T, aspect='auto')
    ax3_3.set_yticks(np.arange(x3_wavelet.T.shape[0]))
    ax3_3.invert_yaxis()
    ax3_3.set_title('Wavelet Transformations for PC_3')

    ax1_2.set_xlim(0,interval[1]-interval[0])
    ax2_2.set_xlim(0,interval[1]-interval[0])
    ax3_3.set_xlim(0,interval[1]-interval[0])
    ax1_2.set_xticks(x_ticks_location_list, x_ticks_label_list)
    ax2_2.set_xticks(x_ticks_location_list, x_ticks_label_list)
    ax3_3.set_xticks(x_ticks_location_list, x_ticks_label_list)

    if frequency_activations_save_pdf_path is not None:
        fig_2.savefig(frequency_activations_save_pdf_path)
    
    return fig, fig_2


def plot_physical_trajectories(
    parameters,
    wshed_dict,
    fk,
    day,
    interval = [54000,55500],
    save_pdf_path: os.path = None,
    plot_figure= False
):
    '''
    plots the trajectory in physical space for one individual, indicated with `fk`
    for a specific `day` given a certain `interval`.
    '''
    sum_data = load_summerized_data(
        wshed_dict,
        parameters, 
        fk,
        day
    )
    positions = sum_data['positions']
    area_box = sum_data['area']
    fig, ax = plt.subplots(figsize=(10,10))

    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    area_box = np.concatenate((area_box, [area_box[0]]))
    ax.plot(*area_box.T, color="black")
    ax.axis('off')
    (line,) = ax.plot([],[], "-o", markersize=3, color="blue")
    line.set_data(*positions[interval[0]:interval[1]].T)
    if save_pdf_path is not None:
        fig.savefig(save_pdf_path)
    return fig, ax


def plot_multiple_umap_trajectories_and_watershed_characteristics(
        parameters,
        wshed_path,
        identifier_dict,
        mode = 'clusters', 
        include_boundaries = True,
        data_restriction= None,
        data_restriction_additional = None,
        axis_limit_tuple = ([-100, 100], [-100, 100]),
        trajectory_alpha_transparency = None,
        include_axis_visualization= False, 
        cmap = 'default',
        save_pdf_path: os.path = None,
        plot_figure= False
    ) -> plt.figure:
    ''' 
    plots umap-trajectories of multiple fishes for each a specific day and,
    depending on the mode, with either `clusters` a colorized overview of different 
    clusters with cluster-ids or a combination of the umap-density and the 
    watershed-cluster boundaries. 
    the number of trajectories can be limited using the `data_restriction_limit` flag.
    The individuals have to be indicated using the `identifier_dict` in a way, s.t. 
    the `fish_key` resembles the key of an object and the value is an embedding of 
    another object including the respective `day` and the `trajectory_color`.
    ```identifier_dict = {
        'block2_23484201_back': {
            'day': '20211112_060000',
            'trajectory_color': 'red'
            }, ...
        }
    '''
    wshed_dict = load_watershed_file(wshed_path)
    


    fig, ax = plt.subplots()
    if mode == 'clusters':
        get_watershed_clusters_figure(cluster_embeddings = wshed_dict['LL'], 
            extent_factor = wshed_dict['xx'][0][-1], 
            original_figure_width = wshed_dict['density'].shape[0], 
            axis_limit_tuple = axis_limit_tuple,
            overloaded_figure = (fig, ax), 
            include_axis_visualization = False, 
            cmap= cmap,
            plot_figure = False
        )
    else:
        get_umap_density_figure(
            umap_embedding = wshed_dict['density'],
            extent_factor = wshed_dict['xx'][0][-1],
            axis_limit_tuple = axis_limit_tuple,
            overloaded_figure = (fig,ax),
            include_axis_visualization = False,
            cmap=cmap,
            plot_figure = False
        )

        if include_boundaries:
            get_watershed_boundaries_figure(
                boundaries_embedding= wshed_dict['wbounds'],
                extent_factor= wshed_dict['xx'][0][-1],
                original_figure_width= wshed_dict['density'].shape[0],
                axis_limit_tuple = axis_limit_tuple,
                overloaded_figure= (fig,ax),
                include_axis_visualization = False,
                plot_figure = False
            )

    for fish_key, fish_values in identifier_dict.items():
        get_umap_trajectories_figure(
            parameters = parameters,
            fish_key = fish_key,
            day = fish_values['day'],
            figure_color= fish_values['trajectory_color'],
            data_restriction= data_restriction,
            data_restriction_additional= data_restriction_additional,
            axis_limit_tuple = axis_limit_tuple,
            alpha_transparency = trajectory_alpha_transparency,
            overloaded_figure = (fig,ax),
            include_axis_visualization = False, 
            plot_figure = False
        )

    if include_axis_visualization:
        ax.axis('on')
    else:
        ax.axis('off')
    
    if save_pdf_path is not None:
        fig.savefig(
            fname = save_pdf_path, 
            format = 'pdf'
        )
    if plot_figure:
        fig.show()
    return fig, ax


def plot_umap_density_and_watershed_boundaries(
    wshed_path,
    axis_limit_tuple = ([-100, 100], [-100, 100]),
    include_axis_visualization= False, 
    cmap = 'default',
    save_pdf_path: os.path = None,
    plot_figure= False
)-> plt.figure:
    '''
    plotting of the umap density and the watershed boundaries in one figure
    this figure can be saved as a pdf and/or plotted
    '''
    wshed_dict = load_watershed_file(wshed_path)
    
    fig, ax = plt.subplots()
    get_umap_density_figure(
        umap_embedding = wshed_dict['density'],
        extent_factor = wshed_dict['xx'][0][-1],
        axis_limit_tuple = axis_limit_tuple,
        overloaded_figure = (fig,ax),
        include_axis_visualization = False,
        cmap=cmap,
        plot_figure = False
    )

    get_watershed_boundaries_figure(
        boundaries_embedding= wshed_dict['wbounds'],
        extent_factor= wshed_dict['xx'][0][-1],
        original_figure_width= wshed_dict['density'].shape[0],
        axis_limit_tuple = axis_limit_tuple,
        overloaded_figure= (fig,ax),
        include_axis_visualization = False,
        plot_figure = False
    )
    if include_axis_visualization:
        ax.axis('on')
    else:
        ax.axis('off')
    
    if save_pdf_path is not None:
        fig.savefig(
            fname = save_pdf_path, 
            format = 'pdf'
        )
    if plot_figure:
        fig.show()
    return fig, ax


def get_color_map(n):
    cmap = mmpy.motionmapper.gencmap()
    return lambda cid: cmap(cid*64//n)