
import os
import matplotlib.pyplot as plt
import numpy as np
from random import sample
from time import gmtime, strftime
import hdf5storage
import motionmapperpy as mmpy
from clustering.clustering import get_results_filepath, boxplot_characteristics_of_cluster
from config import BLOCK, VIS_DIR
from .processing import get_regions_for_fish_key, load_zVals_concat
from .utils import pointsInCircum, get_individuals_keys, get_days
from clustering.transitions_cluster import transition_rates, draw_transition_graph

        
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
    
    
def ethnogram_of_clusters(parameters, clusters, start_time=0, end_time=8*(60**2)*5, fish_key="", day="",rows=4, write_fig=False, name_append=""):
    wregs = clusters[start_time:end_time]
    f2min = (60**2)*5
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


def cluster_density_umap(embeddings, clusters, filename=None,n_select=10000):
    m = np.abs(embeddings).max()
    cmap= get_color_map(clusters.max())
    sigma=2.0
    off_set = 10
    _, xx, density = mmpy.findPointDensity(embeddings, sigma, 511, [-m-off_set, m+off_set])
    subset = sample(range(embeddings.shape[0]), min(n_select,embeddings.shape[0]))
    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    sc = axes[0].scatter(embeddings[subset][:,0], embeddings[subset][:,1], marker='.', c=cmap(clusters[subset]), s=0.2)
    axes[0].set_xlim([-m-off_set, m+off_set])
    axes[0].set_ylim([-m-off_set, m+off_set])
    
    for i in np.unique(clusters):
            fontsize = 8
            X_i = embeddings[clusters == i]
            axes[0].text(*np.mean(X_i,axis=0), str(i), fontsize=fontsize, fontweight='bold', backgroundcolor=cmap(i))
            
    #fig.colorbar(sc)
    for ax in axes:
        ax.axis('off')

    axes[1].imshow(density, cmap=mmpy.gencmap(), extent=(xx[0], xx[-1], xx[0], xx[-1]), origin='lower')
    if filename:
        fig.tight_layout()
        fig.savefig(filename, bbox_inches='tight')
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
    ax.imshow(
        X= umap_embedding,
        extent=(-extent_factor, extent_factor, -extent_factor, extent_factor), 
        origin='lower', 
        cmap=cmap
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
        color='k', 
        s=0.1
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
        overloaded_figure=None, 
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
        color=figure_color
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
    return fig, ax


def umap_scatter_figure_for_all(
        parameters, 
        point_size = 15,
        alpha_transparency = 0.5,
        figure_color= 'red',
        data_restriction = None, 
        elements_restriction = None,
        axis_limit_tuple = ([-100, 100], [-100, 100]),
        overloaded_figure=None, 
        include_axis_visualization = False, 
        save_pdf_path: os.path = None,
        plot_figure = False
    ) -> plt.figure:
    '''
    scatter umap-trajectory points for the whole dataset of all fish_keys and day
    by following a `data_restriction` rule, the following rules 
    are supported and have to be formulated in a dictionary
    >>> data_restriction: {'limit': 200} or {'nth_value': 50}
    the arguments `point_size` and `alpha_transparency` directly
    affect the visual appearance.
    By using an `elements_restriction`, it is possible to only use a certain number 
    of samples instead of the whole dataset.
    '''

    if overloaded_figure:
        fig, ax = overloaded_figure
    else: 
        fig, ax = plt.subplots()
        
    fishkey_list = get_individuals_keys(parameters= parameters)
    days_list = get_days(parameters= parameters)

    elements_counter = 0
    for fk in fishkey_list:
        for day in days_list:
            zVal_path = parameters.projectPath+f'/Projections/{fk}_{day}_pcaModes_uVals.mat'
            if os.path.exists(zVal_path):
                elements_counter += 1
                if (elements_restriction is not None) and (elements_counter > elements_restriction):
                    return (fig, ax)
                fig, ax = get_umap_scatter_figure_per_fk_day(
                    parameters= parameters,
                    fish_key= fk,
                    day= day,
                    point_size= point_size,
                    alpha_transparency= alpha_transparency,
                    figure_color= figure_color,
                    data_restriction= data_restriction,
                    overloaded_figure = (fig, ax)
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


def plot_umap_trajectories_and_watershed_characteristics(
        parameters,
        wshed_path,
        fish_key,
        day,
        mode = 'clusters', 
        trajectory_color = 'red',
        data_restriction= None,
        data_restriction_additional = None,
        axis_limit_tuple = ([-100, 100], [-100, 100]),
        include_axis_visualization= False, 
        cmap = 'default',
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