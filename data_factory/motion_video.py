import numpy as np
# this packages helps load and save .mat files older than v7
import hdf5storage, h5py, os, tqdm 
from time import gmtime, strftime
from matplotlib import pyplot as plt
from scipy.stats import entropy as entropy_m
from matplotlib.animation import FuncAnimation
# moviepy helps open the video files in Python
from moviepy.editor import VideoClip, VideoFileClip, clips_array, vfx
from moviepy.video.io.bindings import mplfig_to_npimage
from data_factory.plasticity import compute_cluster_distribution
from data_factory.plotting import get_umap_density_figure, get_watershed_boundaries_figure
import motionmapperpy as mmpy

#from .plotting import get_color_map
from .processing import load_summerized_data, get_fish_info_from_wshed_idx, get_regions_for_fish_key
from .utils import get_cluster_sequences
from fishproviz.utils import get_date_string, get_seconds_from_day, get_camera_pos_keys, get_all_days_of_context

STIME = "060000"
VIDEOS_DIR = "videos"

def get_color_map(n):
    cmap = mmpy.motionmapper.gencmap()
    return lambda cid: cmap(cid*64//n)

def motion_video(wshedfile, parameters,fish_key, day, start=0, duration_seconds=20, save=False, filename="", score="", axis_limit_tuple=None):
    try:
        tqdm._instances.clear()
    except:
        pass

    #data 
    sum_data = load_summerized_data(wshedfile,parameters,fish_key,day)
    zValues = sum_data['embeddings']
    positions = sum_data['positions']
    clusters = sum_data["clusters"]
    area_box = sum_data['area']
    time_df = sum_data['df_time_index']

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10,5))
   
    get_umap_density_figure(
        umap_embedding= wshedfile['density'],
        extent_factor= wshedfile['xx'][0][-1],
        axis_limit_tuple= axis_limit_tuple,
        overloaded_figure= (fig,ax1),
        include_axis_visualization= False,
        plot_figure= False
        )
    get_watershed_boundaries_figure(
        boundaries_embedding= wshedfile['wbounds'],
        extent_factor= wshedfile['xx'][0][-1],
        axis_limit_tuple= axis_limit_tuple,
        original_figure_width= wshedfile['density'].shape[0],
        overloaded_figure= (fig,ax1),
        include_axis_visualization= False,
        plot_figure= False
        )
    # zoom in on ax1 by 2 std using the extent factor of the density plot
    #ax1.set_xlim(
    ax1.axis('off')
    ax1.set_title(' ')

    # tight layout 
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)

    #ax1.set_xlim
    lineZ = ax1.plot([], [], "-", color="red", alpha=0.5, linewidth=2) 
    sc = ax1.scatter([],[],marker='o', c="b", s=50)
    area_box = np.concatenate((area_box, [area_box[0]]))
    ax2.plot(*area_box.T, color="black")
    (line,) = ax2.plot([],[], "-o", markersize=3, color="blue")
    tstart = start 
    offset_df = 100
    entro = entropy_m(compute_cluster_distribution(clusters[start-offset_df:start+offset_df+duration_seconds*5], parameters.kmeans))
    
    cmap = get_color_map(clusters.max())
    def animate(t):
        t = int(t*20)+tstart
        line.set_data(*positions[t-offset_df:t+offset_df].T)
        ax2.axis('off')
        ax1.set_title('%s time %s  entropy: %.02f'%(get_date_string(day), strftime("%H:%M:%S",gmtime(time_df[t]//5)), entro))  
        lineZ[0].set_data(*zValues[t-offset_df:t+offset_df].T) 
        sc.set_offsets(zValues[t])
        sc.set_color(cmap(clusters[t]))
        return mplfig_to_npimage(fig) #im, ax

    anim = VideoClip(animate, duration=duration_seconds) # will throw memory error for more than 100.
    plt.close()
    if save:
        dir_v = f'{parameters.projectPath}/{VIDEOS_DIR}/{parameters.kmeans}_clusters'
        os.makedirs(dir_v, exist_ok=True)
        anim.write_videofile(f'{dir_v}/{filename}{fish_key}_{day}.mp4', fps=10, audio=False, threads=1)
    return anim


def motion_video_for_low_and_high_entropy_individuals(
    wshedfile, 
    parameters,
    low_entropy_fk_day_dict,
    high_entropy_fk_day_dict,
    start = 0, 
    duration_seconds = 1800, 
    speedup_factor = 6,
    fps = 30,
    save=False, 
    filename="", 
    axis_limit_tuple = ([-50, 75], [-50, 75])
    ): 
    tqdm.tqdm._instances.clear()

    anim_trajectories_low_entropy, anim_umap_low_entropy = motion_video_for_3_individuals(
        wshedfile= wshedfile, 
        parameters= parameters, 
        fk_day_dict= low_entropy_fk_day_dict, 
        start= start, 
        duration_seconds= duration_seconds, 
        axis_limit_tuple= axis_limit_tuple
    )
    anim_trajectories_high_entropy, anim_umap_high_entropy = motion_video_for_3_individuals(
        wshedfile= wshedfile, 
        parameters= parameters, 
        fk_day_dict= high_entropy_fk_day_dict, 
        start= start, 
        duration_seconds= duration_seconds, 
        axis_limit_tuple= axis_limit_tuple
    )
    clip_traj_low_entropy = anim_trajectories_low_entropy.resize((250,750))
    clip_umap_low_entropy = anim_umap_low_entropy.resize((750,750))
    low_entropy_clip = clips_array([[clip_traj_low_entropy, clip_umap_low_entropy]])

    clip_traj_high_entropy = anim_trajectories_high_entropy.resize((250,750))
    clip_umap_high_entropy = anim_umap_high_entropy.resize((750,750))
    high_entropy_clip = clips_array([[clip_traj_high_entropy, clip_umap_high_entropy]])

    combined_entropy_clips = clips_array([[low_entropy_clip, high_entropy_clip]])

    combined_entropy_clips = combined_entropy_clips.set_fps(fps)
    print("fps: {}".format(combined_entropy_clips.fps))
    # speedup
    combined_entropy_clip_final = combined_entropy_clips.fx(vfx.speedx, speedup_factor)
    print("fps: {}".format(combined_entropy_clip_final.fps))

    if save:
        combined_entropy_clip_final.write_videofile(f'combined_entropy_speedup_{speedup_factor}.mp4', audio=None, threads=10)
    return combined_entropy_clip_final


def motion_video_for_3_individuals(
    wshedfile, 
    parameters, 
    fk_day_dict, 
    start, 
    duration_seconds, 
    axis_limit_tuple
    ):
    try:
        tqdm.tqdm._instances.clear()
    except:
        pass
    # data
    sum_data_1 = load_summerized_data(wshedfile,parameters,
        fish_key= list(fk_day_dict.keys())[0],
        day= fk_day_dict[list(fk_day_dict.keys())[0]]['day4']
    )
    zValues_1 = sum_data_1['embeddings']
    positions_1 = sum_data_1['positions']
    clusters_1 = sum_data_1["clusters"]
    area_box_1 = sum_data_1['area']
    time_df_1 = sum_data_1['df_time_index']
    sum_data_2 = load_summerized_data(wshedfile,parameters,
        fish_key= list(fk_day_dict.keys())[1],
        day= fk_day_dict[list(fk_day_dict.keys())[1]]['day4']
    )
    zValues_2 = sum_data_2['embeddings']
    positions_2 = sum_data_2['positions']
    clusters_2 = sum_data_2["clusters"]
    area_box_2 = sum_data_2['area']
    time_df_2 = sum_data_2['df_time_index']
    sum_data_3 = load_summerized_data(wshedfile,parameters,
        fish_key= list(fk_day_dict.keys())[2],
        day= fk_day_dict[list(fk_day_dict.keys())[2]]['day4']
    )
    zValues_3 = sum_data_3['embeddings']
    positions_3 = sum_data_3['positions']
    clusters_3 = sum_data_3["clusters"]
    area_box_3 = sum_data_3['area']
    time_df_3 = sum_data_3['df_time_index']
    
    # trajectories
    fig, (ax1,ax2, ax3) = plt.subplots(3,1 , figsize=(5,15))
    area_box = np.concatenate((area_box_1, [area_box_1[0]]))
    ax1.plot(*area_box.T, color="black")
    ax2.plot(*area_box.T, color="black")
    ax3.plot(*area_box.T, color="black")

    (line_1,) = ax1.plot([],[], "-o", markersize=3, color="red")
    (line_2,) = ax2.plot([],[], "-o", markersize=3, color="blue")
    (line_3,) = ax3.plot([],[], "-o", markersize=3, color="black")
    tstart = start 
    offset_df = 100
    def animate(t):
        t = int(t*20)+tstart
        line_1.set_data(*positions_1[t-offset_df:t+offset_df].T)
        line_2.set_data(*positions_2[t-offset_df:t+offset_df].T)
        line_3.set_data(*positions_3[t-offset_df:t+offset_df].T)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        return mplfig_to_npimage(fig) #im, ax
    anim_trajectories = VideoClip(animate, duration=duration_seconds) 

    # umap
    fig2, ax4 = plt.subplots(1, 1, figsize=(5,5)) 
    get_umap_density_figure(
            umap_embedding= wshedfile['density'],
            extent_factor= wshedfile['xx'][0][-1],
            axis_limit_tuple= axis_limit_tuple,
            overloaded_figure= (fig2,ax4),
            include_axis_visualization= False,
            plot_figure= False
            )
    get_watershed_boundaries_figure(
        boundaries_embedding= wshedfile['wbounds'],
        extent_factor= wshedfile['xx'][0][-1],
        axis_limit_tuple= axis_limit_tuple,
        original_figure_width= wshedfile['density'].shape[0],
        overloaded_figure= (fig2,ax4),
        include_axis_visualization= False,
        plot_figure= False
        )
    ax4.axis('off')
    ax4.set_title(' ')

    fig2.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    lineZ_1 = ax4.plot([], [], "-", color="red", alpha=0.5, linewidth=2) 
    sc_1 = ax4.scatter([],[],marker='o', c="r", s=50)
    lineZ_2 = ax4.plot([], [], "-", color="blue", alpha=0.5, linewidth=2) 
    sc_2 = ax4.scatter([],[],marker='o', c="b", s=50)
    lineZ_3 = ax4.plot([], [], "-", color="black", alpha=0.5, linewidth=2) 
    sc_3 = ax4.scatter([],[],marker='o', c="black", s=50)
    offset_df = 20
    cmap = get_color_map(clusters_1.max())
    def animate_umap(t):
        t = int(t*20)+tstart
        lineZ_1[0].set_data(*zValues_1[t-offset_df:t+offset_df].T) 
        sc_1.set_offsets(zValues_1[t+offset_df])
        lineZ_2[0].set_data(*zValues_2[t-offset_df:t+offset_df].T) 
        sc_2.set_offsets(zValues_2[t+offset_df])
        lineZ_3[0].set_data(*zValues_3[t-offset_df:t+offset_df].T) 
        sc_3.set_offsets(zValues_3[t+offset_df])
        return mplfig_to_npimage(fig2) #im, ax

    anim_umap = VideoClip(animate_umap, duration=duration_seconds) 
    return anim_trajectories, anim_umap

def get_color(ci):
    color = ['lightcoral', 'darkorange', 'olive', 'teal', 'violet', 
         'skyblue']
    return color[ci%len(color)]


def cluster_motion_axes(wshedfile, parameters,fish_key, day, start=0, ax=None, score=0):
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=(5,5))
    sum_data = load_summerized_data(wshedfile,parameters,fish_key,day)
    embeddings = sum_data['embeddings']
    positions = sum_data['positions']
    clusters = sum_data["clusters"]
    area = sum_data['area']
    
    area_box = np.concatenate((area, [area[0]]))
    ax.plot(*area_box.T, color="black")
    (line,) = ax.plot([],[], "-o", color="blue")
    
    day_start = get_seconds_from_day(day+"_"+STIME)
    def update_line(t):
        t = int(t*50)+start
        line.set_data(*positions[t-100:t+100].T)
        ax.axis('off')
        ax.set_title('%s %s %s ratio:%.2f'%(fish_key, get_date_string(day), strftime("%H:%M:%S",gmtime((t//5)+day_start)), score))
        
    return update_line

def cluster_motion_video(wshedfile, parameters, clusters, cluster_id, rows=2, cols=2, th=0.5):
    lens = wshedfile['zValLens'].flatten()
    cumsum_lens = lens.cumsum()[:-1]
    clusters[cumsum_lens]=-1 # indicate the end of the day
    results = get_cluster_sequences(clusters, [cluster_id], sw=2*60*5, th=th)
    sequences_of_cid = []
    for s,e,score in results[cluster_id]:
        try: 
            fk, day, start, end = get_fish_info_from_wshed_idx(wshedfile,s,e)
            sequences_of_cid.append((fk, day, start, end, score))
        except ValueError as e:
            pass
            
    fig, axes = plt.subplots(rows, cols, figsize=(5*rows,5*cols))
    update_functions = list()
    for (fk, day, start, end, score),ax in zip(sequences_of_cid, axes.flatten()):
        up_f = cluster_motion_axes(wshedfile, parameters, fk, day, start, ax=ax, score=score)
        update_functions.append(up_f)
 
    def animate(t):
        for f in update_functions:
            f(t)
        return mplfig_to_npimage(fig)
    
    anim = VideoClip(animate, duration=30) # will throw memory error for more than 100.
    plt.close()
    dir_v = f'{parameters.projectPath}/{VIDEOS_DIR}/{parameters.kmeans}_clusters'
    if not os.path.exists(dir_v):
        os.mkdir(dir_v)
    anim.write_videofile(f'{dir_v}/cluster_{str(cluster_id)}.mp4', fps=10, audio=False, threads=4)
    return anim
    
    