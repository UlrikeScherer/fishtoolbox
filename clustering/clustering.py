from config import (
    BATCH_SIZE,
    BLOCK,
    BACK,
    sep,
    VIS_DIR,
    DIR_TRACES,
    CAM_POS,
    DAY,
    BATCH,
    DATAFRAME,
)
from fishproviz.utils.error_filter import all_error_filters, error_default_points
from data_factory.plot_helpers import remove_spines
from fishproviz.utils.transformation import rotation, pixel_to_cm
from fishproviz.metrics import (
    entropy,
    distance_to_wall,
    activity,
    turning_angle,
    absolute_angles,
)
from fishproviz.utils import get_fish2camera_map, csv_of_the_day, get_date_string
from fishproviz.utils.tank_area_config import get_area_functions
from itertools import product
import os
import pandas as pd
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from sklearn.cluster import MiniBatchKMeans

MU_STR, SD_STR = "mu", "sd"


def get_traces_type():
    _, names = get_metrics_for_traces()
    traces_type = np.dtype(
        {
            "names": [CAM_POS, DAY, BATCH, DATAFRAME] + names,
            "formats": ["str"] * 4 + ["f4"] * len(names),
        }
    )
    return traces_type


def get_results_filepath(trace_size, file_name, subfolder=None, format="pdf"):
    if subfolder:
        path_ = "%s/%s_trace_size_%s/%s" % (VIS_DIR, BLOCK, trace_size, subfolder)
    else:
        path_ = "%s/%s_trace_size_%s" % (VIS_DIR, BLOCK, trace_size)

    if not os.path.exists(path_):
        os.makedirs(path_)
    return "%s/%s_%d.%s" % (path_, file_name, trace_size, format)


def get_trace_file_path(trace_size, format="csv"):
    return "%s/traces_%s_%s.%s" % (DIR_TRACES, BLOCK, trace_size, format)


def load_traces(trace_size):
    trace_path = get_trace_file_path(trace_size)
    trace_path_npy = get_trace_file_path(trace_size, "npy")
    if not os.path.exists(trace_path) or not os.path.exists(trace_path_npy):
        raise FileNotFoundError("Trace for path %s does not exist" % trace_path)
    traces = pd.read_csv(trace_path, delimiter=sep, index_col=0)
    with open(trace_path_npy, "rb") as f:
        samples = np.load(f)
    return traces, samples


def get_trace_filter(traces):
    return traces.isna().any(axis=1)


def calculate_traces(fish_indices, days, trace_size, write_to_file=False):
    fish2cams = get_fish2camera_map()
    Xs, nSs = list(), list()
    area_func = get_area_functions()

    for i in fish_indices:
        cam, is_back = fish2cams[i][0], fish2cams[i][1] == BACK
        fish_key = "%s_%s" % (cam, fish2cams[i, 1])
        for d in days:
            batch_keys, batches = csv_of_the_day(cam, d, is_back=is_back)

            if len(batches) == 0:
                continue
            b = pd.concat(batches)
            fit_len = fit_data_to_trace_size(len(b), trace_size)
            data = b[["xpx", "ypx"]].to_numpy()[:fit_len]
            area_tuple = (fish_key, area_func(fish_key))
            # filter for errorouse data points
            filter_index = all_error_filters(data, area_tuple)[:fit_len]

            #X = transform_to_traces_metric_based(
            X = transform_to_traces(
                data, trace_size, filter_index, area_tuple
            )
            X_df = table_factory(fish_key, d, batch_keys, X, trace_size)
            Xs.append(X_df)
            nSs.append(trajectory_snippets(data, trace_size))
    traces = pd.concat(Xs)
    traces_size = traces.shape[0]
    tfilter = traces.isna().any(axis=1)
    traces = traces[~tfilter]
    traces.reset_index(drop=True, inplace=True)
    nSs = np.concatenate(nSs)[~tfilter]
    print(
        "Out of %d traces %d where filtered out because of nan values"
        % (traces_size, sum(tfilter))
    )
    if write_to_file:
        os.makedirs(DIR_TRACES, exist_ok=True)
        traces.to_csv(get_trace_file_path(trace_size), sep=sep)
        with open(get_trace_file_path(trace_size, format="npy"), "wb") as f:
            np.save(f, nSs)
    return traces, nSs


def traces_as_numpy(traces):
    return traces.to_numpy()[:, 4:].astype(float)


def get_traces_columns():
    _, names = get_metrics_for_traces()
    return [CAM_POS, DAY, BATCH, DATAFRAME] + names


def table_factory(key_c_p, day, batch_keys, traces_of_day, trace_size):
    col = get_traces_columns()
    if len(col[4:])< traces_of_day.shape[1]:
        col = col[:4] + list(range(traces_of_day.shape[1]))
    traces_df = pd.DataFrame(columns=col, index=range(traces_of_day.shape[0]))
    traces_df.loc[:, col[4:]] = traces_of_day

    traces_df.loc[:, col[0:2]] = key_c_p, day

    dataframe_pointer = np.array(range(traces_of_day.shape[0])) * trace_size
    traces_df.loc[:, col[3]] = dataframe_pointer
    for i, b_key in enumerate(batch_keys):
        mask = (dataframe_pointer >= (BATCH_SIZE * i)) & (
            dataframe_pointer < (BATCH_SIZE * (i + 1))
        )
        traces_df.loc[mask, col[2]] = b_key
    return traces_df


def fit_data_to_trace_size(size1, trace_size):
    n_snippets = size1 // trace_size
    length = n_snippets * trace_size
    return length


def trajectory_snippets(data, trace_size):
    size1, size2 = data.shape
    n_snippets = (size1-2) // trace_size
    return np.reshape(data[:n_snippets*trace_size], (n_snippets, trace_size, size2))


def rotate_trace(trace):
    alph = np.arctan2(*trace[0])
    return np.dot(trace, rotation(-alph))


def transform_to_traces(batch, trace_size):
    setX = batch[["xpx", "ypx"]].to_numpy()
    setX = setX[1:] - setX[:-1]
    lenX = setX.shape[0]
    sizeSet = int(np.ceil(lenX / trace_size))
    newSet = np.zeros((sizeSet, trace_size, 2))
    for i in range(sizeSet - 1):
        newSet[i, :, :] = rotate_trace(setX[i * trace_size : (i + 1) * trace_size])
    X = np.reshape(newSet, (sizeSet, trace_size * 2))
    return X, newSet


def get_metrics_for_traces():
    metrics_f = [
        activity,
        turning_angle,
        absolute_angles,
        entropy,
        distance_to_wall,
    ]
    names = [
        "%s_%s" % (m, s)
        for m, s in product(map(lambda m: m.__name__, metrics_f), [MU_STR, SD_STR])
    ]
    names.remove("%s_%s" % (entropy.__name__, SD_STR))  # remove entropy_sd
    return metrics_f, names


def transform_to_traces_metric_based(data, trace_size, filter_index, area_tuple):
    lenX = data.shape[0]
    sizeSet = lenX // trace_size
    metric_functions, _ = get_metrics_for_traces()  #
    newSet = np.zeros((sizeSet, len(metric_functions) * 2))
    for i, f in enumerate(metric_functions):
        idx = i * 2
        if f.__name__ == distance_to_wall.__name__:
            newSet[:, idx : idx + 2] = f(data, trace_size, filter_index, area_tuple)[:, :2]
        elif f.__name__ == entropy.__name__:
            entropy_idx = i * 2 + 1
            newSet[:, idx : idx + 2] = f(data, trace_size, filter_index, area_tuple)[
                :, :2
            ]  # only take entropy not std over hist.
        else:
            newSet[:, idx : idx + 2] = f(pixel_to_cm(data), trace_size, filter_index)[
                :, :2
            ]
    newSet = np.delete(newSet, entropy_idx, axis=1)
    # np.nan_to_num(newSet, copy=False, nan=0.0)
    return newSet


def normalize_data_metrics(traces):
    d_std, d_mean = np.std(traces,axis=0), np.mean(traces, axis=0)
    traces = (traces - d_mean) / d_std
    return traces


def clustering(traces, n_clusters, model=None, rating_feature=None):
    if model is None:
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=12)
    if rating_feature is None:
        rating_feature = traces[:, 0]
    clusters = model.fit_predict(traces)
    avg_feature = np.zeros(n_clusters)
    for i in range(n_clusters):
        avg_feature[i] = np.mean(rating_feature[clusters == i])
    index_map = np.argsort(avg_feature)
    renamed_clusters = np.empty(clusters.shape, dtype=int)
    for i, j in enumerate(index_map):
        renamed_clusters[clusters == j] = i
    return renamed_clusters

def get_convex_hull(points):
    hull = ConvexHull(points)
    cycle = points[hull.vertices]
    cycle = np.concatenate([cycle, cycle[:1]])
    return cycle


def get_neighbourhood_selection(X_embedded, nSs, c_x=None, c_y=None, radius=1):
    if c_x is None and c_y is None:
        hist, x, y = np.histogram2d(X_embedded[:, 0], X_embedded[:, 1], bins=10)
        max_x, max_y = int(hist.argmax() / 10), hist.argmax() % 10
        c_x, c_y = (x[max_x] + x[max_x + 1]) / 2, (y[max_y] + y[max_y + 1]) / 2
    find = np.nonzero(
        (np.abs(X_embedded[:, 0] - c_x) + np.abs(X_embedded[:, 1] - c_y)) <= radius
    )
    return nSs[find[0]]


def set_of_neighbourhoods(X_embedded, nSs, radius=1, bins=15):
    hist, x, y = np.histogram2d(X_embedded[:, 0], X_embedded[:, 1], bins=bins)
    len_x = len(x)
    neighbourhoods = dict()
    centers = list()
    for (max_x, max_y) in zip(range(len_x), hist.argsort()[:, -1]):
        c_x, c_y = (x[max_x] + x[max_x + 1]) / 2, (y[max_y] + y[max_y + 1]) / 2
        centers.append([c_x, c_y, hist[max_x, max_y]])
        neighbourhoods[
            "x:%.2f, y:%.2f, n:%d" % (c_x, c_y, hist[max_x, max_y])
        ] = get_neighbourhood_selection(X_embedded, nSs, c_x, c_y, radius=radius)
    return neighbourhoods, centers


###########################################################


def plot_lines(lines_to_plot, ax=None, title="x:, y: "):
    if ax is not None:
        ax.set_title(title)
    for line in lines_to_plot:
        line = line[~error_default_points(line)]
        if ax is None:
            plt.plot(line[:, 0], line[:, 1])
            plt.savefig("lines_%s.pdf" % (title))
        else:
            ax.plot(line[:, 0], line[:, 1])


def boxplot_characteristics_of_cluster(
    traces_c, ax, metric_names=get_metrics_for_traces()[1]
):
    # _, metric_names = get_metrics_for_traces()
    ax.boxplot([*traces_c.T], labels=metric_names, showfliers=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def plot_lines_for_cluster(
    traces,
    samples,
    clusters,
    n_clusters,
    trace_size,
    limit=10,
    fig_name="cluster_characteristics.pdf",
):
    nrows = 2
    fig, axs = plt.subplots(
        nrows=nrows, ncols=n_clusters, figsize=(n_clusters * 4, nrows * 4), sharey="row"
    )
    for cluster_id in range(n_clusters):
        ax_b = axs[1, cluster_id]
        ax = axs[0, cluster_id]
        boxplot_characteristics_of_cluster(traces[clusters == cluster_id], ax_b)
        samples_c_i = samples[clusters == cluster_id]
        cluster_share = samples_c_i.shape[0] / samples.shape[0]
        select = sample(range(len(samples_c_i)), k=limit)
        plot_lines(
            samples_c_i[select],
            ax=ax,
            title="cluster: %d,      share: %.2f" % (cluster_id, cluster_share),
        )
        ax_b.yaxis.set_tick_params(which="both", labelbottom=True)

    fig.savefig(get_results_filepath(trace_size, fig_name), bbox_inches="tight")
    plt.close(fig)


def sub_figure(ax, x, y, clusters, x_label, y_label, limits=None, zorder=-1):
    scatter = ax.scatter(x, y, c=clusters, cmap="tab10", alpha=0.5, s=2, zorder=zorder)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    remove_spines(ax)
    if limits is None:
        limits = sub_figure_get_limits(x, y)
    else:
        pass
        # print("limits are set")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    return scatter


def sub_figure_get_limits(x, y, s=4):
    m_x, std_x, x_min, x_max = np.mean(x), np.std(x), np.min(x), np.max(x)
    m_y, std_y, y_min, y_max = np.mean(y), np.std(y), np.min(y), np.max(y)
    xmax, xmin = max(-s * std_x + m_x, x_min), min(m_x + std_x * s, x_max)
    ymax, ymin = max(-s * std_y + m_y, y_min), min(m_y + std_y * s, y_max)
    return xmax, xmin, ymax, ymin


def single_plot_components(
    X,
    clusters,
    x_label="t-SNE C1",
    y_label="t-SNE C2",
    file_name=None,
):
    max_number_of_rows = 20000
    if X.shape[0] > max_number_of_rows:
        s = sample(range(X.shape[0]), max_number_of_rows)
        X, clusters = X[s], clusters[s]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    _ = sub_figure(ax, X[:, 0], X[:, 1], clusters, x_label, y_label)
    fig.tight_layout()
    if file_name is not None:
        fig.savefig(file_name)
        plt.close(fig)


def plot_components(X_pca, X_tsne, clusters, file_name=None):
    max_number_of_rows = 20000
    if X_pca.shape[0] > max_number_of_rows:
        rand_select = np.random.choice(
            X_pca.shape[0], size=max_number_of_rows, replace=False
        )
        X_pca, X_tsne, clusters = (
            X_pca[rand_select],
            X_tsne[rand_select],
            clusters[rand_select],
        )
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5 * 2, 5 * 1))

    _ = sub_figure(axs[0], X_pca[:, 0], X_pca[:, 1], clusters, "PC1", "PC2")
    scatter2 = sub_figure(
        axs[1], X_tsne[:, 0], X_tsne[:, 1], clusters, "t-SNE C1", "t-SNE C2"
    )
    legend1 = axs[1].legend(
        *scatter2.legend_elements(),
        loc="best",
        bbox_to_anchor=(0.8, 0.5, 0.5, 0.5),
        title="Cluster"
    )
    fig.add_artist(legend1)
    fig.tight_layout()
    if file_name is not None:
        fig.savefig(file_name)
        plt.close(fig)


def fish_individuality_tsne(
    fish_keys, X_embedded, traces_all, clusters, n_clusters, trace_size
):
    if len(fish_keys) % 2 != 0:
        raise Exception("This method does not support odd lengths of fish keys. ")
    if len(fish_keys) % 4 == 0 and len(fish_keys) > 8:
        nrows = 4
    else:
        nrows = 2

    ncols = len(fish_keys) // nrows
    max_size = 5000
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey=True,
        squeeze=True,
        figsize=(3 * ncols, 3 * nrows),
    )
    axs = np.concatenate(axs)

    for i, f_key in enumerate(fish_keys):
        filter_sum = traces_all["CAMERA_POSITION"] == f_key

        X = X_embedded[filter_sum]
        c = clusters[filter_sum]
        if X.shape[0] > max_size:
            s = sample(range(X.shape[0]), max_size)
            X = X[s]
            c = c[s]
        sub_figure(axs[i], X[:, 0], X[:, 1], c, "t-SNE C1", "t-NSE C2")
        axs[i].set_title(f_key)
    fig.tight_layout()
    fig.savefig(
        get_results_filepath(trace_size, "fish_individuality_tsne_%d" % n_clusters)
    )
    plt.close(fig)


def fish_development_tsne(
    fish_key, days, X_embedded, traces_all, clusters, n_clusters, trace_size
):

    if len(days) % 2 != 0 or len(days) < 8:
        nrows = 1
    elif len(days) % 4 == 0 and len(days) > 8:
        nrows = 4
    else:
        nrows = 2
    ncols = len(days) // nrows
    max_size = 5000
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey=True,
        squeeze=True,
        figsize=(3 * ncols, 3 * nrows),
    )
    if len(axs.shape) > 1:
        axs = np.concatenate(axs)
    fish_filter = traces_all["CAMERA_POSITION"] == fish_key
    X_embedded = X_embedded[fish_filter]
    clusters = clusters[fish_filter]
    traces = traces_all[fish_filter]
    day_before = "0"
    limits = sub_figure_get_limits(X_embedded[:, 0], X_embedded[:, 1])
    for i in range(len(days)):
        filter_sum = (traces["DAY"] > day_before) & (traces["DAY"] <= days[i])
        X = X_embedded[filter_sum]
        c = clusters[filter_sum]
        if X.shape[0] > max_size:
            s = sample(range(X.shape[0]), max_size)
            X = X[s]
            c = c[s]
        if X.shape[0] == 0:
            continue
        sub_figure(axs[i], X[:, 0], X[:, 1], c, "t-SNE C1", "t-NSE C2", limits=limits)
        axs[i].set_title("upto %s" % get_date_string(days[i]))
        day_before = days[i]
    fig.tight_layout()
    fig.savefig(
        get_results_filepath(
            trace_size,
            "fish_development_tsne_%d_%s" % (n_clusters, fish_key),
            subfolder="fish_development",
        )
    )
    plt.close(fig)
