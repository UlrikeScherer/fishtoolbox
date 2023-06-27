import numpy as np
import matplotlib.pyplot as plt

# A helper function that removes the spline from the given axis ax.
def remove_spines(ax):
    """Remove the spines from the given axis ax."""
    for s in ax.spines.values():
        s.set_visible(False)

def average_fit_plot(t, polys, ax=plt, title="",xlabel="x", ylabel="y", alpha=0.5):
    time = np.linspace(t.min(),t.max(),100)
    if type(alpha) is float:
        alpha=[alpha]* len(polys)
    colors = get_custom_colors(len(polys))
    for i in range(len(polys)):
        ax.plot(time, np.polyval(polys[i],time), alpha=alpha[i], lw=0.7, color=colors[i])
    ax.plot(time, np.polyval(polys.mean(axis=0),time), c="k", label="average")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax

def get_custom_colors(cnt):
    colors = ['darkblue', 'orange', 'red']
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom_cmap', colors)
    # Generate color indices
    indices = np.linspace(0, 1, cnt)
    # Select colors from colormap using indices
    colors = cmap(indices)
    return colors

def sparse_scatter_plot(df, ax):
    colors = get_custom_colors(df.columns.size)
    for i,col in enumerate(df.columns):
        x = df.index.to_numpy()
        y = df[col]
        ax.scatter(x=x,y=y,s=0.5, color=colors[i])

def get_polys(df):
    t = df.index
    return np.array([np.polyfit(
        t[np.isfinite(df[c])], df[c][np.isfinite(df[c])], deg=2
    ) for c in df.columns[:45]])