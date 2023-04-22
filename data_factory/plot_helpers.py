

# A helper function that removes the spline from the given axis ax.
def remove_spines(ax):
    """Remove the spines from the given axis ax."""
    for s in ax.spines.values():
        s.set_visible(False)