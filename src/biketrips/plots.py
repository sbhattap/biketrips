"""Figures for the notebooks. Each function draws on an axis and returns it."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biketrips.features import to_gdf


def _ax(ax, figsize=(7, 4)):
    return ax if ax is not None else plt.subplots(figsize=figsize)[1]


def bar(series, ax=None, title="", ylabel="", log=False, horizontal=False):
    ax = _ax(ax)
    series.plot.barh(ax=ax, log=log) if horizontal else series.plot.bar(ax=ax, log=log)
    ax.set(title=title, ylabel=ylabel if not horizontal else "", xlabel=ylabel if horizontal else "")
    return ax


def temporal(df, axes=None):
    axes = axes if axes is not None else plt.subplots(1, 2, figsize=(14, 4))[1]
    bar(df.groupby("day")["n"].sum(), axes[0], "Trips per day (start time)", "trips")
    bar(df.groupby("hour")["n"].sum(), axes[1], "Trips per hour of day", "trips")
    return axes


def hist(values, ax=None, bins=50, title="", xlabel="", log=False):
    ax = _ax(ax)
    ax.hist(values, bins=bins, log=log)
    ax.set(title=title, xlabel=xlabel, ylabel="count")
    return ax


def cell_map(cells, column, ax=None, title="", log=False, cmap="viridis", **kw):
    """Choropleth of an H3 cell table (needs an `h3` column)."""
    ax = _ax(ax, (9, 6))
    g = to_gdf(cells)
    if log:
        g[column] = np.log1p(g[column])
    g.plot(column=column, ax=ax, cmap=cmap, legend=True, linewidth=0, **kw)
    ax.set_title(title + (" (log1p)" if log else ""))
    ax.set_axis_off()
    return ax


def scatter(x, y, ax=None, title="", xlabel="", ylabel="", loglog=False, diagonal=False):
    ax = _ax(ax, (5, 5))
    ax.scatter(x, y, s=4, alpha=0.4)
    if loglog:
        ax.set(xscale="symlog", yscale="symlog")
    if diagonal:
        lim = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
        ax.plot(lim, lim, "k--", lw=1)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return ax


def importance(series, ax=None, top=20, title="Feature importance (total gain)"):
    ax = _ax(ax, (7, 7))
    s = series.head(top)[::-1]
    return bar(s / s.max(), ax, title, "relative gain", horizontal=True)
