from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import matplotlib as mpl

from seaborn._marks.base import Mark, Feature

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Union
    from matplotlib.artist import Artist

    MappableStr = Union[str, Feature]
    MappableFloat = Union[float, Feature]
    MappableColor = Union[str, tuple, Feature]


class Point(Mark):  # TODO types

    supports = ["color"]

    def __init__(
        self,
        *,
        color=Feature("C0"),
        alpha=Feature(1),  # TODO auto alpha?
        fill=Feature(True),
        fillcolor=Feature(depend="color"),
        fillalpha=Feature(.2),
        marker=Feature(rc="scatter.marker"),
        pointsize=Feature(5),  # TODO rcParam?
        linewidth=Feature(.75),  # TODO rcParam?
        jitter=None,  # TODO Does Feature always mean mappable?
        **kwargs,  # TODO needed?
    ):

        super().__init__(**kwargs)

        # TODO should this use SEMANTICS as the source of possible features?
        self.features = dict(
            color=color,
            alpha=alpha,
            fill=fill,
            fillcolor=fillcolor,
            fillalpha=fillalpha,
            marker=marker,
            pointsize=pointsize,
            linewidth=linewidth,
        )

        self.jitter = jitter  # TODO decide on form of jitter and add type hinting

    def _adjust(self, df):

        if self.jitter is None:
            return df

        x, y = self.jitter  # TODO maybe not format, and do better error handling

        # TODO maybe accept a Jitter class so we can control things like distribution?
        # If we do that, should we allow convenient flexibility (i.e. (x, y) tuple)
        # in the object interface, or be simpler but more verbose?

        # TODO note that some marks will have multiple adjustments
        # (e.g. strip plot has both dodging and jittering)

        # TODO native scale of jitter? maybe just for a Strip subclass?

        rng = np.random.default_rng()  # TODO seed?

        n = len(df)
        x_jitter = 0 if not x else rng.uniform(-x, +x, n)
        y_jitter = 0 if not y else rng.uniform(-y, +y, n)

        # TODO: this fails if x or y are paired. Apply to all columns that start with y?
        return df.assign(x=df["x"] + x_jitter, y=df["y"] + y_jitter)

    def _plot_split(self, keys, data, ax, kws):

        # TODO Not backcompat with allowed (but nonfunctional) univariate plots
        # (That should be solved upstream by defaulting to "" for unset x/y?)
        # (Be mindful of xmin/xmax, etc!)

        kws = kws.copy()

        markers = self._resolve(data, "marker")
        fill = self._resolve(data, "fill")
        fill & np.array([m.is_filled() for m in markers])

        edgecolors = self._resolve_color(data)
        facecolors = self._resolve_color(data, "fill")
        facecolors[~fill, 3] = 0

        linewidths = self._resolve(data, "linewidth")
        pointsize = self._resolve(data, "pointsize")

        paths = []
        path_cache = {}
        for m in markers:
            if m not in path_cache:
                path_cache[m] = m.get_path().transformed(m.get_transform())
            paths.append(path_cache[m])

        sizes = pointsize ** 2
        offsets = data[["x", "y"]].to_numpy()

        points = mpl.collections.PathCollection(
            paths=paths,
            sizes=sizes,
            offsets=offsets,
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=linewidths,
            transOffset=ax.transData,
            transform=mpl.transforms.IdentityTransform(),
        )
        ax.add_collection(points)

    def _legend_artist(self, variables: list[str], value: Any) -> Artist:

        key = {v: value for v in variables}

        # TODO do we need to abstract "get feature kwargs"?
        marker = self._resolve(key, "marker")
        path = marker.get_path().transformed(marker.get_transform())

        edgecolor = self._resolve_color(key)
        facecolor = self._resolve_color(key, "fill")

        fill = self._resolve(key, "fill") and marker.is_filled()
        if not fill:
            facecolor = facecolor[0], facecolor[1], facecolor[2], 0

        linewidth = self._resolve(key, "linewidth")
        pointsize = self._resolve(key, "pointsize")
        size = pointsize ** 2

        return mpl.collections.PathCollection(
            paths=[path],
            sizes=[size],
            facecolors=[facecolor],
            edgecolors=[edgecolor],
            linewidths=[linewidth],
            transform=mpl.transforms.IdentityTransform(),
        )


@dataclass
class Line(Mark):

    color: MappableColor = Feature("C0", groups=True)
    alpha: MappableFloat = Feature(1, groups=True)
    linewidth: MappableFloat = Feature(rc="lines.linewidth", groups=True)
    linestyle: MappableStr = Feature(rc="lines.linestyle", groups=True)

    sort: bool = True

    def _plot_split(self, keys, data, ax, kws):

        keys = self.resolve_features(keys)

        if self.sort:
            data = data.sort_values(self.orient)

        line = mpl.lines.Line2D(
            data["x"].to_numpy(),
            data["y"].to_numpy(),
            color=keys["color"],
            linewidth=keys["linewidth"],
            **kws,
        )
        ax.add_line(line)

    def _legend_artist(self, variables, value):

        key = self.resolve_features({v: value for v in variables})

        return mpl.lines.Line2D(
            [], [],
            color=key["color"],
            linewidth=key["linewidth"],
            linestyle=key["linestyle"],
        )


@dataclass
class Area(Mark):

    color: MappableColor = Feature("C0", groups=True)
    alpha: MappableFloat = Feature(1, groups=True)

    def _plot_split(self, keys, data, ax, kws):

        keys = self.resolve_features(keys)
        kws["facecolor"] = self._resolve_color(keys)
        kws["edgecolor"] = self._resolve_color(keys)

        # TODO how will orient work here?
        # Currently this requires you to specify both orient and use y, xmin, xmin
        # to get a fill along the x axis. Seems like we should need only one of those?
        # Alternatively, should we just make the PolyCollection manually?
        if self.orient == "x":
            ax.fill_between(data["x"], data["ymin"], data["ymax"], **kws)
        else:
            ax.fill_betweenx(data["y"], data["xmin"], data["xmax"], **kws)
