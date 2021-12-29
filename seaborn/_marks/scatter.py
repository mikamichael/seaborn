from __future__ import annotations
import numpy as np
import matplotlib as mpl

from seaborn._marks.base import Mark, Feature

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from matplotlib.artist import Artist


class Scatter(Mark):

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

        rng = np.random.default_rng()  # TODO seed?

        def jitter(values, width):
            extent = width / 2
            return values + rng.uniform(-extent, +extent, len(values))

        return df.assign(
            x=jitter(df["x"], self.jitter[0]),
            y=jitter(df["y"], self.jitter[1])
        )

    def _resolve_paths(self, data):

        paths = []
        path_cache = {}
        marker = self._resolve(data, "marker")

        def get_transformed_path(m):
            return m.get_path().transformed(m.get_transform())

        if isinstance(marker, mpl.markers.MarkerStyle):
            return get_transformed_path(marker)

        for m in marker:
            if m not in path_cache:
                path_cache[m] = get_transformed_path(m)
            paths.append(path_cache[m])
        return paths

    def _plot_split(self, keys, data, ax, kws):

        # TODO Not backcompat with allowed (but nonfunctional) univariate plots
        # (That should be solved upstream by defaulting to "" for unset x/y?)
        # (Be mindful of xmin/xmax, etc!)

        kws = kws.copy()

        paths = self._resolve_paths(data)

        fill = self._resolve(data, "fill")
        fill & np.array([m.is_filled() for m in self._resolve(data, "marker")])

        edgecolors = self._resolve_color(data)
        facecolors = self._resolve_color(data, "fill")
        facecolors[~fill, 3] = 0

        linewidths = self._resolve(data, "linewidth")
        pointsize = self._resolve(data, "pointsize")

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
        path = self._resolve_paths(key)

        edgecolor = self._resolve_color(key)
        facecolor = self._resolve_color(key, "fill")

        fill = self._resolve(key, "fill") and marker.is_filled()
        if not fill:
            facecolor = facecolor[0], facecolor[1], facecolor[2], 0

        linewidth = self._resolve(key, "linewidth")
        pointsize = self._resolve(key, "pointsize")

        # PathCollection expects area here, while pointsize is diameter
        # TODO use paths to get true equal area markers?
        size = pointsize ** 2

        return mpl.collections.PathCollection(
            paths=[path],
            sizes=[size],
            facecolors=[facecolor],
            edgecolors=[edgecolor],
            linewidths=[linewidth],
            transform=mpl.transforms.IdentityTransform(),
        )


class Dot(Scatter):

    def __init__(
        self,
        *,
        color=Feature("C0"),
        alpha=Feature(1),  # TODO auto alpha?
        edgecolor=Feature(depend="color"),
        edgealpha=Feature(depend="alpha"),
        marker=Feature("o"),
        pointsize=Feature(6),  # TODO rcParam?
        edgewidth=Feature(.5),  # TODO function of pointsize?
        jitter=None,  # TODO Does Feature always mean mappable?
        **kwargs,  # TODO needed?
    ):

        super().__init__(**kwargs)

        self.features = dict(
            color=color,
            alpha=alpha,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            marker=marker,
            pointsize=pointsize,
            edgewidth=edgewidth,
        )

        self.jitter = jitter  # TODO decide on form of jitter and add type hinting

    def _plot_split(self, keys, data, ax, kws):

        kws = kws.copy()

        markers = self._resolve(data, "marker")
        paths = self._marker_paths(markers)

        facecolors = self._resolve_color(data)
        edgecolors = self._resolve_color(data, "edge")

        filled = np.array([m.is_filled() for m in markers])
        edgecolors[~filled] = facecolors[~filled]

        linewidths = self._resolve(data, "edgewidth")
        pointsize = self._resolve(data, "pointsize")

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

        marker = self._resolve(key, "marker")
        path, = self._marker_paths([marker])

        facecolor = self._resolve_color(key)
        if marker.is_filled():
            edgecolor = self._resolve_color(key, "edge")
        else:
            edgecolor = facecolor

        linewidth = self._resolve(key, "edgewidth")
        size = self._resolve(key, "pointsize") ** 2

        return mpl.collections.PathCollection(
            paths=[path],
            sizes=[size],
            facecolors=[facecolor],
            edgecolors=[edgecolor],
            linewidths=[linewidth],
            transform=mpl.transforms.IdentityTransform(),
        )


class Strip(Scatter):

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
        width=.8,
        **kwargs,  # TODO needed?
    ):

        super().__init__(**kwargs)

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

        self.width = width

    def _adjust(self, df):

        rng = np.random.default_rng()  # TODO seed?
        spacing = np.diff(np.unique(df[self.orient].dropna())).min()
        extent = self.width * spacing / 2
        jitter = rng.uniform(-extent, +extent, len(df))
        values = df[self.orient] + jitter

        return df.assign(**{self.orient: values})
