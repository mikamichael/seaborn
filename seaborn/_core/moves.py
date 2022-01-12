from __future__ import annotations
import numpy as np

from dataclasses import dataclass
from typing import Optional
from pandas import DataFrame, Series


@dataclass
class Move:

    def __call__(self, data: DataFrame, orient: str) -> DataFrame:
        # TODO type orient as a Literal["x", "y"]
        raise NotImplementedError


@dataclass
class Jitter(Move):

    # Alt name ... relwidth? rwidth? wspace?
    width: Optional[float] = None
    height: Optional[float] = None

    x: Optional[float] = None
    y: Optional[float] = None
    seed: Optional[int] = None

    # TODO would be nice to put in base class but then it becomes the first positional arg
    # and we want that to be width
    by: Optional[list[str]] = None  # TODO or str, and convert to list internally (post_init?)

    # TODO what is the best way to have a reasonable default?
    # The problem is that "reasonable" seems dependent on the mark

    def __call__(self, data: DataFrame, groupby, orient: str) -> DataFrame:

        data = data.copy(deep=False)

        rng = np.random.default_rng(self.seed)

        wcoord = orient
        hcoord = {"x": "y", "y": "x"}[orient]

        def jitter(data, space, scale):

            noise = rng.uniform(-.5, +.5, len(data))

            # TODO this should be handled upstream to always define width/height
            if space is None:
                space = np.diff(np.unique(data.dropna())).min()

            offsets = noise * scale * space
            return data + offsets

        # TODO how do we make it such that `width` as a settable parameter
        # of the Mark (e.g. for boxplots) initializes the default in data here

        if self.width:
            data[wcoord] = jitter(data[wcoord], data.get("width"), self.width)

        if self.height:
            data[hcoord] = jitter(data[hcoord], data.get("height"), self.height)

        if self.x:
            data["x"] = self._add_jitter(data["x"], 1, self.x)

        if self.y:
            data["y"] = self._add_jitter(data["y"], 1, self.y)

        return data


@dataclass
class Dodge(Move):

    empty: str = "keep"  # TODO annotate with Literal?
    gap: float = 0

    by: Optional[list[str]] = None  # TODO or str, and convert to list internally (post_init?)

    # TODO allow user to pass in grouping variables; e.g. with three variables
    # you may want to dodge by one of them (two strips) and then jitter the others

    def __call__(self, data, groupby, orient):

        # TODO allow user to specify during init, change _orderings to public attribute?
        grouping_vars = [v for v in groupby._orderings if v in data]

        groups = (
            groupby
            .agg(data, "width", "max", missing=self.empty != "fill")
        )

        def groupby_pos(s):
            grouper = [groups[v] for v in [orient, "col", "row"] if v in data]
            return s.groupby(grouper, sort=False, observed=True)

        def scale_widths(w):
            # TODO what value to fill missing widths??? Hard problem...
            # TODO short circuit this if outer widths has no variance?
            space = 0 if self.empty == "fill" else w.mean()
            filled = w.fillna(space)
            scale = filled.max()
            norm = filled.sum()
            if self.empty == "keep":
                w = filled
            return w / norm * scale

        def widths_to_offsets(w):
            # TODO implement gap; scale by width factor?
            # TODO need to account for gap upstream too
            return w.shift(1).fillna(0).cumsum() + (w - w.sum()) / 2

        # new_widths = groupby_pos(groups["width"]).transform(lambda w: w / len(w))
        new_widths = groupby_pos(groups["width"]).transform(scale_widths)
        offsets = groupby_pos(new_widths).transform(widths_to_offsets)

        if self.gap:
            new_widths *= 1 - self.gap

        groups["_dodged"] = groups[orient] + offsets
        groups["width"] = new_widths

        out = (
            data
            .drop("width", axis=1)
            .merge(groups, on=grouping_vars, how="left")
            .drop(orient, axis=1)
            .rename(columns={"_dodged": orient})
        )

        return out
