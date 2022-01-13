from __future__ import annotations
import numpy as np

from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal, Optional
    from pandas import DataFrame
    from seaborn._core.groupby import GroupBy


@dataclass
class Move:

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: Literal["x", "y"],
    ) -> DataFrame:
        raise NotImplementedError


@dataclass
class Jitter(Move):

    width: float = 0
    height: float = 0

    x: float = 0
    y: float = 0

    seed: Optional[int] = None

    # TODO what is the best way to have a reasonable default?
    # The problem is that "reasonable" seems dependent on the mark

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: Literal["x", "y"],
    ) -> DataFrame:

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
            data["x"] = jitter(data["x"], 1, self.x)

        if self.y:
            data["y"] = jitter(data["y"], 1, self.y)

        return data


@dataclass
class Dodge(Move):

    empty: Literal["keep", "drop", "fill"] = "keep"
    gap: float = 0

    # TODO accept just a str here?
    by: Optional[list[str]] = None

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: Literal["x", "y"],
    ) -> DataFrame:

        # TODO change _orderings to public attribute?
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
            return w.shift(1).fillna(0).cumsum() + (w - w.sum()) / 2

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
