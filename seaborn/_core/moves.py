from __future__ import annotations
import numpy as np

from dataclasses import dataclass
from typing import Optional
from pandas import DataFrame, Series


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

    # TODO what is the best way to have a reasonable default?
    # The problem is that "reasonable" seems dependent on the mark

    def _add_jitter(
        self, data: Series, space: float | None, scale: float, rng
    ) -> Series:

        noise = rng.uniform(-.5, +.5, len(data))

        if space is None:
            space = np.diff(np.unique(data.dropna())).min()

        offsets = noise * scale * space
        return data + offsets

    def __call__(self, data: DataFrame, orient: str, marshal) -> DataFrame:

        data = data.copy(deep=False)

        rng = np.random.default_rng(self.seed)

        width_coord = orient
        height_coord = {"x": "y", "y": "x"}[orient]

        # TODO how do we make it such that `width` as a settable parameter
        # of the Mark (e.g. for boxplots) initializes the default in data here

        if self.width:
            data[width_coord] = self._add_jitter(
                data[width_coord], data.get("width"), self.width, rng
            )

        if self.height:
            data[height_coord] = self._add_jitter(
                data[height_coord], data.get("height"), self.height, rng
            )

        if self.x:
            data["x"] = self._add_jitter(data["x"], 1, self.x, rng)

        if self.y:
            data["y"] = self._add_jitter(data["y"], 1, self.y, rng)

        return data


@dataclass
class Dodge(Move):

    fill: bool = False  # TODO fill original width with dodged objects
    center: bool = False  # TODO remove empty spaces and center remaining objects
    gap: float = 0

    # TODO allow user to pass in grouping variables; e.g. with three variables
    # you may want to dodge by one of them (two strips) and then jitter the others

    def __call__(self, data, orient, groupby):

        # TODO resolve width, account for spacing
        # (We want convenience but need to let this work for, e.g. histograms. blegh)
        if "width" not in data:
            data["width"] = 0.8

        # TODO allow user to specify during init, change _orderings to public attribute?
        grouping_vars = [v for v in groupby._orderings if v in data]

        # TODO what value to fill missing widths with for default behavior
        missing_width = 0 if self.center else 0.8

        groups = (
            groupby
            .agg(data, "width", "max", missing=not self.fill)
            .fillna({"width": missing_width})
        )

        def groupby_pos(s):
            grouper = [groups[v] for v in [orient, "col", "row"] if v in data]
            return s.groupby(grouper, sort=False, observed=True)

        def dodge(w):
            # TODO implement gap; scale by width factor?
            # TODO need to account for gap upstream too
            return w.shift(1).fillna(0).cumsum() + (w - w.sum()) / 2

        new_widths = groupby_pos(groups["width"]).transform(lambda x: x / len(x))
        movements = groupby_pos(new_widths).transform(dodge)

        if self.gap:
            new_widths *= 1 - self.gap

        groups["_dodged"] = groups[orient] + movements
        groups["width"] = new_widths

        out = (
            data
            .drop("width", axis=1)
            .merge(groups, on=grouping_vars)
            .drop(orient, axis=1)
            .rename(columns={"_dodged": orient})
        )

        return out
