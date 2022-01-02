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

    def __call__(self, data, orient, marshal):

        # TODO resolve width, account for spacing
        # (We want convenience but need to let this work for, e.g. histograms. blegh)
        if "width" not in data:
            data["width"] = 0.8

        # TODO allow user to specify during init
        # TODO this is now repeated with what is in marshal,
        # but we need to use it again in the merge -- UGH UGH UGH
        # TODO note: weird stuff happens when we get this wrong!
        grouping_vars = [v for v in marshal._variables if v in data]

        groups = (
            marshal
            .groupby(data)
            .agg("width", "max", missing=True)
        )

        # TODO need to consider facets too!
        pos = groups[orient]
        width = groups["width"]

        missing = width.isna()
        if not self.fill:
            width = width.fillna(0.8)  # TODO what value here

        width_by_pos = width.groupby(pos, sort=False, observed=True)
        max_by_pos = width_by_pos.max()
        sum_by_pos = width_by_pos.sum()
        width *= pos.map(max_by_pos / sum_by_pos)  # TODO map needed?

        if self.center:
            width = width[~missing]
            each_pos_width = width.groupby(pos, sort=False, observed=True).sum()
        else:
            each_pos_width = max_by_pos

        shift_left = pos.map(each_pos_width) / 2
        recenter = width / 2

        def offset(x):
            # TODO implement gap; scale by width factor?
            # TODO need to account for gap upstream too
            return x.shift(1).add(self.gap).fillna(0).cumsum()

        offsets = width.groupby(pos, sort=False, observed=True).transform(offset)

        new_pos = pos - shift_left + offsets + recenter
        groups["_dodged"] = new_pos
        groups["width"] = width

        out = (
            data
            .drop("width", axis=1)
            .merge(groups, on=grouping_vars)
            .drop(orient, axis=1)
            .rename(columns={"_dodged": orient})
        )

        return out
