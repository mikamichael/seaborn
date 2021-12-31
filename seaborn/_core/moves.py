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

    def __call__(self, data: DataFrame, orient: str) -> DataFrame:

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

    fill: bool = False
    gap: float = 0

    def __call__(self, data: DataFrame, orient: str) -> DataFrame:

        data = data.copy(deep=False)

        # Initialize vales for bar shape/location parameterization
        # TODO
        # (a) how to distinguish between actual and relative width?
        # (b) how to set these defaults upstream using Mark features
        if "width" not in data:
            spacing = np.min(np.diff(np.unique(data[orient])))
            data["width"] = .8 * spacing

        # TODO  Now we need to know the levels of the grouping variables, hmmm.
        # TODO maybe instead of that we have the dataframe sorted by categorical order?

        # TODO implement self.gap

        width_by_pos = data.groupby(orient, sort=False)["width"]
        if self.fill:  # Not great name given other "fill"
            # TODO e.g. what should we do here with empty categories?
            # is it too confusing if we appear to ignore "dodgefill",
            # or is it inconsistent with behavior elsewhere?
            max_by_pos = width_by_pos.max()
            sum_by_pos = width_by_pos.sum()
        else:
            # TODO meanwhile here, we do get empty space, but
            # it is always to the right of the bars that are there
            max_width = data["width"].max()
            max_by_pos = {p: max_width for p, _ in width_by_pos}
            max_sum = width_by_pos.sum().max()
            sum_by_pos = {p: max_sum for p, _ in width_by_pos}

        data["width"] = width_by_pos.transform(
            lambda x: (x / sum_by_pos[x.name]) * max_by_pos[x.name]
        )

        # TODO maybe this should be building a mapping dict for pos?
        # (It is probably less relevent for bars, but what about e.g.
        # a dense stripplot, where we'd be doing a lot more operations
        # than we need to be doing this way.
        data[orient] = (
            data[orient]
            - data[orient].map(max_by_pos) / 2
            + width_by_pos.transform(
                lambda x: x.shift(1).fillna(0).cumsum()
            )
            + data["width"] / 2
        )

        return data
