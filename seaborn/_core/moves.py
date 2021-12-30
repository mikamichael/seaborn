from __future__ import annotations
import numpy as np

from dataclasses import dataclass
from typing import Optional
from pandas import DataFrame, Series


class Move:

    pass


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
