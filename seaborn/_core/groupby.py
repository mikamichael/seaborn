
import pandas as pd

from seaborn._core.rules import categorical_order


class Marshal:

    def __init__(self, variables, scales):

        self._variables = variables
        self._orderings = {
            var: scales[var].order if var in scales else None
            for var in variables
        }

    def groupby(self, data, grouping_vars=None):

        if grouping_vars is None:
            grouping_vars = self._variables

        levels = {}
        for var in (v for v in grouping_vars if v in data):
            order = self._orderings.get(var)
            if order is None:
                order = categorical_order(data[var])
            levels[var] = order

        groups = pd.MultiIndex.from_product(levels.values(), names=levels.keys())
        return GroupBy(data, groups)


class GroupBy:

    def __init__(self, data, groups):

        self._data = data
        self._groups = groups
        self._groupmap = {g: i for i, g in enumerate(groups)}

    def agg(self, col, func, missing=False):

        res = (
            self._data
            .set_index(self._groups.names)
            .groupby(self._groupmap)
            .agg({col: func})
        )

        res = res.set_index(self._groups[res.index])
        if missing:
            res = res.reindex(self._groups)

        return res.reset_index()
