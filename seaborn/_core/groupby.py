
import pandas as pd

from seaborn._core.rules import categorical_order


class GroupBy:

    def __init__(self, variables, scales):

        # TODO call self.order, use in function that consumes this object?

        self._orderings = {
            var: scales[var].order if var in scales else None
            for var in variables
        }

    def _group(self, data):

        # TODO cache this? Or do in __init__? Do we need to call on different data?

        levels = {}
        for var in (v for v in self._orderings if v in data):
            order = self._orderings.get(var)
            if order is None:
                order = categorical_order(data[var])
            levels[var] = order

        groups = pd.MultiIndex.from_product(levels.values(), names=levels.keys())
        groupmap = {g: i for i, g in enumerate(groups)}

        return groups, groupmap

    def agg(self, data, col, func, missing=False):

        groups, groupmap = self._group(data)

        res = (
            data
            .set_index(groups.names)
            .groupby(groupmap)
            .agg({col: func})
        )

        res = res.set_index(groups[res.index])
        if missing:
            res = res.reindex(groups)

        return res.reset_index()
