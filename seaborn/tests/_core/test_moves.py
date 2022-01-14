
from itertools import product

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal

from seaborn._core.moves import Dodge, Jitter
from seaborn._core.rules import categorical_order
from seaborn._core.groupby import GroupBy

import pytest


class MoveFixtures:

    @pytest.fixture
    def df(self, rng):

        n = 50
        data = {
            "x": rng.choice([0., 1., 2., 3.], n),
            "y": rng.normal(0, 1, n),
            "grp2": rng.choice(["a", "b"], n),
            "grp3": rng.choice(["x", "y", "z"], n),
            "width": 0.8
        }
        return pd.DataFrame(data)


class TestJitter(MoveFixtures):

    @pytest.fixture
    def groupby(self):
        return GroupBy([], {})

    def check_same(self, res, df, *cols):
        for col in cols:
            assert_series_equal(res[col], df[col])

    def check_pos(self, res, df, var, limit):

        assert (res[var] != df[var]).all()
        assert (res[var] < df[var] + limit / 2).all()
        assert (res[var] > df[var] - limit / 2).all()

    def test_width(self, df, groupby):

        width = .4
        res = Jitter(width=width)(df, groupby, "x")
        self.check_same(res, df, "y", "grp2", "width")
        self.check_pos(res, df, "x", width * df["width"])

    def test_height(self, df, groupby):

        df["height"] = df["width"]
        height = .4
        res = Jitter(height=height)(df, groupby, "y")
        self.check_same(res, df, "y", "grp2", "width")
        self.check_pos(res, df, "x", height * df["height"])

    def test_x(self, df, groupby):

        val = .2
        res = Jitter(x=val)(df, groupby, "x")
        self.check_same(res, df, "y", "grp2", "width")
        self.check_pos(res, df, "x", val)

    def test_y(self, df, groupby):

        val = .2
        res = Jitter(y=val)(df, groupby, "x")
        self.check_same(res, df, "x", "grp2", "width")
        self.check_pos(res, df, "y", val)

    def test_seed(self, df, groupby):

        kws = dict(width=.2, y=.1, seed=0)
        res1 = Jitter(**kws)(df, groupby, "x")
        res2 = Jitter(**kws)(df, groupby, "x")
        for var in "xy":
            assert_series_equal(res1[var], res2[var])


class TestDodge(MoveFixtures):

    # First some very simple toy examples

    @pytest.fixture
    def toy_df(self):

        data = {
            "x": [0, 0, 1],
            "y": [1, 2, 3],
            "grp": ["a", "b", "b"],
            "width": .8,
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def toy_df_widths(self, toy_df):

        toy_df["width"] = [.8, .2, .4]
        return toy_df

    @pytest.fixture
    def toy_df_facets(self):

        data = {
            "x": [0, 0, 1, 0, 1, 2],
            "y": [1, 2, 3, 1, 2, 3],
            "grp": ["a", "b", "a", "b", "a", "b"],
            "col": ["x", "x", "x", "y", "y", "y"],
            "width": .8,
        }
        return pd.DataFrame(data)

    def test_default(self, toy_df):

        groupby = GroupBy(["x", "grp"], {})
        res = Dodge()(toy_df, groupby, "x")

        assert_array_equal(res["y"], [1, 2, 3]),
        assert_array_almost_equal(res["x"], [-.2, .2, 1.2])
        assert_array_almost_equal(res["width"], [.4, .4, .4])

    def test_fill(self, toy_df):

        groupby = GroupBy(["x", "grp"], {})
        res = Dodge(empty="fill")(toy_df, groupby, "x")

        assert_array_equal(res["y"], [1, 2, 3]),
        assert_array_almost_equal(res["x"], [-.2, .2, 1])
        assert_array_almost_equal(res["width"], [.4, .4, .8])

    def test_drop(self, toy_df):

        groupby = GroupBy(["x", "grp"], {})
        res = Dodge("drop")(toy_df, groupby, "x")

        assert_array_equal(res["y"], [1, 2, 3])
        assert_array_almost_equal(res["x"], [-.2, .2, 1])
        assert_array_almost_equal(res["width"], [.4, .4, .4])

    def test_gap(self, toy_df):

        groupby = GroupBy(["x", "grp"], {})
        res = Dodge(gap=.25)(toy_df, groupby, "x")

        assert_array_equal(res["y"], [1, 2, 3])
        assert_array_almost_equal(res["x"], [-.2, .2, 1.2])
        assert_array_almost_equal(res["width"], [.3, .3, .3])

    def test_widths_default(self, toy_df_widths):

        groupby = GroupBy(["x", "grp"], {})
        res = Dodge()(toy_df_widths, groupby, "x")

        assert_array_equal(res["y"], [1, 2, 3])
        assert_array_almost_equal(res["x"], [-.08, .32, 1.1])
        assert_array_almost_equal(res["width"], [.64, .16, .2])

    def test_widths_fill(self, toy_df_widths):

        groupby = GroupBy(["x", "grp"], {})
        res = Dodge(empty="fill")(toy_df_widths, groupby, "x")

        assert_array_equal(res["y"], [1, 2, 3])
        assert_array_almost_equal(res["x"], [-.08, .32, 1])
        assert_array_almost_equal(res["width"], [.64, .16, .4])

    def test_widths_drop(self, toy_df_widths):

        groupby = GroupBy(["x", "grp"], {})
        res = Dodge(empty="drop")(toy_df_widths, groupby, "x")

        assert_array_equal(res["y"], [1, 2, 3])
        assert_array_almost_equal(res["x"], [-.08, .32, 1])
        assert_array_almost_equal(res["width"], [.64, .16, .2])

    def test_faceted_default(self, toy_df_facets):

        groupby = GroupBy(["x", "grp", "col"], {})
        res = Dodge()(toy_df_facets, groupby, "x")

        assert_array_equal(res["y"], [1, 2, 3, 1, 2, 3])
        assert_array_almost_equal(res["x"], [-.2, .2, .8, .2, .8, 2.2])
        assert_array_almost_equal(res["width"], [.4] * 6)

    def test_faceted_fill(self, toy_df_facets):

        groupby = GroupBy(["x", "grp", "col"], {})
        res = Dodge(empty="fill")(toy_df_facets, groupby, "x")

        assert_array_equal(res["y"], [1, 2, 3, 1, 2, 3])
        assert_array_almost_equal(res["x"], [-.2, .2, 1, 0, 1, 2])
        assert_array_almost_equal(res["width"], [.4, .4, .8, .8, .8, .8])

    def test_faceted_drop(self, toy_df_facets):

        groupby = GroupBy(["x", "grp", "col"], {})
        res = Dodge(empty="drop")(toy_df_facets, groupby, "x")

        assert_array_equal(res["y"], [1, 2, 3, 1, 2, 3])
        assert_array_almost_equal(res["x"], [-.2, .2, 1, 0, 1, 2])
        assert_array_almost_equal(res["width"], [.4] * 6)

    def test_orient(self, toy_df):

        df = toy_df.assign(x=toy_df["y"], y=toy_df["x"])

        groupby = GroupBy(["y", "grp"], {})
        res = Dodge("drop")(df, groupby, "y")

        assert_array_equal(res["x"], [1, 2, 3])
        assert_array_almost_equal(res["y"], [-.2, .2, 1])
        assert_array_almost_equal(res["width"], [.4, .4, .4])

    # Now tests with slightly more complicated data

    @pytest.mark.parametrize("grp", ["grp2", "grp3"])
    def test_single_semantic(self, df, grp):

        groupby = GroupBy(["x", grp], {})
        res = Dodge()(df, groupby, "x")

        levels = categorical_order(df[grp])
        w, n = 0.8, len(levels)

        shifts = np.linspace(0, w - w / n, n)
        shifts -= shifts.mean()

        assert_series_equal(res["y"], df["y"])
        assert_series_equal(res["width"], df["width"] / n)

        for val, shift in zip(levels, shifts):
            rows = df[grp] == val
            assert_series_equal(res.loc[rows, "x"], df.loc[rows, "x"] + shift)

    def test_two_semantics(self, df):

        groupby = GroupBy(["x", "grp2", "grp3"], {})
        res = Dodge()(df, groupby, "x")

        levels = categorical_order(df["grp2"]), categorical_order(df["grp3"])
        w, n = 0.8, len(levels[0]) * len(levels[1])

        shifts = np.linspace(0, w - w / n, n)
        shifts -= shifts.mean()

        assert_series_equal(res["y"], df["y"])
        assert_series_equal(res["width"], df["width"] / n)

        for (v2, v3), shift in zip(product(*levels), shifts):
            rows = (df["grp2"] == v2) & (df["grp3"] == v3)
            assert_series_equal(res.loc[rows, "x"], df.loc[rows, "x"] + shift)
