"""Accuracy properties of weighting and synthesis (two-level Arizona example).

These are thresholds, not snapshots: they check that the results are *right*,
independently of what any earlier version produced.  The first test would have
caught the 3.0.1/3.0.2 regression, whose forward IPU sweep left household
totals 4.5 % off on average (14 % max) in two-level runs.
"""
import os

import numpy as np
import pandas as pd
import pytest

from conftest import read_marginals, run_first_scenario, run_single, sha256

REGION = 9
GEO_CONTROLS = {"household": ["hinc", "hsize", "hhtotals"], "groupquarter": ["gqtotals"],
                "person": ["psex", "page", "empl", "ptotals"]}
REGION_CONTROLS = {"household": ["rhinc", "rhsize", "hhrtotals"], "groupquarter": ["gqrtotals"],
                   "person": ["rpsex", "rpage", "rempl", "rptotals"]}
FILES = {"household": "household_sample.csv", "groupquarter": "groupquarter_sample.csv",
         "person": "person_sample.csv"}


def _apd(achieved, target):
    a, t = np.asarray(achieved, float), np.asarray(target, float)
    keep = t > 0
    e = np.abs(a[keep] - t[keep]) / t[keep]
    return e.mean(), e.max()


def _weighted_fit(workdir, out_dir, level):
    az = os.path.join(workdir, "Arizona")
    w = pd.read_csv(os.path.join(out_dir, "weights.csv")).set_index("hid")
    w.columns = [int(c) for c in w.columns]
    rgm = pd.read_csv(os.path.join(az, "region_geo_mapping.csv"))
    geos = rgm.loc[rgm.region == REGION, "geo"].tolist()
    controls = GEO_CONTROLS if level == "geo" else REGION_CONTROLS
    prefix = "" if level == "geo" else "region_"
    res = {}
    for entity, vars_ in controls.items():
        sample = pd.read_csv(os.path.join(az, FILES[entity]))
        marg = read_marginals(os.path.join(az, f"{prefix}{entity}_marginals.csv"))
        ids = geos if level == "geo" else [REGION]
        A, T = [], []
        for gid in ids:
            wcol = w[gid] if level == "geo" else w[geos].sum(axis=1)
            ww = sample["hid"].map(wcol).fillna(0.0)
            for var in vars_:
                tot = ww.groupby(sample[var]).sum()
                for (v, cat), tgt in marg.loc[gid].items():
                    if v == var:
                        A.append(tot.get(int(cat), 0.0))
                        T.append(tgt)
        res[entity] = _apd(A, T)
    return res


@pytest.fixture(scope="module")
def main_run(tmp_path_factory):
    from conftest import DATA, copy_inputs
    workdir = copy_inputs(DATA, tmp_path_factory.mktemp("acc") / "data")
    project, scenario = run_first_scenario(workdir, "configuration_arizona.yaml")
    out = [d for d in os.listdir(os.path.join(workdir, "Arizona")) if d.endswith("Arizona TAZ Scenario")]
    return workdir, os.path.join(workdir, "Arizona", out[0]), project, scenario


@pytest.mark.parametrize("level", ["geo", "region"])
def test_two_level_household_and_gq_totals_are_matched(main_run, level):
    workdir, out, _, _ = main_run
    fit = _weighted_fit(workdir, out, level)
    for entity in ("household", "groupquarter"):
        mean, mx = fit[entity]
        assert mx < 1e-3, f"{level} {entity}: max deviation {mx:.3%} (mean {mean:.3%}) exceeds 0.1%"


@pytest.mark.parametrize("level", ["geo", "region"])
def test_two_level_person_totals_within_known_range(main_run, level):
    # IPU cannot match household and person totals exactly with this sample; 3.0.3/3.0.4
    # reach ~2.7 % mean / ~5 % max.  The bounds catch a material regression.
    workdir, out, _, _ = main_run
    mean, mx = _weighted_fit(workdir, out, level)["person"]
    assert mean < 0.04 and mx < 0.07, f"{level} person fit degraded: mean {mean:.3%}, max {mx:.3%}"


def test_ipf_constraints_reproduce_marginals(main_run):
    workdir, _, project, scenario = main_run
    ipf = scenario.run_ipf_obj
    az = os.path.join(workdir, "Arizona")
    for entity, vars_ in GEO_CONTROLS.items():
        cons = ipf.geo_constraints_dict[entity]          # cells x geos
        marg = read_marginals(os.path.join(az, f"{entity}_marginals.csv"))
        for gid in cons.columns:
            for k, var in enumerate(vars_):
                level = cons.index.names.index(var)
                fitted = cons[gid].groupby(level=level).sum()
                for (v, cat), tgt in marg.loc[gid].items():
                    if v == var and tgt > 0:
                        assert abs(fitted.get(int(cat), 0.0) - tgt) / tgt < 1e-3, (entity, gid, var, cat)


def test_bucket_rounding_preserves_totals(main_run):
    # geo_constraints_dict[entity]: IPF cells x geos;  geo_frequencies: geos x cells (integerized)
    ipf = main_run[3].run_ipf_obj
    freq = ipf.geo_frequencies
    for entity in ("household", "groupquarter"):
        cons = ipf.geo_constraints_dict[entity]
        cells = list(cons.index)
        for gid in cons.columns:
            c = cons[gid].to_numpy(float)
            f = np.array([float(freq[cell].loc[gid]) for cell in cells])  # columns are tuples
            assert np.all(f == np.round(f)), "frequencies must be integers"
            assert np.all(np.abs(f - c) < 1.0 + 1e-9), "each cell within 1 of its IPF value"
            assert abs(f.sum() - c.sum()) <= 0.5 + 1e-6, "total within 0.5 of the IPF total"


def test_synthetic_households_reproduce_integerized_table(main_run):
    _, _, _, scenario = main_run
    freq = scenario.run_ipf_obj.geo_frequencies
    stacked = scenario.run_reweighting_obj.geo_stacked
    for gid, rows in scenario.draw_population_obj.geo_id_rows_syn_dict.items():
        drawn = stacked.take(rows)
        for col in freq.columns:
            assert int((drawn[col] > 0).sum()) == int(freq[col].loc[gid]), (gid, col)


def test_synthetic_records_are_real_sample_records(main_run):
    workdir, out, _, _ = main_run
    az = os.path.join(workdir, "Arizona")
    hh = pd.read_csv(os.path.join(out, "housing_synthetic.csv"))
    sample_ids = set(pd.read_csv(os.path.join(az, "household_sample.csv"))["hid"]) | \
        set(pd.read_csv(os.path.join(az, "groupquarter_sample.csv"))["hid"])
    assert set(hh["hid"]) <= sample_ids          # no invented (structurally impossible) records
    pp = pd.read_csv(os.path.join(out, "person_synthetic.csv"))
    persons = pd.read_csv(os.path.join(az, "person_sample.csv")).groupby("hid").size()
    drawn = hh[hh["hid"].isin(persons.index)].groupby("hid").size()
    expected = int((drawn * persons.reindex(drawn.index)).sum())
    assert len(pp) == expected                   # each drawn household brings all its persons


def test_runs_are_reproducible(tmp_path):
    from conftest import DATA, copy_inputs
    a = run_single(copy_inputs(DATA, tmp_path / "a"), "configuration_arizona.yaml")
    b = run_single(copy_inputs(DATA, tmp_path / "b"), "configuration_arizona.yaml")
    for fn in ("weights.csv", "housing_synthetic.csv", "person_synthetic.csv"):
        assert sha256(os.path.join(a, fn)) == sha256(os.path.join(b, fn)), fn


def test_numpy_global_random_state_is_left_alone(arizona_dir):
    from conftest import run_config
    np.random.seed(7)
    expected = np.random.random(3)
    np.random.seed(7)
    run_config(arizona_dir, "configuration_arizona_single_level.yaml")
    assert np.array_equal(np.random.random(3), expected)
