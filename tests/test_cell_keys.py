"""Entities with different numbers of control variables (fixed in 3.0.6).

A cell key is (entity, category of each control variable), so its length
depends on the entity.  Up to 3.0.5 the keys sat in pandas MultiIndexes and the
entity tables were joined with ``pd.concat``; with keys of different lengths
the result depended on pandas:

* household + person without a group-quarter entity: the longer person keys
  were truncated, so writing the synthetic population failed ("Not allowed to
  merge between different levels"), and the fit summary reported every person
  margin as 100 % off;
* with two household controls the truncated person keys collided and IPU
  crashed ("cannot convert the series to <class 'float'>");
* household only, synthesize: true: writing the synthetic population failed.

Only some layouts happened to work - the Arizona example (its one-variable
group-quarter entity made pandas fall back to a flat index of tuples) or the
entities listed person first.  The fingerprints in golden_306.json were
produced by 3.0.5 in such a working layout (see make_golden_306.py); 3.0.6 must
give the same files with the usual order.  weights.csv is compared through a
numeric fingerprint (conftest.weights_fingerprint) because its last decimal can
differ between machines and pandas versions; the other files byte for byte.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from conftest import (DATA, REPO, SURVEY, SURVEY_HOUSEHOLD_CONTROLS, SURVEY_TWO_LEVEL, arizona_without_group_quarters,
                      copy_inputs, read_marginals, run_single, sha256, survey_variant, weights_fingerprint,
                      weights_fingerprint_mismatch)
from popgen.cells import flat_keys, stack_cells

with open(os.path.join(REPO, "tests", "golden", "golden_306.json")) as _f:
    GOLDEN_306 = json.load(_f)


def assert_matches_305(out, name):
    """Outputs of the run in *out* against the 3.0.5 fingerprints of golden_306.json[runs][name]."""
    ref = GOLDEN_306["runs"][name]
    changed = [fn for fn, h in ref["files"].items() if sha256(os.path.join(out, fn)) != h]
    assert not changed, f"{name}: outputs differ from the 3.0.5 reference: {changed}"
    problems = weights_fingerprint_mismatch(weights_fingerprint(os.path.join(out, "weights.csv")), ref["weights"])
    assert not problems, f"{name}: weights differ from the 3.0.5 reference: {problems}"


@pytest.mark.parametrize("name", sorted(n for n in GOLDEN_306["runs"] if n.startswith("arizona")))
def test_arizona_without_group_quarters_matches_305_reference(name, tmp_path):
    workdir = copy_inputs(DATA, tmp_path / "data")
    out = run_single(workdir, arizona_without_group_quarters(workdir, name))
    assert_matches_305(out, name)

    fit = pd.read_csv(os.path.join(out, "reweighting_fit_summary.csv"))
    geo = fit[fit["level"] == "geo"].set_index("entity")
    assert (geo.loc["household", "margin_max_abs_rel_dev"] < 1e-3).all()
    # person margins only through the household weights: the IPU corner solution, a few percent
    assert (geo.loc["person", "margin_max_abs_rel_dev"] < 0.10).all()


@pytest.mark.parametrize("entities", [["household", "person"], ["person", "household"]],
                         ids=["household_first", "person_first"])
def test_survey_two_levels_matches_305_reference(entities, survey_dir):
    out = run_single(survey_dir, survey_variant(survey_dir, "two_levels", entities, ["person"], SURVEY_TWO_LEVEL, True))
    assert_matches_305(out, "survey_two_level_synthesis")


def test_fit_summary_reports_person_margins(survey_dir):
    # 3.0.5: the person keys were truncated, every person margin was "missing"
    # and reported as 1.0 (100 % off) although the weights were right.
    out = run_single(survey_dir, survey_variant(survey_dir, "two_levels_weights", ["household", "person"], ["person"],
                                                SURVEY_TWO_LEVEL, False))
    fit = pd.read_csv(os.path.join(out, "reweighting_fit_summary.csv"))
    person = fit[fit["entity"] == "person"]
    assert len(person) == 4                                   # region and geo level, two geographies
    assert (person["margin_max_abs_rel_dev"] < 1e-2).all(), person
    assert person["worst_cell"].str.count(",").eq(len(SURVEY_TWO_LEVEL["person"])).all()  # full keys


def test_household_only_synthesis(survey_dir):
    weights_only = run_single(survey_dir, survey_variant(survey_dir, "weights", ["household"], ["person"],
                                                         {"household": SURVEY_HOUSEHOLD_CONTROLS}, False))
    with_persons = run_single(survey_dir, survey_variant(survey_dir, "synth", ["household"], ["person"],
                                                         {"household": SURVEY_HOUSEHOLD_CONTROLS}, True))
    without_persons = run_single(survey_dir, survey_variant(survey_dir, "synth_no_person", ["household"], [],
                                                            {"household": SURVEY_HOUSEHOLD_CONTROLS}, True))

    # drawing does not change the weights
    for out in (with_persons, without_persons):
        assert sha256(os.path.join(out, "weights.csv")) == sha256(os.path.join(weights_only, "weights.csv"))

    housing = pd.read_csv(os.path.join(with_persons, "housing_synthetic.csv"))
    sample = pd.read_csv(os.path.join(survey_dir, "survey_sample.csv"))
    totals = read_marginals(os.path.join(survey_dir, "marginals.csv"))[("hhtotals", "1")]
    assert housing.groupby("geo").size().to_dict() == {g: int(t) for g, t in totals.items()}
    assert housing["hid"].isin(sample["hid"]).all()

    # person table optional: written from the person sample when one is configured
    persons = pd.read_csv(os.path.join(with_persons, "person_synthetic.csv"))
    assert sorted(persons["hid"]) == sorted(housing["hid"])
    assert not os.path.exists(os.path.join(without_persons, "person_synthetic.csv"))
    assert sha256(os.path.join(without_persons, "housing_synthetic.csv")) == \
        sha256(os.path.join(with_persons, "housing_synthetic.csv"))


# --- the helper itself -------------------------------------------------------------
def _frame(keys, index, values):
    return pd.DataFrame(np.array(values, dtype=float), index=index, columns=pd.MultiIndex.from_tuples(keys))


def test_stack_cells_keeps_keys_of_different_lengths():
    household = _frame([("household", 1, 1), ("household", 2, 1)], [1, 2], [[1, 0], [0, 1]])
    person = _frame([("person", 1, 1, 1, 1), ("person", 1, 2, 1, 1), ("person", 2, 1, 1, 1)], [1, 3],
                    [[2, 0, 0], [0, 1, 1]])
    out = stack_cells([person, household], fill_and_sort_rows=True)
    assert out.columns.nlevels == 1
    assert list(out.columns) == sorted(list(household.columns) + list(person.columns))
    assert list(out.index) == [1, 2, 3] and not out.isna().any().any()
    assert out[("person", 1, 1, 1, 1)].loc[1] == 2 and out[("household", 1, 1)].loc[3] == 0


def test_stack_cells_same_as_pandas_when_pandas_kept_the_keys():
    # equal lengths: pd.concat kept the MultiIndex, and 3.0.5 results came from it
    a = _frame([("household", 2, 1), ("household", 1, 2)], [2, 1], [[1, 0], [0, 3]])
    b = _frame([("person", 1, 1), ("person", 2, 2)], [1, 3], [[1, 1], [2, 0]])
    ref = pd.concat([a, b], axis=1, join="outer").fillna(0).sort_index().sort_index(axis=1)
    out = stack_cells([a, b], fill_and_sort_rows=True)
    assert list(out.columns) == list(ref.columns)
    assert (out.to_numpy() == ref.to_numpy()).all() and list(out.index) == list(ref.index)
    # a single table keeps its order and only gets flat keys
    single = stack_cells([a])
    assert list(single.columns) == list(a.columns) and list(single.index) == [2, 1]
    assert list(flat_keys(a.columns)) == [("household", 2, 1), ("household", 1, 2)]
