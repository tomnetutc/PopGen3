"""Input checks (new in 3.0.5): report problems, never change results."""
import os

import numpy as np
import pandas as pd
import pytest

import popgen
from conftest import (SURVEY_HOUSEHOLD_CONTROLS, output_dirs, read_marginals, run_config, run_single, sha256,
                      survey_variant)

CFG = "config_weighting.yaml"


def _sample(survey_dir):
    return os.path.join(survey_dir, "survey_sample.csv")


def _codes(caplog):
    return {line.split("[", 1)[1].split("]", 1)[0] for line in caplog.text.splitlines() if "Input check" in line}


def test_clean_inputs_pass(survey_dir, caplog):
    import logging
    caplog.set_level(logging.INFO)
    run_single(survey_dir, CFG)
    assert "Input checks passed." in caplog.text


def test_duplicate_household_ids(survey_dir, caplog):
    df = pd.read_csv(_sample(survey_dir))
    df["hid"] = df["hid"] // 50          # many households now share an id (like Excel's 2.00607E+13)
    df.to_csv(_sample(survey_dir), index=False)
    with pytest.raises(popgen.ScenarioError) as info:
        run_config(survey_dir, CFG, strict=True)
    assert isinstance(info.value.failures[0][1], popgen.InputValidationError)
    assert "duplicate-household-id" in _codes(caplog)


def test_missing_control_values_reported(survey_dir, caplog):
    df = pd.read_csv(_sample(survey_dir))
    df.loc[:4, "educ"] = np.nan
    df.to_csv(_sample(survey_dir), index=False)
    with pytest.raises(popgen.ScenarioError):
        run_config(survey_dir, CFG)     # non-strict: reported, then IPF fails as before
    assert "control-missing-values" in _codes(caplog)


def test_unknown_sample_category(survey_dir, caplog):
    df = pd.read_csv(_sample(survey_dir))
    df.loc[:2, "educ"] = 9
    df.to_csv(_sample(survey_dir), index=False)
    with pytest.raises(popgen.ScenarioError):
        run_config(survey_dir, CFG)
    assert "category-not-in-marginals" in _codes(caplog)


def test_marginal_category_without_sample_records(survey_dir, caplog):
    df = pd.read_csv(_sample(survey_dir))
    df = df[~((df["age_group"] == 4) & (df["sample_geo"] == 1))]
    df.loc[df["sample_geo"] == 2, "age_group"] = df.loc[df["sample_geo"] == 2, "age_group"].clip(upper=3)
    df.to_csv(_sample(survey_dir), index=False)
    run_config(survey_dir, CFG)          # still runs; the warning explains the misfit
    assert "category-without-sample" in _codes(caplog)


def test_inconsistent_marginal_totals(survey_dir, caplog):
    path = os.path.join(survey_dir, "marginals.csv")
    lines = open(path).read().splitlines()
    head, cats = lines[0].split(","), lines[1].split(",")
    row = lines[3].split(",")
    j = [i for i, (v, c) in enumerate(zip(head, cats)) if v == "hh_size" and c == "4"][0]
    row[j] = str(float(row[j]) + 800.0)  # hh_size now sums to 8 % more than the other variables
    lines[3] = ",".join(row)
    open(path, "w").write("\n".join(lines) + "\n")
    run_config(survey_dir, CFG)
    assert "inconsistent-marginal-totals" in _codes(caplog)


def test_sample_geo_without_records(survey_dir, caplog):
    path = os.path.join(survey_dir, "geo_sample_mapping.csv")
    open(path, "w").write("geo,sample_geo\n1,7\n2,2\n")
    with pytest.raises(popgen.ScenarioError):
        run_config(survey_dir, CFG)
    assert "sample-geo-not-in-sample" in _codes(caplog)


def test_strict_flag_in_yaml(survey_dir):
    df = pd.read_csv(_sample(survey_dir))
    df["hid"] = df["hid"] // 50
    df.to_csv(_sample(survey_dir), index=False)
    cfg = os.path.join(survey_dir, CFG)
    open(cfg, "a").write("      validation:\n        strict: true\n")
    with pytest.raises(popgen.ScenarioError) as info:
        run_config(survey_dir, CFG)
    assert isinstance(info.value.failures[0][1], popgen.InputValidationError)


def test_checks_do_not_change_results(survey_dir, tmp_path):
    from conftest import GOLDEN
    out = run_single(survey_dir, CFG, strict=True)  # strict mode on clean inputs
    assert sha256(os.path.join(out, "weights.csv")) == GOLDEN["runs"]["survey_weighting"]["files"]["weights.csv"]


# --- new in 3.0.6 -------------------------------------------------------------------
def _set_targets(survey_dir, changes):
    """Overwrite targets of geo 1 in marginals.csv: {(variable, category): value}."""
    path = os.path.join(survey_dir, "marginals.csv")
    lines = open(path).read().splitlines()
    head, cats, row = lines[0].split(","), lines[1].split(","), lines[3].split(",")
    for (var, cat), value in changes.items():
        j = [i for i, (v, c) in enumerate(zip(head, cats)) if v == var and c == cat][0]
        row[j] = str(value)
    lines[3] = ",".join(row)
    open(path, "w").write("\n".join(lines) + "\n")


def test_negative_and_zero_targets(survey_dir, caplog):
    # A negative target (e.g. the difference of two sets of totals) and a zero target for a
    # category that has sample records: IPF replaces both by the zero-marginal
    # correction, so those records end up with almost no weight.  Totals unchanged.
    m = read_marginals(os.path.join(survey_dir, "marginals.csv")).loc[1]
    _set_targets(survey_dir, {("hispanic", "1"): -2.0, ("hispanic", "2"): m[("hispanic", "2")] + m[("hispanic", "1")] + 2.0,
                              ("educ", "1"): 0.0, ("educ", "2"): m[("educ", "2")] + m[("educ", "1")]})
    out = run_single(survey_dir, CFG)            # reported; the run itself is unchanged
    assert {"negative-target", "zero-target"} <= _codes(caplog)
    assert "negative-target" in caplog.text and "ERROR" in caplog.text
    w = pd.read_csv(os.path.join(out, "weights.csv"), index_col=0).iloc[:, 0]
    s = pd.read_csv(_sample(survey_dir)).set_index("hid")
    assert w[s.loc[w.index, "hispanic"] == 1].max() < 1e-3   # what the check warns about

    with pytest.raises(popgen.ScenarioError) as info:
        run_config(survey_dir, CFG, strict=True)
    assert isinstance(info.value.failures[0][1], popgen.InputValidationError)


def test_sample_column_named_like_geo_stops_synthesis_early(survey_dir, caplog):
    df = pd.read_csv(_sample(survey_dir))
    df["geo"] = df["sample_geo"]
    df.to_csv(_sample(survey_dir), index=False)
    cfg = survey_variant(survey_dir, "synth", ["household"], ["person"], {"household": SURVEY_HOUSEHOLD_CONTROLS}, True)
    with pytest.raises(popgen.ScenarioError) as info:
        run_config(survey_dir, cfg)                 # 3.0.5: failed at the very end ("columns overlap")
    assert isinstance(info.value.failures[0][1], popgen.InputValidationError)
    assert "sample-column-name-clash" in _codes(caplog)
    assert not output_dirs(survey_dir)              # stopped before IPF, nothing written
    run_single(survey_dir, CFG)                     # weights only: the column does no harm


def test_zero_targets_in_zones_sharing_a_sample_are_not_reported(arizona_dir):
    # TAZs that draw on one PUMA sample often have no 7-person households or group
    # quarters: an empty category there is normal, not an input problem.
    from popgen import validation
    from popgen.project import Project

    cwd = os.getcwd()
    os.chdir(arizona_dir)
    try:
        p = Project("configuration_arizona.yaml")
        p.load_project()
        sc = next(iter(p._config.project.scenario))
        p.db.enumerate_geo_ids_for_scenario(sc)
        inputs = p._config.project.inputs
        issues = validation.validate_scenario(p.db, sc, inputs.column_names, inputs.entities,
                                              inputs.housing_entities, inputs.person_entities, synthesize=True)
    finally:
        os.chdir(cwd)
    zero = read_marginals(os.path.join(arizona_dir, "Arizona", "household_marginals.csv"))[("hsize", "7")]
    assert (zero == 0).any()                         # the example does have empty TAZ categories
    assert not [i for i in issues if i.code == "zero-target" and i.message.startswith("geo ")]
    assert not [i for i in issues if i.level == "ERROR"]
