"""Input checks (new in 3.0.5): report problems, never change results."""
import os

import numpy as np
import pandas as pd
import pytest

import popgen
from conftest import run_config, run_single, sha256

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
