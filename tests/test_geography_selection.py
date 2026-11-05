"""Which geographies a scenario runs for (geos_to_synthesize)."""
import os
import re

import pandas as pd
import pytest

import popgen
from conftest import run_config, run_single


def _weight_columns(out):
    return list(pd.read_csv(os.path.join(out, "weights.csv"), nrows=1).columns[1:])


def _edit(path, old, new):
    text = open(path).read()
    assert old in text, old
    open(path, "w").write(text.replace(old, new))


def test_ids_in_geo_block_single_level(survey_dir):
    # THA Wave 2 pattern
    assert _weight_columns(run_single(survey_dir, "config_weighting.yaml")) == ["1"]


def test_ids_in_region_block_single_level(survey_dir):
    # THA Wave 1 pattern: region ids + template `geo: {ids: [], all_ids: True}`.
    # 3.0.4 ignored the region ids and ran every geography (here: 1 and 2).
    assert _weight_columns(run_single(survey_dir, "config_wave1_style.yaml")) == ["1"]


def test_conflicting_ids_prefer_geo_block_and_warn(survey_dir, caplog):
    cfg = os.path.join(survey_dir, "config_wave1_style.yaml")
    _edit(cfg, "     geo:\n      ids: []", "     geo:\n      ids: [2]")
    out = run_single(survey_dir, "config_wave1_style.yaml")
    assert _weight_columns(out) == ["2"]
    assert "differ" in caplog.text


def test_no_ids_runs_every_geography(survey_dir):
    cfg = os.path.join(survey_dir, "config_weighting.yaml")
    _edit(cfg, "ids: [1]\n          all_ids: false", "ids: []\n          all_ids: true")
    assert _weight_columns(run_single(survey_dir, "config_weighting.yaml")) == ["1", "2"]


@pytest.mark.parametrize("old,new", [
    ("ids: [1]\n          all_ids: false", "ids: [99]\n          all_ids: false"),   # unknown id
    ("ids: [1]\n          all_ids: false", "ids: []\n          all_ids: false"),    # nothing selected
])
def test_invalid_selection_fails_instead_of_running_everything(survey_dir, old, new):
    # 3.0.4 caught these errors itself and silently ran ALL geographies.
    _edit(os.path.join(survey_dir, "config_weighting.yaml"), old, new)
    with pytest.raises(popgen.ScenarioError) as info:
        run_config(survey_dir, "config_weighting.yaml")
    assert isinstance(info.value.failures[0][1], popgen.ConfigurationError)
