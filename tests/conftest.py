"""Shared helpers for the PopGen test suite.

Every test runs PopGen on a fresh copy of its inputs in a temporary
directory, so the repository data are never modified.
"""
import glob
import hashlib
import json
import os
import re
import shutil

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
SURVEY = os.path.join(REPO, "tests", "data", "survey_single")
with open(os.path.join(REPO, "tests", "golden", "golden_304.json")) as _f:
    GOLDEN = json.load(_f)
OUTPUT_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2} ")


def _ignore_outputs(_, names):
    return [n for n in names if OUTPUT_DIR_RE.match(n) or n == ".ipynb_checkpoints"]


def copy_inputs(src, dst):
    shutil.copytree(src, dst, ignore=_ignore_outputs)
    return str(dst)


@pytest.fixture
def arizona_dir(tmp_path):
    return copy_inputs(DATA, tmp_path / "data")


@pytest.fixture
def survey_dir(tmp_path):
    return copy_inputs(SURVEY, tmp_path / "survey")


def output_dirs(workdir):
    found = []
    for d, subdirs, _ in os.walk(workdir):
        for s in subdirs:
            if OUTPUT_DIR_RE.match(s):
                found.append(os.path.join(d, s))
    return sorted(found)


def run_config(workdir, config, strict=None):
    """Run PopGen in *workdir* (configs use paths relative to it); return new output folders."""
    import popgen

    before = set(output_dirs(workdir))
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        popgen.run(config, strict=strict)
    finally:
        os.chdir(cwd)
    return [d for d in output_dirs(workdir) if d not in before]


def run_single(workdir, config, strict=None):
    dirs = run_config(workdir, config, strict=strict)
    assert len(dirs) == 1, dirs
    return dirs[0]


def run_first_scenario(workdir, config):
    """Run the first scenario in-process and return (project, scenario) so tests
    can inspect the IPF / reweighting / drawing objects."""
    from popgen import validation
    from popgen.config import Config
    from popgen.project import Project, Scenario

    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        p = Project(config)
        p.load_project()
        sc = next(iter(p._config.project.scenario))
        if "parameters" not in sc._data or sc._data["parameters"] is None:
            sc._data["parameters"] = Config.DEFAULT_PARAMETERS
        if not bool(sc.get_raw("apply_cross_level", True)):
            p.process_geo_region_mappings(sc)
        synth = sc.get_raw("synthesize", None)
        if synth is None:
            synth = p._config.project.get_raw("synthesize", True)
        inputs = p._config.project.inputs
        s = Scenario(p.location, inputs.entities, inputs.housing_entities, inputs.person_entities,
                     inputs.column_names, sc, p.db, bool(synth),
                     strict=validation.resolve_strict(p._config.project, sc))
        s.run_scenario()
    finally:
        os.chdir(cwd)
    return p, s


def sha256(path, replace=()):
    """Fingerprint of an output file, independent of the platform's line endings.

    pandas >= 1.5 writes os.linesep, so the same run gives CRLF files on Windows
    and LF files elsewhere; the fingerprints are taken on the LF form.
    *replace* is an optional list of (old, new) byte strings applied first.
    """
    with open(path, "rb") as f:
        data = f.read().replace(b"\r\n", b"\n")
    for old, new in replace:
        data = data.replace(old, new)
    return hashlib.sha256(data).hexdigest()


def weights_fingerprint(path):
    """Fingerprint of weights.csv that does not depend on its last printed digit.

    weights.csv has 10 decimals, and the last one can differ between machines
    and pandas versions (the order of floating-point additions differs), so a
    file hash is too strict for golden values that must hold on every platform.
    The fingerprint holds the household ids exactly and, per geography, the
    total weight, two sums with fixed pseudo-random coefficients per household
    and the sum of squared weights; compare with weights_fingerprint_mismatch().
    """
    import numpy as np
    import pandas as pd

    w = pd.read_csv(path, index_col=0).sort_index()
    rank = np.arange(len(w))
    coefs = [np.ones(len(w)), (rank % 101) / 101.0, (rank * 7919 % 1013) / 1013.0]
    sums = {}
    for geo in w.columns:
        x = w[geo].to_numpy(dtype=float)
        sums[str(geo)] = [float(np.dot(x, c)) for c in coefs] + [float(np.dot(x, x))]
    ids = "\n".join(str(i) for i in w.index).encode()
    return {"households": len(w), "ids_sha256": hashlib.sha256(ids).hexdigest(), "sums": sums}


def weights_fingerprint_mismatch(actual, expected, rtol=1e-9, atol=1e-8):
    """Differences between two weights fingerprints; empty if they agree.

    The tolerance admits differences in the last written decimal of the weights
    (1e-10 per weight).
    """
    import numpy as np

    problems = [f"{k}: {actual[k]} != {expected[k]}" for k in ("households", "ids_sha256") if actual[k] != expected[k]]
    if sorted(actual["sums"]) != sorted(expected["sums"]):
        problems.append(f"geographies: {sorted(actual['sums'])} != {sorted(expected['sums'])}")
    else:
        for geo, ref in sorted(expected["sums"].items()):
            if not np.allclose(actual["sums"][geo], ref, rtol=rtol, atol=atol):
                problems.append(f"geo {geo}: {actual['sums'][geo]} != {ref}")
    return problems


def read_marginals(path):
    import pandas as pd
    return pd.read_csv(path, header=[0, 1], index_col=0)


# ---------------------------------------------------------------------------
# Derived configurations (3.0.6): variants of the repository configurations,
# written next to the copied inputs so the repository data stay untouched.
# ---------------------------------------------------------------------------
def derived_config(workdir, base, name, edit):
    """Write <workdir>/<name>.yaml: the YAML file *base* (in workdir) changed by edit(cfg)."""
    import yaml

    with open(os.path.join(workdir, base)) as f:
        cfg = yaml.safe_load(f)
    edit(cfg)
    with open(os.path.join(workdir, name + ".yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return name + ".yaml"


# Arizona example without the group-quarter entity, so that the household and
# person cell keys have different lengths and nothing else levels them out.
# name: (household geo controls, person geo controls)
ARIZONA_NO_GQ = {
    "arizona_no_gq": (["hinc", "hsize", "hhtotals"], ["psex", "page", "empl", "ptotals"]),
    "arizona_no_gq_two_household_controls": (["hinc", "hhtotals"], ["psex", "page", "empl", "ptotals"]),
}
_ARIZONA_REGION = {"hinc": "rhinc", "hsize": "rhsize", "hhtotals": "hhrtotals",
                   "psex": "rpsex", "page": "rpage", "empl": "rempl", "ptotals": "rptotals"}


def arizona_without_group_quarters(workdir, name, entities=("household", "person")):
    household, person = ARIZONA_NO_GQ[name]

    def edit(cfg):
        inputs = cfg["project"]["inputs"]
        inputs["entities"] = list(entities)
        inputs["housing_entities"] = ["household"]
        inputs["person_entities"] = ["person"]
        location = inputs["location"]
        for block in (location["sample"], location["marginals"]["geo"], location["marginals"]["region"]):
            block.pop("groupquarter", None)
        scenario = cfg["project"]["scenario"][0]
        scenario["description"] = name
        scenario["control_variables"] = {
            "region": {"household": [_ARIZONA_REGION[v] for v in household],
                       "person": [_ARIZONA_REGION[v] for v in person]},
            "geo": {"household": list(household), "person": list(person)},
        }

    return derived_config(workdir, "configuration_arizona.yaml", name, edit)


# The synthetic survey with household and person controls taken from the same
# file (one respondent per household): 3 household and 4 person variables.
SURVEY_HOUSEHOLD_CONTROLS = ["age_group", "educ", "hispanic", "hh_income", "hh_size", "hhtotals"]
SURVEY_TWO_LEVEL = {"household": ["hh_income", "hh_size", "hhtotals"],
                    "person": ["age_group", "educ", "hispanic", "ptotals"]}


def survey_variant(workdir, name, entities, person_entities, controls, synthesize):
    def edit(cfg):
        project = cfg["project"]
        project["synthesize"] = synthesize
        project["inputs"]["entities"] = list(entities)
        project["inputs"]["person_entities"] = list(person_entities)
        scenario = project["scenario"][0]
        scenario["description"] = name
        scenario["control_variables"] = {"geo": {k: list(v) for k, v in controls.items()}}
        scenario["geos_to_synthesize"] = {"geo": {"ids": [1, 2], "all_ids": False}}

    return derived_config(workdir, "config_weighting.yaml", name, edit)
