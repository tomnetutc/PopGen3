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


def read_marginals(path):
    import pandas as pd
    return pd.read_csv(path, header=[0, 1], index_col=0)
