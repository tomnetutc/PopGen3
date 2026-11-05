"""Side-by-side comparison with the released popgen3 3.0.4 (opt-in).

Set POPGEN_DIFFERENTIAL=1 to run.  The 3.0.4 wheel is downloaded from PyPI and
both versions are run on the same inputs in the *same* environment, so every
output file (diagnostics included) must match byte for byte -- this check is
independent of the numpy/pandas/BLAS versions installed.
"""
import os
import subprocess
import sys
import zipfile

import pytest

from conftest import DATA, SURVEY, copy_inputs, output_dirs

import pandas as _pd

pytestmark = [
    pytest.mark.skipif(os.environ.get("POPGEN_DIFFERENTIAL") != "1",
                       reason="set POPGEN_DIFFERENTIAL=1 to compare against PyPI popgen3 3.0.4"),
    # 3.0.4 itself fails on pandas >= 3 (read-only arrays under copy-on-write), so it
    # cannot serve as the reference there; the golden tests cover pandas 3.
    pytest.mark.skipif(int(_pd.__version__.split(".")[0]) >= 3,
                       reason="popgen3 3.0.4 does not run on pandas >= 3"),
]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_FILES = {"reweighting_fit_summary.csv"}  # diagnostics added in 3.0.5

CASES = [  # (input dir, config) - inputs where 3.0.4 behaves as intended
    (DATA, "configuration_arizona.yaml"),
    (DATA, "configuration_arizona_two_levels.yaml"),
    (DATA, "configuration_arizona_single_level.yaml"),
    (SURVEY, "config_weighting.yaml"),
    (SURVEY, "config_wave1_style_geoids.yaml"),
]
if os.environ.get("POPGEN_DIFFERENTIAL_SLOW") == "1":
    CASES.append((DATA, "configuration_arizona_scenario2.yaml"))  # ~5 minutes


@pytest.fixture(scope="module")
def popgen_304(tmp_path_factory):
    """Extract the 3.0.4 wheel: from $POPGEN_304_WHEEL if set (offline), else pip download."""
    d = tmp_path_factory.mktemp("popgen304")
    wheel = os.environ.get("POPGEN_304_WHEEL")
    if not wheel:
        res = subprocess.run([sys.executable, "-m", "pip", "download", "popgen3==3.0.4", "--no-deps",
                              "--only-binary=:all:", "-d", str(d)], capture_output=True, text=True)
        if res.returncode != 0:
            pytest.skip("could not download popgen3==3.0.4 (set POPGEN_304_WHEEL to a local wheel): "
                        + res.stderr.strip()[-300:])
        wheel = str(d / next(p for p in os.listdir(d) if p.endswith(".whl")))
    pkg = d / "pkg"
    zipfile.ZipFile(wheel).extractall(pkg)
    return str(pkg)


def _run(pkg_dir, workdir, config):
    code = ("import sys, os; sys.path.insert(0, sys.argv[1]); import popgen; "
            "os.chdir(sys.argv[2]); popgen.run(sys.argv[3])")
    res = subprocess.run([sys.executable, "-I", "-c", code, pkg_dir, workdir, config],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-3000:]
    dirs = output_dirs(workdir)
    assert len(dirs) == 1, dirs
    return dirs[0]


@pytest.mark.parametrize("src,config", CASES, ids=[c for _, c in CASES])
def test_identical_to_pypi_304(popgen_304, tmp_path, src, config):
    old = _run(popgen_304, copy_inputs(src, tmp_path / "old"), config)
    new = _run(REPO, copy_inputs(src, tmp_path / "new"), config)
    old_files, new_files = set(os.listdir(old)), set(os.listdir(new))
    assert new_files - old_files <= NEW_FILES, new_files - old_files
    assert old_files <= new_files, old_files - new_files
    differ = [fn for fn in sorted(old_files)
              if open(os.path.join(old, fn), "rb").read() != open(os.path.join(new, fn), "rb").read()]
    assert not differ, f"{config}: files differ from 3.0.4: {differ}"
