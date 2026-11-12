"""Side-by-side comparison with the released popgen3 3.0.4 and 3.0.5 (opt-in).

Set POPGEN_DIFFERENTIAL=1 to run.  Each release's wheel is downloaded from PyPI
(or taken from $POPGEN_304_WHEEL / $POPGEN_305_WHEEL) and run on the same inputs
in the *same* environment as the working tree, so every output file
(diagnostics included) must match byte for byte -- this check is independent
of the numpy/pandas/BLAS versions installed.
"""
import os
import subprocess
import sys
import zipfile

import pytest

from conftest import DATA, SURVEY, copy_inputs, output_dirs

import pandas as _pd

pytestmark = pytest.mark.skipif(os.environ.get("POPGEN_DIFFERENTIAL") != "1",
                                reason="set POPGEN_DIFFERENTIAL=1 to compare against PyPI releases")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASES = {  # version: output files the working tree writes and that release does not
    "3.0.4": {"reweighting_fit_summary.csv"},  # diagnostics added in 3.0.5
    "3.0.5": set(),
}

CASES = [  # (input dir, config) - inputs where 3.0.4 and 3.0.5 behave as intended
    (DATA, "configuration_arizona.yaml"),
    (DATA, "configuration_arizona_two_levels.yaml"),
    (DATA, "configuration_arizona_single_level.yaml"),
    (SURVEY, "config_weighting.yaml"),
    (SURVEY, "config_wave1_style_geoids.yaml"),
]
if os.environ.get("POPGEN_DIFFERENTIAL_SLOW") == "1":
    CASES.append((DATA, "configuration_arizona_scenario2.yaml"))  # ~5 minutes


@pytest.fixture(scope="module", params=sorted(RELEASES))
def release(request, tmp_path_factory):
    """(version, extracted wheel) from $POPGEN_<version>_WHEEL if set (offline), else pip download."""
    version = request.param
    if version == "3.0.4" and int(_pd.__version__.split(".")[0]) >= 3:
        # 3.0.4 itself fails on pandas >= 3 (read-only arrays under copy-on-write).
        pytest.skip("popgen3 3.0.4 does not run on pandas >= 3")
    d = tmp_path_factory.mktemp("popgen" + version.replace(".", ""))
    wheel = os.environ.get(f"POPGEN_{version.replace('.', '')}_WHEEL")
    if not wheel:
        res = subprocess.run([sys.executable, "-m", "pip", "download", f"popgen3=={version}", "--no-deps",
                              "--only-binary=:all:", "-d", str(d)], capture_output=True, text=True)
        if res.returncode != 0:
            pytest.skip(f"could not download popgen3=={version} (set POPGEN_{version.replace('.', '')}_WHEEL "
                        f"to a local wheel): " + res.stderr.strip()[-300:])
        wheel = str(d / next(p for p in os.listdir(d) if p.endswith(".whl")))
    pkg = d / "pkg"
    zipfile.ZipFile(wheel).extractall(pkg)
    return version, str(pkg)


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
def test_identical_to_pypi_release(release, tmp_path, src, config):
    version, pkg = release
    old = _run(pkg, copy_inputs(src, tmp_path / "old"), config)
    new = _run(REPO, copy_inputs(src, tmp_path / "new"), config)
    old_files, new_files = set(os.listdir(old)), set(os.listdir(new))
    assert new_files - old_files <= RELEASES[version], new_files - old_files
    assert old_files <= new_files, old_files - new_files
    differ = [fn for fn in sorted(old_files)
              if open(os.path.join(old, fn), "rb").read() != open(os.path.join(new, fn), "rb").read()]
    assert not differ, f"{config}: files differ from {version}: {differ}"
