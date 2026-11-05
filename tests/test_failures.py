"""A failed scenario must be reported, not just logged (changed in 3.0.5)."""
import os
import subprocess
import sys

import pytest

import popgen
from conftest import REPO, output_dirs, run_config


def _cli(workdir, *args):
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, "-m", "popgen.cli", *args], cwd=workdir, env=env,
                          capture_output=True, text=True)


def _break_marginals(survey_dir):
    cfg = os.path.join(survey_dir, "config_weighting.yaml")
    text = open(cfg).read().replace("household: marginals.csv", "household: no_such_file.csv")
    open(cfg, "w").write(text)


def test_failing_scenario_raises(survey_dir):
    _break_marginals(survey_dir)
    with pytest.raises(popgen.ScenarioError) as info:
        run_config(survey_dir, "config_weighting.yaml")
    assert info.value.failures and info.value.failures[0][0] == "SingleLevel"


def test_other_scenarios_still_run(survey_dir):
    cfg = os.path.join(survey_dir, "config_weighting.yaml")
    text = open(cfg).read()
    good = text.split("  scenario:\n", 1)[1]
    bad = good.replace("description: SingleLevel", "description: BadIds").replace("ids: [1]", "ids: [99]")
    open(cfg, "w").write(text.split("  scenario:\n", 1)[0] + "  scenario:\n" + bad + good)
    cwd = os.getcwd()
    os.chdir(survey_dir)
    try:
        with pytest.raises(popgen.ScenarioError) as info:
            popgen.run("config_weighting.yaml")
    finally:
        os.chdir(cwd)
    assert [d for d, _ in info.value.failures] == ["BadIds"]
    assert any(d.endswith(" SingleLevel") for d in output_dirs(survey_dir))  # the good one ran


def test_cli_exit_codes(survey_dir):
    ok = _cli(survey_dir, "config_weighting.yaml")
    assert ok.returncode == 0, ok.stderr[-2000:]
    _break_marginals(survey_dir)
    bad = _cli(survey_dir, "config_weighting.yaml")
    assert bad.returncode == 1
    assert "FAILED" in bad.stderr
    assert _cli(survey_dir, "missing.yaml").returncode == 1


def test_cli_version(survey_dir):
    res = _cli(survey_dir, "--version")
    assert res.returncode == 0 and popgen.__version__ in res.stdout
