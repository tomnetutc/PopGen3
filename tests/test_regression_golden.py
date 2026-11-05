"""Outputs must stay byte-for-byte identical to popgen3 3.0.4.

The fingerprints in tests/golden/golden_304.json were produced by the PyPI
release 3.0.4.  A failure here means a code change altered the weights or the
synthetic population for an unchanged input -- exactly the kind of silent
change (3.0.1/3.0.2 vs 3.0.3) these tests exist to stop.
"""
import os

import pytest

from conftest import DATA, GOLDEN, SURVEY, copy_inputs, run_single, sha256


@pytest.mark.parametrize("name", sorted(GOLDEN["runs"]))
def test_outputs_identical_to_304(name, tmp_path):
    spec = GOLDEN["runs"][name]
    src = DATA if spec["config"].startswith("data/") else SURVEY
    workdir = copy_inputs(src, tmp_path / "work")
    out = run_single(workdir, os.path.basename(spec["config"]))

    missing = [fn for fn in spec["files"] if not os.path.exists(os.path.join(out, fn))]
    assert not missing, f"expected output files not written: {missing}"
    changed = [fn for fn, h in spec["files"].items() if sha256(os.path.join(out, fn)) != h]
    assert not changed, (
        f"{name}: outputs differ from popgen3 3.0.4: {changed}. If this change is intended, "
        f"document it in CHANGELOG.md and regenerate tests/golden/golden_304.json.")
