"""The weights must not depend on what the entities are called.

3.0.3/3.0.4 restored the reverse IPU sweep, which ends each sweep on the
household and group-quarter cells -- but only because "person" sorts after
"household" and "groupquarter".  With the person entity named e.g. "adult",
3.0.4 swept the person cells last and household totals came out about 4 % off
(11 % max), the same failure as the 3.0.2 regression.  Since 3.0.5 the cells
of the housing entities are always swept last; with the usual names the order
(and therefore every output) is unchanged.
"""
import os
import re

import pandas as pd
import pytest

from conftest import DATA, GOLDEN, copy_inputs, run_single, sha256
from popgen.reweighting import Run_Reweighting


def _order(index, housing):
    rw = Run_Reweighting.__new__(Run_Reweighting)  # only the ordering helper is needed
    rw.housing_entities = housing
    return rw._ipu_order(pd.Index(index))


def test_usual_names_keep_the_304_order():
    # cells of different arity, as in the Arizona example (padded by pandas)
    index = pd.Index(sorted([("groupquarter", 1), ("household", 1, 1), ("household", 2, 1),
                             ("person", 1, 1), ("person", 2, 3)], key=str))
    got = _order(index, ["household", "groupquarter"])
    assert pd.Index(got).equals(index[::-1])  # NaN-aware comparison of the padded labels


def test_housing_cells_are_swept_last_whatever_their_names():
    index = sorted([("adult", 1), ("adult", 2), ("groupquarter", 1), ("household", 1), ("household", 2)])
    assert _order(index, ["household", "groupquarter"]) == [
        ("adult", 2), ("adult", 1), ("household", 2), ("household", 1), ("groupquarter", 1)]


def test_without_housing_information_the_order_is_the_reverse_order():
    index = sorted([("adult", 1), ("household", 1)])
    assert _order(index, None) == list(reversed(index))


def _rename_person_entity(workdir, new_name):
    """Write a copy of the Arizona config with the person entity called *new_name*
    (entity keys and lists only; file names such as person_sample.csv are kept)."""
    with open(os.path.join(workdir, "configuration_arizona.yaml")) as f:
        lines = f.read().splitlines(keepends=True)
    out, changed = [], 0
    for line in lines:
        s = line.strip()
        # Leave comments alone, and the 'person:' key under outputs.synthetic_population,
        # which names the entity *type* (housing/person), not an entity.
        if s.startswith("#") or s == "person:":
            out.append(line)
            continue
        new = re.sub(r"\bperson\b", new_name, line)  # \b skips person_sample.csv, person_entities
        changed += new != line
        out.append(new)
    assert changed == 8, f"expected to rename the person entity on 8 lines, renamed {changed}"
    name = f"configuration_arizona_{new_name}.yaml"
    with open(os.path.join(workdir, name), "w") as f:
        f.write("".join(out))
    return name


@pytest.mark.parametrize("new_name", ["adult"])
def test_renamed_person_entity_gives_identical_weights_and_population(new_name, tmp_path):
    workdir = copy_inputs(DATA, tmp_path / "data")
    out = run_single(workdir, _rename_person_entity(workdir, new_name))
    spec = GOLDEN["runs"]["arizona_main"]["files"]
    changed = []
    for fn, expected in spec.items():
        path = os.path.join(out, fn)
        if not os.path.exists(path):
            path = os.path.join(out, fn.replace("person", new_name))
        assert os.path.exists(path), f"missing output {fn}"
        # the entity name itself appears in a few files (e.g. the 'entity' column)
        if sha256(path, replace=[(new_name.encode(), b"person")]) != expected:
            changed.append(fn)
    assert not changed, (f"naming the person entity '{new_name}' changed {changed}; "
                         "the IPU sweep order must not depend on entity names")
