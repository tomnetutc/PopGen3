"""Regenerate tests/golden/golden_306.json (fingerprints for tests/test_cell_keys.py).

The fingerprints come from popgen3 3.0.5 run with the entities listed person
first, ``entities: [person, household]``.  For that order pandas happened to
keep the cell keys of 3.0.5 intact, so 3.0.5 handled these configurations
correctly.  With the usual order, ``[household, person]``, 3.0.5 failed on the
two Arizona variants (synthesis: "Not allowed to merge between different
levels"; with two household controls IPU already crashed) and could not write
the synthetic population of the survey variant.  Since 3.0.6 the cell keys no
longer depend on pandas, and either order must give exactly these files.
weights.csv is stored as a numeric fingerprint (tests/conftest.py:
weights_fingerprint): its last decimal can differ between machines and pandas
versions, while the other files are platform independent.

usage (from the repository root):
    python tests/golden/make_golden_306.py <directory that contains the 3.0.5 'popgen' package>
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

FILES_ARIZONA = [  # hashed; weights.csv gets a numeric fingerprint instead
    "hhsize_income.csv", "housing_synthetic.csv", "person_synthetic.csv", "rpsex_rpage.csv",
    "summary_geo.csv", "summary_region.csv",
    "ipf_geo_iters_convergence_household.csv", "ipf_geo_iters_convergence_person.csv",
    "ipf_region_iters_convergence_household.csv", "ipf_region_iters_convergence_person.csv",
]
FILES_SURVEY = ["housing_synthetic.csv", "person_synthetic.csv"]


def main(package_dir):
    sys.path.insert(0, os.path.abspath(package_dir))
    sys.path.insert(1, os.path.join(REPO, "tests"))
    import popgen
    from conftest import (ARIZONA_NO_GQ, DATA, SURVEY, SURVEY_TWO_LEVEL, arizona_without_group_quarters,
                          copy_inputs, run_single, sha256, survey_variant, weights_fingerprint)

    def fingerprints(out, files):
        return {"files": {fn: sha256(os.path.join(out, fn)) for fn in files},
                "weights": weights_fingerprint(os.path.join(out, "weights.csv"))}

    assert popgen.__version__ == "3.0.5", popgen.__version__
    runs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in sorted(ARIZONA_NO_GQ):
            work = copy_inputs(DATA, os.path.join(tmp, name))
            out = run_single(work, arizona_without_group_quarters(work, name, entities=("person", "household")))
            runs[name] = fingerprints(out, FILES_ARIZONA)
        name = "survey_two_level_synthesis"
        work = copy_inputs(SURVEY, os.path.join(tmp, name))
        out = run_single(work, survey_variant(work, name, ["person", "household"], ["person"], SURVEY_TWO_LEVEL, True))
        runs[name] = fingerprints(out, FILES_SURVEY)

    import numpy, pandas, scipy
    meta = {
        "generated_with": (f"popgen3 3.0.5 with entities [person, household]; Python {sys.version.split()[0]}, "
                           f"numpy {numpy.__version__}, pandas {pandas.__version__}, scipy {scipy.__version__}"),
        "configs": "built by the helpers in tests/conftest.py (arizona_without_group_quarters, survey_variant)",
        "weights": "weights.csv as a numeric fingerprint (tests/conftest.py: weights_fingerprint), compared "
                   "with a relative tolerance of 1e-9: its last decimal can differ between machines and pandas versions",
        "excluded": "reweighting_fit_summary.csv (its rows follow the entity order), full-precision "
                    "diagnostics (*average_diffs*, draws.csv) and the scenario YAML copies",
    }
    with open(os.path.join(HERE, "golden_306.json"), "w") as f:
        json.dump({"meta": meta, "runs": runs}, f, indent=1)
        f.write("\n")
    print("wrote", os.path.join(HERE, "golden_306.json"), sorted(runs))


if __name__ == "__main__":
    main(sys.argv[1])
