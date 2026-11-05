"""Generate the synthetic single-level survey fixture used by the tests.

It mimics how PopGen is used for survey weighting (e.g. the THA surveys):
one respondent per household, one geography level, a single marginals file
used for both the household and the person entity.  The data are random
(fixed seed) - no real survey records are involved.

Run from the repository root:  python tests/data/make_survey_fixture.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "survey_single")
rng = np.random.RandomState(20261104)

VARS = {  # name: (categories, sample probabilities, population shares)
    "age_group": ([1, 2, 3, 4], [0.15, 0.30, 0.35, 0.20], [0.22, 0.28, 0.27, 0.23]),
    "educ": ([1, 2, 3], [0.20, 0.30, 0.50], [0.38, 0.29, 0.33]),
    "hispanic": ([1, 2], [0.10, 0.90], [0.18, 0.82]),
    "hh_income": ([1, 2, 3, 4], [0.15, 0.25, 0.30, 0.30], [0.28, 0.27, 0.24, 0.21]),
    "hh_size": ([1, 2, 3, 4], [0.25, 0.40, 0.15, 0.20], [0.28, 0.34, 0.16, 0.22]),
}
GEOS = {1: (400, 10000.0), 2: (300, 6500.0)}  # geo: (sample size, households)

rows = []
hid = 100
for geo, (n, _) in GEOS.items():
    for _ in range(n):
        hid += 1
        r = {"hid": hid, "pid": 1}
        for var, (cats, p, _) in VARS.items():
            r[var] = int(rng.choice(cats, p=p))
        r.update(hhtotals=1, ptotals=1, sample_geo=geo)
        rows.append(r)
sample = pd.DataFrame(rows)

# Marginals in PopGen's layout: 2 header rows (variable_names / variable_categories),
# then a row naming the index ("geo"), then one row per geography.
cols = [("hhtotals", 1)] + [(v, c) for v, (cats, _, _) in VARS.items() for c in cats] + [("ptotals", 1)]
table = []
for geo, (_, total) in GEOS.items():
    vals = [total]
    for v, (cats, _, share) in VARS.items():
        alloc = np.round(np.array(share) * total, 3)
        alloc[-1] = round(total - alloc[:-1].sum(), 3)  # totals agree across variables
        vals += alloc.tolist()
    vals.append(total)
    table.append([geo] + vals)

with open(os.path.join(HERE, "marginals.csv"), "w") as f:
    f.write("variable_names," + ",".join(c[0] for c in cols) + "\n")
    f.write("variable_categories," + ",".join(str(c[1]) for c in cols) + "\n")
    f.write("geo" + "," * len(cols) + "\n")
    for row in table:
        f.write(",".join(str(x) for x in row) + "\n")

sample.to_csv(os.path.join(HERE, "survey_sample.csv"), index=False)
pd.DataFrame({"geo": list(GEOS), "sample_geo": list(GEOS)}).to_csv(
    os.path.join(HERE, "geo_sample_mapping.csv"), index=False)
print("wrote", HERE, len(sample), "records")
