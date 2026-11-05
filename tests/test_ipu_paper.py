"""PopGen's IPU kernel reproduces the published worked example.

Ye, X., Konduri, K., Pendyala, R.M., Sana, B., Waddell, P. (2009).
A methodology to match distributions of both household and person attributes
in the generation of synthetic populations. 88th TRB Annual Meeting, Table 1:
8 households, 2 household types, 3 person types.
"""
import numpy as np
import pandas as pd
import pytest

from popgen.reweighting import Run_Reweighting

D = np.array([  # household type 1, type 2, person type 1, 2, 3
    [1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 2, 1, 0], [0, 1, 1, 0, 2],
    [0, 1, 0, 2, 1], [0, 1, 1, 1, 0], [0, 1, 2, 1, 2], [0, 1, 1, 1, 0]])
COLS = ["hh1", "hh2", "p1", "p2", "p3"]
C = pd.Series([35.0, 65.0, 91.0, 65.0, 104.0], index=COLS)

PAPER_WEIGHTS_AFTER_ONE_ITERATION = [12.37, 14.61, 8.05, 16.28, 16.91, 8.97, 13.78, 8.97]
PAPER_FINAL_WEIGHTS = [1.36, 25.66, 7.98, 27.79, 18.45, 8.64, 1.47, 8.64]


@pytest.fixture
def kernel():
    rw = Run_Reweighting.__new__(Run_Reweighting)  # only the IPU kernel is exercised
    rw.geo_contrib = {c: D[:, j].astype(int) for j, c in enumerate(COLS)}
    rw.geo_row_idx = {c: np.where(D[:, j] > 0)[0] for j, c in enumerate(COLS)}
    return rw


def delta(w):
    return float(np.mean([abs(w.dot(D[:, j]) - C[c]) / C[c] for j, c in enumerate(COLS)]))


def test_initial_goodness_of_fit():
    assert round(delta(np.ones(8)), 4) == 0.9127


def test_first_iteration_matches_table_1(kernel):
    # PopGen sweeps the constraints in reverse order; feeding them reversed
    # reproduces the paper's order (household type 1, 2, person type 1, 2, 3).
    w = kernel._ipu_adjust_sample_weights(np.ones(8), C[COLS[::-1]], 1, geo=True)
    assert np.round(w, 2).tolist() == PAPER_WEIGHTS_AFTER_ONE_ITERATION
    assert abs(delta(w) - 0.0954) < 2e-4


@pytest.mark.parametrize("order", ["paper", "popgen_default"])
def test_converges_to_published_final_weights(kernel, order):
    cons = C[COLS[::-1]] if order == "paper" else C
    w = np.ones(8)
    for _ in range(638):
        w = kernel._ipu_adjust_sample_weights(w, cons, 1, geo=True)
    assert np.round(w, 2).tolist() == PAPER_FINAL_WEIGHTS
    assert delta(w) < 2e-5  # paper: 8.51e-6 after 638 iterations
