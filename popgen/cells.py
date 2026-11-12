"""Cell keys shared by IPF, reweighting and drawing (new in 3.0.6).

A *cell* is one combination of control-variable categories of one entity.  Its
key is a plain tuple ``(entity, category of variable 1, category of variable 2, ...)``,
so keys of different entities have different lengths (one element per control
variable, plus the entity name).

Up to 3.0.5 these keys lived in pandas MultiIndexes whose number of levels
followed the number of control variables, and the per-entity tables were joined
with ``pd.concat``.  When the entities had different numbers of control
variables, the result depended on pandas' handling of MultiIndexes with
different numbers of levels: keys were truncated (and could collide, so IPU
failed), synthesis failed with "Not allowed to merge between different levels",
or - in the configurations that worked, such as the Arizona example with a
group-quarter entity - pandas fell back to a flat index of tuples.

The helpers below always build that flat index of tuples explicitly and sort the
cells exactly as before, so every configuration that worked gives the same
results, and the others no longer depend on pandas internals.
"""

from typing import List, Optional, Sequence

import pandas as pd


def flat_keys(index) -> pd.Index:
    """Return *index* (a MultiIndex of cell keys, or an Index of tuples) as a flat
    object Index of tuples.

    Iterating a MultiIndex yields its keys as tuples of Python scalars - the same
    objects the flat index of tuples held when pandas built it in 3.0.5.
    """
    keys = [k if isinstance(k, tuple) else (k,) for k in index]
    return pd.Index(keys, dtype=object, tupleize_cols=False)


def stack_cells(frames: Sequence[pd.DataFrame], fill_and_sort_rows: bool = False) -> Optional[pd.DataFrame]:
    """Put per-entity tables side by side; the columns of every table are cell keys.

    Rows are outer-joined; columns are the cell keys of all tables in sorted
    order - what ``pd.concat(axis=1, join="outer")`` followed by
    ``sort_index(axis=1)`` did whenever pandas kept the keys intact.  With
    *fill_and_sort_rows* (sample tables, one row per household) missing values
    become 0 and the rows are sorted, as the sample stacking always did.  A single
    table keeps its rows and column order (as before) and only gets flat keys.

    The per-entity tables themselves (e.g. ``Run_IPF.geo_constraints_dict``) keep
    their MultiIndex with one named level per control variable.
    """
    frames = list(frames)
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0].set_axis(flat_keys(frames[0].columns), axis=1)

    keys: List[tuple] = []
    positional = []
    for frame in frames:
        cols = list(flat_keys(frame.columns))
        positional.append(frame.set_axis(pd.RangeIndex(len(keys), len(keys) + len(cols)), axis=1))
        keys.extend(cols)
    if len(set(keys)) != len(keys):
        raise ValueError("The same cell key occurs in more than one entity; entity names must be unique.")

    stacked = pd.concat(positional, axis=1, join="outer")
    if fill_and_sort_rows:
        stacked = stacked.fillna(0)
        stacked.sort_index(inplace=True)
    order = sorted(range(len(keys)), key=keys.__getitem__)
    stacked = stacked.iloc[:, order]
    stacked.columns = flat_keys([keys[i] for i in order])
    return stacked
