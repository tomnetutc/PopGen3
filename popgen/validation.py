"""Pre-flight checks of PopGen inputs (new in 3.0.5).

The checks only *read* the loaded data; they never modify it, so they cannot
change any result.  Every finding is logged:

* ``ERROR``   - the run will fail later, or records will silently receive no
                weight (e.g. missing values in a control variable);
* ``WARNING`` - the run works but cannot match the targets exactly
                (e.g. a marginal category with no sample records, or
                variables whose marginal totals disagree).

By default the run then continues exactly as in earlier releases.  With
``validation: {strict: true}`` (scenario or project level), ``popgen --strict``
or ``popgen.run(..., strict=True)``, any ERROR stops the scenario before IPF.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import Config
from .errors import InputValidationError

TOTALS_SPREAD_WARN = 0.01  # 1 % disagreement between variable totals of one entity


@dataclass
class Issue:
    level: str  # "ERROR" | "WARNING"
    code: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"[{self.level}] {self.code}: {self.message}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _flatten(values) -> List:
    out: List = []
    for v in values or []:
        if isinstance(v, Config):
            out.extend(_flatten(v.return_list()))
        elif isinstance(v, (list, tuple)):
            out.extend(_flatten(v))
        elif v is not None:
            out.append(v)
    return out


def _raw(cfg, key, default=None):
    if isinstance(cfg, Config):
        return cfg.get_raw(key, default)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default


def _controls(controls_cfg, level: str, entity: str) -> List[str]:
    level_cfg = _raw(controls_cfg, level)
    if isinstance(level_cfg, Config):
        level_cfg = level_cfg._data
    if not isinstance(level_cfg, dict):
        return []
    return [str(v) for v in _flatten(level_cfg.get(entity))]


def _column(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    if name in df.columns:
        col = df[name]
        return col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col
    if df.index.name == name:
        return pd.Series(df.index.values, index=df.index)
    return None


def _has_category(columns: pd.Index, var: str, cat) -> bool:
    """Mirror the lookup IPF performs: (var, cat) first, then (str(var), str(cat))."""
    try:
        if (var, cat) in columns:
            return True
    except TypeError:
        pass
    return (str(var), str(cat)) in columns


def _marginal_categories(marg: pd.DataFrame, var: str) -> List[str]:
    if not isinstance(marg.columns, pd.MultiIndex):
        return []
    return [str(c) for v, c in marg.columns if str(v) == var]


def _as_float_set(values: Iterable) -> set:
    out = set()
    for v in values:
        try:
            out.add(float(v))
        except (TypeError, ValueError):
            pass
    return out


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def validate_scenario(db, scenario_config, column_names_config, entities: Sequence[str],
                      housing_entities: Sequence[str], person_entities: Sequence[str]) -> List[Issue]:
    """Run all checks for one scenario and return the findings (does not log)."""
    issues: List[Issue] = []

    def add(level, code, msg):
        issues.append(Issue(level, code, msg))

    def guarded(fn):
        try:
            fn()
        except Exception as exc:  # a broken check must never break a run
            logging.debug("Input check %s could not run: %s", fn.__name__, exc, exc_info=True)

    hid = str(_raw(column_names_config, "hid", "hid"))
    pid = str(_raw(column_names_config, "pid", "pid"))
    sample_geo = str(_raw(column_names_config, "sample_geo", "sample_geo"))
    controls_cfg = _raw(scenario_config, "control_variables")
    entities = list(entities or [])
    housing_entities = [e for e in (housing_entities or []) if e in db.sample]
    levels = [
        ("region", db.region_marginals or {}, list(db.region_ids or []), "region_to_sample"),
        ("geo", db.geo_marginals or {}, list(db.geo_ids or []), "geo_to_sample"),
    ]

    def check_samples_present():
        for e in entities:
            if e not in db.sample:
                add("ERROR", "sample-missing", f"no sample table was loaded for entity '{e}'.")

    def check_household_ids():
        seen: Dict = {}
        for e in housing_entities:
            ids = _column(db.sample[e], hid)
            if ids is None:
                add("ERROR", "id-column-missing", f"{e} sample has no '{hid}' column.")
                continue
            n, u = len(ids), ids.nunique(dropna=False)
            if u < n:
                add("ERROR", "duplicate-household-id",
                    f"{e} sample has {n} rows but only {u} distinct '{hid}' values; rows sharing an id are "
                    f"merged into one household (check for ids mangled by Excel, e.g. 2.00607E+13).")
            seen[e] = set(ids.dropna().tolist())
        es = list(seen)
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                both = seen[es[i]] & seen[es[j]]
                if both:
                    add("WARNING", "household-id-shared",
                        f"{len(both)} '{hid}' values appear in both the {es[i]} and {es[j]} samples.")

    def check_person_ids():
        housing_ids = set()
        for e in housing_entities:
            ids = _column(db.sample[e], hid)
            if ids is not None:
                housing_ids |= set(ids.dropna().tolist())
        for e in person_entities or []:
            if e not in entities or e not in db.sample:
                continue  # person table not used for weighting
            df = db.sample[e]
            hids = _column(df, hid)
            if hids is None:
                add("ERROR", "id-column-missing", f"{e} sample has no '{hid}' column.")
                continue
            orphans = int((~hids.isin(housing_ids)).sum()) if housing_ids else 0
            if orphans:
                add("WARNING", "person-without-household",
                    f"{orphans} {e} records have a '{hid}' that is not in any housing sample; "
                    f"they will not receive weights or appear in the synthetic population.")
            pids = _column(df, pid)
            if pids is not None:
                dup = int(pd.DataFrame({"h": hids.values, "p": pids.values}).duplicated().sum())
                if dup:
                    add("WARNING", "duplicate-person-id", f"{dup} duplicated ('{hid}', '{pid}') pairs in the {e} sample.")

    def check_controls():
        for level, margs, ids, _ in levels:
            for e in entities:
                vars_ = _controls(controls_cfg, level, e)
                if not vars_ or e not in db.sample:
                    continue
                df = db.sample[e]
                marg = margs.get(e)
                if marg is None:
                    add("ERROR", "marginals-missing",
                        f"{level}-level controls are set for entity '{e}' but no {level} marginals were loaded.")
                for var in vars_:
                    col = _column(df, var)
                    if col is None:
                        add("ERROR", "control-column-missing", f"control variable '{var}' is not a column of the {e} sample.")
                        continue
                    n_na = int(col.isna().sum())
                    if n_na:
                        add("ERROR", "control-missing-values",
                            f"{n_na} {e} records have no value for control '{var}'; such records are dropped "
                            f"by IPF/reweighting and receive no weight.")
                    if marg is None:
                        continue
                    cats = np.unique(col.dropna().values)
                    unmatched = [c for c in cats if not _has_category(marg.columns, var, c)]
                    if unmatched:
                        add("ERROR", "category-not-in-marginals",
                            f"{level} {e} control '{var}': sample categories {[str(c) for c in unmatched][:10]} have no "
                            f"column in the marginals file (IPF will fail with a KeyError).")
                    sample_labels = {str(c) for c in cats}
                    sel = marg.loc[marg.index.intersection(ids)] if ids else marg
                    for mcat in _marginal_categories(marg, var):
                        if mcat in sample_labels:
                            continue
                        vals = pd.to_numeric(sel[(var, mcat)] if (var, mcat) in sel.columns else pd.Series(dtype=float),
                                             errors="coerce")
                        n_pos = int((vals > 0).sum())
                        if n_pos:
                            add("WARNING", "category-without-sample",
                                f"{level} {e} control '{var}' = {mcat}: target > 0 in {n_pos} {level}(s) but no sample "
                                f"record has this category, so it cannot be matched.")

    def check_totals():
        for level, margs, ids, _ in levels:
            for e in entities:
                vars_ = _controls(controls_cfg, level, e)
                marg = margs.get(e)
                if len(vars_) < 2 or marg is None or not isinstance(marg.columns, pd.MultiIndex):
                    continue
                sel = marg.loc[marg.index.intersection(ids)] if ids else marg
                if sel.empty:
                    continue
                totals = {}
                for var in vars_:
                    cols = [c for c in sel.columns if str(c[0]) == var]
                    if cols:
                        totals[var] = sel[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
                if len(totals) < 2:
                    continue
                t = pd.DataFrame(totals)
                hi = t.max(axis=1)
                spread = (hi - t.min(axis=1)) / hi.where(hi > 0)
                worst = spread.idxmax() if spread.notna().any() else None
                if worst is not None and spread[worst] > TOTALS_SPREAD_WARN:
                    detail = ", ".join(f"{k}={v:.1f}" for k, v in t.loc[worst].items())
                    add("WARNING", "inconsistent-marginal-totals",
                        f"{level} {e}: control totals disagree by up to {spread[worst]:.1%} "
                        f"({int((spread > TOTALS_SPREAD_WARN).sum())} {level}(s) above {TOTALS_SPREAD_WARN:.0%}; "
                        f"worst {level} {worst}: {detail}); IPF cannot match all of them.")

    def check_geographies():
        for level, margs, ids, map_key in levels:
            entities_here = [e for e in entities if _controls(controls_cfg, level, e)]
            if not entities_here or not ids:
                continue
            for e in entities_here:
                marg = margs.get(e)
                if marg is None:
                    continue
                missing = [i for i in ids if i not in marg.index]
                if missing:
                    add("ERROR", "id-not-in-marginals",
                        f"{len(missing)} selected {level} id(s) are not rows of the {e} {level} marginals: {missing[:10]}")
            mapping = db.geo.get(map_key) if isinstance(db.geo, dict) else None
            if not isinstance(mapping, pd.DataFrame) or mapping.empty:
                continue
            missing = [i for i in ids if i not in mapping.index]
            if missing:
                add("ERROR", "id-not-in-mapping", f"{len(missing)} selected {level} id(s) are missing from {map_key}: {missing[:10]}")
            if sample_geo not in mapping.columns:
                continue
            wanted = _as_float_set(mapping.loc[mapping.index.intersection(ids), sample_geo].values.ravel()) - {-1.0}
            for e in entities_here:
                if e not in db.sample:
                    continue
                col = _column(db.sample[e], sample_geo)
                if col is None:
                    add("ERROR", "sample-geo-column-missing", f"the {e} sample has no '{sample_geo}' column.")
                    continue
                absent = sorted(wanted - _as_float_set(col.dropna().unique()))
                if absent:
                    add("ERROR", "sample-geo-not-in-sample",
                        f"{map_key} points selected {level}s to {sample_geo} {[int(a) if float(a).is_integer() else a for a in absent][:10]}, "
                        f"which has no records in the {e} sample.")

    for check in (check_samples_present, check_household_ids, check_person_ids,
                  check_controls, check_totals, check_geographies):
        guarded(check)
    return issues


def resolve_strict(project_config, scenario_config, override=None) -> bool:
    """Strict mode: explicit override > scenario `validation.strict` > project `validation.strict`."""
    if override is not None:
        return bool(override)
    for cfg in (scenario_config, project_config):
        block = _raw(cfg, "validation")
        if isinstance(block, Config):
            block = block._data
        if isinstance(block, dict) and block.get("strict") is not None:
            return bool(block.get("strict"))
    return False


def report(issues: List[Issue], description: str, strict: bool) -> None:
    """Log findings; raise InputValidationError in strict mode if any ERROR."""
    if not issues:
        logging.info("Input checks passed.")
        return
    for it in issues:
        log = logging.error if it.level == "ERROR" else logging.warning
        log(f"Input check {it.level} [{it.code}] {it.message}")
    n_err = sum(it.level == "ERROR" for it in issues)
    n_warn = len(issues) - n_err
    if n_err and strict:
        raise InputValidationError(
            f"Scenario '{description}': {n_err} input error(s) found (strict validation). See the log above.")
    logging.info(f"Input checks: {n_err} error(s), {n_warn} warning(s)"
                 + ("; set `validation: {strict: true}` to stop on errors." if n_err else "."))
