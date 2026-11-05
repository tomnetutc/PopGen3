import os
import numpy as np
import pandas as pd
import logging
import time
from typing import Tuple
from scipy.optimize import fsolve
from popgen.config import Config

from popgen.output import Syn_Population


class Run_Reweighting:
    """
    Class to perform reweighting of sample weights using iterative proportional updating (IPU)
    or entropy-based optimization techniques.
    """
    DEFAULT_REWEIGHTING_PARAMS = {
        "procedure": "ipu",
        "tolerance": 0.0001,
        "inner_iterations": 1,
        "outer_iterations": 50,
        "archive_performance_frequency": 1
    }

    def __init__(self, entities, column_names_config, scenario_config, db, housing_entities=None):
        self.entities = entities
        self.column_names_config = column_names_config
        self.scenario_config = scenario_config
        self.db = db
        # Used to sweep housing-level constraints last (see _ipu_order).
        if isinstance(housing_entities, str):
            housing_entities = [housing_entities]
        self.housing_entities = list(housing_entities) if housing_entities else None

        if "reweighting" not in scenario_config._data["parameters"] or self.scenario_config._data["parameters"][
            "reweighting"] is None:
            logging.warning("Key 'reweighting' not found in configuration. Using default reweighting parameters.")
            self.scenario_config._data["parameters"]["reweighting"] = Run_Reweighting.DEFAULT_REWEIGHTING_PARAMS

        for key, default_value in self.DEFAULT_REWEIGHTING_PARAMS.items():
            if getattr(self.scenario_config.parameters.reweighting, key, None) in [None, ""]:
                setattr(self.scenario_config.parameters.reweighting, key, default_value)
                logging.warning(f"Setting default value for 'reweighting.{key}' as it was missing or empty.")

        self.outer_iterations = scenario_config.parameters.reweighting.outer_iterations
        self.inner_iterations = scenario_config.parameters.reweighting.inner_iterations
        self.archive_performance_frequency = scenario_config.parameters.reweighting.archive_performance_frequency
        self.procedure = scenario_config.parameters.reweighting.procedure

        # New in 3.0.5 (opt-in, default off): stop the outer IPU loop once the
        # mean relative deviation of every geography is below `tolerance`.
        # With the default the loop always runs `outer_iterations` times, as before.
        rw_cfg = scenario_config.parameters.reweighting
        raw = rw_cfg.get_raw if isinstance(rw_cfg, Config) else (lambda k, d=None: d)
        self.early_stop = bool(raw("early_stop", False))
        try:
            self.tolerance = float(raw("tolerance", self.DEFAULT_REWEIGHTING_PARAMS["tolerance"]))
        except (TypeError, ValueError):
            self.tolerance = self.DEFAULT_REWEIGHTING_PARAMS["tolerance"]
        self.fit_summary_rows = []

    def create_ds(self):
        """Creates datasets required for reweighting at region and geo levels."""
        self.region_stacked, self.region_row_idx, self.region_contrib = self._create_ds_for_resolution(
            self.scenario_config.control_variables.region)

        geo_controls_config = getattr(self.scenario_config.control_variables, "geo", None)

        if geo_controls_config is None:
            logging.info("Skipping geo dataset creation: No geo controls found.")
            self.geo_stacked, self.geo_row_idx, self.geo_contrib = None, None, None
        else:
            self.geo_stacked, self.geo_row_idx, self.geo_contrib = self._create_ds_for_resolution(geo_controls_config)

        self._create_sample_weights_df()
        self._create_reweighting_performance_df()

    def _create_ds_for_resolution(self, control_variables_config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Processes control variables configuration to create reweighting datasets."""
        sample_restruct_list = []
        hid_name = self.column_names_config.hid

        for entity in self.entities:
            variable_names = control_variables_config[entity].return_list()
            # Be robust to nested YAML lists (e.g. [[hinc, hsize]]).
            _flat_vars = []
            for _v in variable_names:
                if isinstance(_v, Config):
                    _flat_vars.extend(_v.return_list())
                elif isinstance(_v, (list, tuple)):
                    _flat_vars.extend(list(_v))
                else:
                    _flat_vars.append(_v)
            variable_names = _flat_vars
            sample = self.db.sample[entity]
            sample_restruct = Syn_Population.get_sample_restructure(entity, sample, variable_names, hid_name)
            sample_restruct_list.append(sample_restruct)

        stacked_sample = Syn_Population.get_stacked_sample_restruct(sample_restruct_list)
        row_idx, contrib = Syn_Population.get_row_idx(stacked_sample)
        return stacked_sample, row_idx, contrib

    def _create_sample_weights_df(self):
        """Initializes an empty DataFrame to store sample weights."""
        self.region_sample_weights = pd.DataFrame(index=self.region_stacked.index)

    def _create_reweighting_performance_df(self):
        """Initializes a DataFrame to track performance during reweighting iterations."""
        self.iters_to_archive = range(0, self.outer_iterations, self.archive_performance_frequency)
        self.average_diffs = pd.DataFrame(index=self.db.geo_ids, columns=self.iters_to_archive)

    def run_reweighting(self, region_constraints: pd.DataFrame, geo_constraints: pd.DataFrame):
        """Executes the reweighting process iteratively."""
        self.fit_summary_rows = []
        if geo_constraints is None:
            # Behaviour unchanged since 3.0.x: without geo-level controls the loop
            # below stops after its first pass.  Flag it so users can check the fit.
            logging.warning(
                "No geo-level constraints: reweighting performs a single IPU pass over the region "
                "constraints, which may not be converged. Check reweighting_fit_summary.csv."
            )
        for region_id in self.db.region_ids:
            logging.info(f"Running {self.procedure} for Region {region_id}")
            # If there are no geo-level constraints, we don't need a region->geo
            # mapping at all (region is the output geography). In that case we
            # treat the region id itself as the single geo id to avoid hard
            # failures when region_to_geo is intentionally absent.
            if geo_constraints is None:
                geo_ids = [region_id]
            else:
                geo_ids = self.db.get_geo_ids_for_region(region_id)
            geo_ids = [geo_ids] if isinstance(geo_ids, int) else geo_ids
            logging.info(f"Geos for region {region_id}: {geo_ids}")

            sample_weights = np.ones((self.region_stacked.shape[0], len(geo_ids)), dtype=float)
            iterations_run = 0

            for iter in range(self.outer_iterations):
                start_time = time.time()
                iterations_run = iter + 1

                if region_constraints is not None and region_id in region_constraints.index:
                    sample_weights = self._adjust_sample_weights(sample_weights, region_constraints.loc[region_id])

                if geo_constraints is not None:
                    for index, geo_id in enumerate(geo_ids):
                        if geo_id not in geo_constraints.index:
                            logging.warning(f"Geo ID {geo_id} not found in constraints; skipping geo reweighting.")
                            continue

                        geo_constraint_row = geo_constraints.loc[geo_id]
                        sample_weights[:, index] = self._adjust_sample_weights(
                            sample_weights[:, index],
                            geo_constraint_row,
                            self.inner_iterations,
                            geo=True,
                        )

                        if iter in self.iters_to_archive:
                            self._calculate_populate_average_deviation(
                                geo_id,
                                iter,
                                sample_weights[:, index],
                                geo_constraint_row,
                            )
                else:
                    logging.info("Skipping geo reweighting: No geo constraints found.")
                    break

                if self.early_stop:
                    worst = self._worst_mean_deviation(region_id, geo_ids, sample_weights,
                                                       region_constraints, geo_constraints)
                    if worst < self.tolerance:
                        logging.info(f"Early stop for region {region_id} after {iterations_run} iterations "
                                     f"(largest mean relative deviation {worst:.3g} < tolerance {self.tolerance:g}).")
                        break

            self._populate_sample_weights(sample_weights, region_id, geo_ids)
            logging.info(f"Sample weights sum: {sample_weights.sum()}")
            try:
                self._record_fit(region_id, geo_ids, sample_weights, region_constraints,
                                 geo_constraints, iterations_run)
            except Exception:  # diagnostics must never break a run
                logging.debug("Could not compute the reweighting fit summary.", exc_info=True)

    def _adjust_sample_weights(self, sample_weights: np.ndarray, constraints: pd.DataFrame,
                               iters: int = 1, geo: bool = False) -> np.ndarray:
        """Adjusts sample weights using the specified reweighting procedure."""
        if self.procedure == "ipu":
            return self._ipu_adjust_sample_weights(sample_weights, constraints, iters, geo)
        elif self.procedure == "entropy":
            return self._entropy_adjust_sample_weights(sample_weights, constraints, iters, geo)
        else:
            raise ValueError(f"Unknown reweighting procedure: {self.procedure}")

    def _ipu_adjust_sample_weights(self, sample_weights, constraints, iters=1, geo=False):
        """Adjust sample weights using iterative proportional updating (IPU).

        Backwards-compatibility note
        ----------------------------
        The original PopGen3 implementation (<= 3.0.3) iterated over the
        constraint cells in *reverse* order. With typical configurations
        (e.g. inner_iterations=1 and a limited number of outer iterations),
        that ordering materially affects convergence and the resulting weights.

        Newer refactors accidentally switched to forward iteration which can
        yield noticeably different weights (and downstream person marginals)
        for the same inputs. We restore the legacy reverse-iteration behaviour
        here for improved convergence and output consistency.
        """
        row_idx, contrib = (self.geo_row_idx, self.geo_contrib) if geo else (self.region_row_idx, self.region_contrib)
        sample_weights = np.array(sample_weights, order="C", dtype=float)
        order = self._ipu_order(constraints.index)

        for _ in range(iters):
            for column in order:
                if geo:
                    weighted_sum = float(sample_weights.dot(contrib[column]))
                else:
                    weighted_sum = float((sample_weights.T.dot(contrib[column])).sum())

                if weighted_sum == 0.0 or not np.isfinite(weighted_sum):
                    logging.warning(f"Weighted sum for {column} is zero/non-finite, skipping adjustment.")
                    continue

                adjustment = float(constraints[column]) / weighted_sum
                sample_weights[row_idx[column]] *= adjustment

        return sample_weights

    def _ipu_order(self, index):
        """Order in which IPU sweeps the constraint cells.

        The reverse of the sorted cell order (as in 3.0.3/3.0.4), except that the
        cells of housing entities are always swept after every person-level cell,
        so household / group-quarter totals are the ones matched exactly at the end
        of each sweep (the household-priority corner solution of Ye et al., 2009).

        With the usual entity names (household, groupquarter, person) this is
        exactly the 3.0.4 order.  3.0.4 obtained it only because "person" sorts
        after "household"; naming the person entity e.g. "adult" put it last and
        left household totals several percent off, as in 3.0.1/3.0.2.
        """
        rev = list(reversed(index))
        housing = getattr(self, "housing_entities", None)
        if not housing:
            return rev
        is_housing = [isinstance(c, tuple) and len(c) > 0 and c[0] in housing for c in rev]
        return [c for c, h in zip(rev, is_housing) if not h] + [c for c, h in zip(rev, is_housing) if h]

    def _entropy_adjust_sample_weights(self, sample_weights, constraints, iters=1, geo=False):
        row_idx = self.geo_row_idx if geo else self.region_row_idx
        contrib = self.geo_contrib if geo else self.region_contrib
        ones_array = np.ones((sample_weights.shape[1]), order="C") if not geo else None

        sample_weights = np.array(sample_weights, order="C")
        for _ in range(iters):
            for column in constraints.index:
                weights_mul_contrib = sample_weights * contrib[column] if geo else np.dot(sample_weights, ones_array) * \
                                                                                   contrib[column]
                root = self._find_root(contrib[column], constraints[column], weights_mul_contrib)
                adjustment = root ** contrib[column]
                sample_weights[row_idx[column]] = np.multiply(
                    sample_weights[row_idx[column]].T, adjustment[row_idx[column]]).T
        return sample_weights

    def _find_equation(self, contrib, weights_mul_contrib):
        root_power_weight = np.bincount(contrib, weights=weights_mul_contrib)
        root_power = np.array(range(contrib.max() + 1))
        return root_power[1:], root_power_weight[1:]

    def _optimizing_function(self, root, root_power, root_power_weight, constraint):
        return root_power_weight.dot(root ** root_power) - constraint

    def _find_root(self, contrib, constraint, weights_mul_contrib):
        root_power, root_power_weight = self._find_equation(contrib, weights_mul_contrib)
        if len(root_power) == 1:
            return constraint / root_power_weight
        return fsolve(self._optimizing_function, 0.0, args=(root_power, root_power_weight, constraint))

    def _calculate_populate_average_deviation(self, geo_id, iter, sample_weights, constraints):
        """Compute average percent deviation between current weighted totals and constraints."""
        eps = 1e-6
        diff_sum = 0.0
        count = 0
        for col in constraints.index:
            denom = float(constraints[col])
            if denom == 0.0:
                denom = eps
            modeled = float(sample_weights.dot(self.geo_contrib[col]))
            diff_sum += abs(modeled - float(constraints[col])) / denom
            count += 1
        self.average_diffs.loc[geo_id, iter] = diff_sum / (count or 1)

    # ------------------------------------------------------------------
    # Fit diagnostics (new in 3.0.5; read-only, never change the weights)
    #
    # IPU fits the multiway *cells* produced by IPF (e.g. income x size).
    # What users usually check are the *control totals* (one-way marginals),
    # so both are reported, per level, geography and entity.  At region level
    # the cells come from an IPF with the region seed, which need not agree
    # with the geo-level cells IPU finishes on; large region cell deviations
    # with small region marginal deviations are therefore normal.
    # ------------------------------------------------------------------
    @staticmethod
    def _rel_dev(modeled: dict, targets: dict):
        rel = {}
        for key, target in targets.items():
            t = float(target)
            if np.isfinite(t) and t > 0 and key in modeled:
                # Rounded so that summation-order noise (~1e-15, which differs between
                # BLAS builds/CPUs) neither changes the file nor picks the "worst" cell.
                rel[key] = round(abs(modeled[key] - t) / t, 12)
        if not rel:
            return 0, float("nan"), float("nan"), None
        worst = max(rel, key=rel.get)
        vals = np.fromiter(rel.values(), dtype=float)
        return len(vals), float(vals.mean()), float(vals.max()), worst

    def _geo_modeled(self, weights_col, constraints):
        return {col: float(weights_col.dot(self.geo_contrib[col])) for col in constraints.index if col in self.geo_contrib}

    def _region_modeled(self, sample_weights, constraints):
        return {col: float((sample_weights.T.dot(self.region_contrib[col])).sum())
                for col in constraints.index if col in self.region_contrib}

    def _control_names(self, level):
        """{entity: [control variable names]} for a level, read without config warnings."""
        from .validation import _flatten
        names = {}
        cv = self.scenario_config.get_raw("control_variables", None) \
            if isinstance(self.scenario_config, Config) else None
        if isinstance(cv, Config):
            cv = cv._data
        level_cfg = cv.get(level) if isinstance(cv, dict) else None
        if isinstance(level_cfg, Config):
            level_cfg = level_cfg._data
        if not isinstance(level_cfg, dict):
            return names
        for entity in self.entities:
            names[entity] = [str(v) for v in _flatten(level_cfg.get(entity))]
        return names

    @staticmethod
    def _to_margins(cells: dict, names: dict):
        """Collapse {(entity, v1, v2, ...): value} to {(entity, var, str(category)): total}."""
        out = {}
        for col, value in cells.items():
            if not isinstance(col, tuple) or not col:
                continue
            vars_ = names.get(col[0])
            if not vars_ or len(col) != len(vars_) + 1:
                continue
            for var, cat in zip(vars_, col[1:]):
                key = (col[0], var, str(cat))
                out[key] = out.get(key, 0.0) + float(value)
        return out

    def _input_margins(self, level, gid, entity, names):
        """Control totals exactly as given in the marginals file (None if unavailable)."""
        margs = self.db.geo_marginals if level == "geo" else self.db.region_marginals
        df = (margs or {}).get(entity)
        if df is None or not isinstance(df.columns, pd.MultiIndex) or gid not in df.index:
            return None
        row = df.loc[gid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        wanted = set(names.get(entity, []))
        out = {}
        for (var, cat), value in row.items():
            if str(var) in wanted:
                try:
                    out[(entity, str(var), str(cat))] = float(value)
                except (TypeError, ValueError):
                    pass
        return out or None

    def _fit_rows(self, level, region_id, gid, modeled, constraints, names, iterations_run):
        targets = {col: float(v) for col, v in constraints.items()}
        rows = []
        for entity in self.entities:
            m_cells = {k: v for k, v in modeled.items() if isinstance(k, tuple) and k and k[0] == entity}
            t_cells = {k: v for k, v in targets.items() if isinstance(k, tuple) and k and k[0] == entity}
            if not t_cells:
                continue
            nc, cmean, cmax, cworst = self._rel_dev(m_cells, t_cells)
            m_margins = self._to_margins(m_cells, names)
            t_margins = self._input_margins(level, gid, entity, names)
            if t_margins is None:
                # Fall back to the IPF-fitted totals; ignore the tiny values that the
                # zero-marginal correction puts into otherwise empty categories.
                t_margins = {k: v for k, v in self._to_margins(t_cells, names).items() if v >= 1e-3}
            m_margins = {k: m_margins.get(k, 0.0) for k in t_margins}  # no sample record -> 0
            nm, mmean, mmax, mworst = self._rel_dev(m_margins, t_margins)
            rows.append({
                "level": level, "region": region_id, "id": gid, "entity": entity, "iterations": iterations_run,
                "margins": nm, "margin_mean_abs_rel_dev": mmean, "margin_max_abs_rel_dev": mmax,
                "worst_margin": "" if mworst is None else f"{mworst[1]}={mworst[2]}",
                "cells": nc, "cell_mean_abs_rel_dev": cmean, "cell_max_abs_rel_dev": cmax,
                "worst_cell": "" if cworst is None else str(cworst),
            })
        return rows

    def _worst_mean_deviation(self, region_id, geo_ids, sample_weights, region_constraints, geo_constraints):
        """Early-stop criterion: largest geo-level mean cell deviation (the quantity
        archived in reweighting_average_diffs.csv)."""
        worst = 0.0
        if geo_constraints is None:
            return worst
        for index, geo_id in enumerate(geo_ids):
            if geo_id in geo_constraints.index:
                row = geo_constraints.loc[geo_id]
                _, mean, _, _ = self._rel_dev(self._geo_modeled(sample_weights[:, index], row),
                                              {k: v for k, v in row.items()})
                worst = max(worst, 0.0 if np.isnan(mean) else mean)
        return worst

    def _record_fit(self, region_id, geo_ids, sample_weights, region_constraints, geo_constraints, iterations_run):
        rows = []
        if region_constraints is not None and region_id in region_constraints.index:
            row = region_constraints.loc[region_id]
            rows += self._fit_rows("region", region_id, region_id, self._region_modeled(sample_weights, row),
                                   row, self._control_names("region"), iterations_run)
        if geo_constraints is not None:
            names = self._control_names("geo")
            for index, geo_id in enumerate(geo_ids):
                if geo_id in geo_constraints.index:
                    row = geo_constraints.loc[geo_id]
                    rows += self._fit_rows("geo", region_id, geo_id, self._geo_modeled(sample_weights[:, index], row),
                                           row, names, iterations_run)
        self.fit_summary_rows.extend(rows)
        parts = []
        for level in ("region", "geo"):
            for entity in self.entities:
                sel = [r for r in rows if r["level"] == level and r["entity"] == entity
                       and np.isfinite(r["margin_mean_abs_rel_dev"])]
                if sel:
                    parts.append(f"{level}/{entity} {np.mean([r['margin_mean_abs_rel_dev'] for r in sel]):.3%}"
                                 f" (max {np.max([r['margin_max_abs_rel_dev'] for r in sel]):.3%})")
        if parts:
            logging.info(f"Fit to control totals after {iterations_run} iteration(s), region {region_id}: "
                         + "; ".join(parts))

    def export_fit_summary(self, output_dir):
        """Write reweighting_fit_summary.csv: final fit of the weighted sample to the
        control totals (margins) and to the IPF multiway cells, per geography/entity."""
        if not self.fit_summary_rows:
            return None
        path = os.path.join(output_dir, "reweighting_fit_summary.csv")
        df = pd.DataFrame(self.fit_summary_rows)
        dev_cols = [c for c in df.columns if c.endswith("_abs_rel_dev")]
        df[dev_cols] = df[dev_cols].astype(float).round(10)
        df.to_csv(path, index=False)
        return path

    def _populate_sample_weights(self, sample_weights, region_id, geo_ids):
        for index, geo_id in enumerate(geo_ids):
            self.region_sample_weights[geo_id] = sample_weights[:, index]

    def _transform_column_index(self):
        """Transforms the column index of sample weights to a MultiIndex format."""
        self.region_sample_weights.columns = pd.MultiIndex.from_tuples(
            self.region_sample_weights.columns.values, names=["region_id", "geo_id"])