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

    def __init__(self, entities, column_names_config, scenario_config, db):
        self.entities = entities
        self.column_names_config = column_names_config
        self.scenario_config = scenario_config
        self.db = db

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

            for iter in range(self.outer_iterations):
                start_time = time.time()

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

            self._populate_sample_weights(sample_weights, region_id, geo_ids)
            logging.info(f"Sample weights sum: {sample_weights.sum()}")

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

        for _ in range(iters):
            for column in reversed(constraints.index):
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

    def _populate_sample_weights(self, sample_weights, region_id, geo_ids):
        for index, geo_id in enumerate(geo_ids):
            self.region_sample_weights[geo_id] = sample_weights[:, index]

    def _transform_column_index(self):
        """Transforms the column index of sample weights to a MultiIndex format."""
        self.region_sample_weights.columns = pd.MultiIndex.from_tuples(
            self.region_sample_weights.columns.values, names=["region_id", "geo_id"])