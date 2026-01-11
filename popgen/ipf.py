import numpy as np
import pandas as pd
import logging
# from .config import Config
import time
from popgen.config import Config

# TODO: Move all DS processing in the Syn_Population Class
class IPF_DS(object):
    def __init__(self, sample, entity, variable_names,
                 variables_count, variables_cats,
                 sample_geo_name=None):
        self.sample = sample
        self.entity = entity
        self.variable_names = variable_names
        self.variables_count = variables_count
        self.variables_cats = variables_cats
        self.sample_geo_name = sample_geo_name

    def get_seed(self):
        groupby_columns = ["entity"] + self.variable_names
        if self.sample_geo_name is not None:
            groupby_columns = ([self.sample_geo_name] +
                               groupby_columns)
        self.sample["entity"] = self.entity
        seed = (self.sample.groupby(groupby_columns)
                .size().astype(float))
        seed.name = "frequency"
        seed = seed.reset_index()
        seed.set_index(keys=groupby_columns,
                       drop=False, inplace=True)
        return seed

    def get_row_idx(self, seed):
        row_idx = {}
        for index, var in enumerate(self.variable_names):
            for cat in self.variables_cats[var]:
                row_idx[(var, cat)] = seed[var].values == cat
        return row_idx

class IPF:
    """Iterative Proportional Fitting (IPF) implementation."""

    def __init__(self, seed_all, seed, idx, marginals, ipf_config,
                 variable_names, variables_cats, variables_cats_count):
        self.seed_all = seed_all
        self.seed = seed
        self.idx = idx
        self.marginals = marginals
        self.ipf_config = ipf_config
        self.variable_names = variable_names
        self.variables_cats = variables_cats
        self.variables_cats_count = variables_cats_count
        self.ipf_iters = self.ipf_config.iterations
        self.ipf_tolerance = self.ipf_config.tolerance
        self.zero_marginal_correction = self.ipf_config.zero_marginal_correction
        self.archive_performance_frequency = self.ipf_config.archive_performance_frequency
        self.average_diff_iters = []
        self.iter_convergence = None

    def run_ipf(self):
        """Executes the IPF process."""
        if not isinstance(self.archive_performance_frequency, int):
            logging.info("Setting default value for 'archive_performance_frequency' (1) as it was missing or invalid.")
            self.archive_performance_frequency = 1

        self.frequencies = self._correct_zero_cell_issue()
        for c_iter in range(self.ipf_iters):
            self._adjust_cell_frequencies()

            if (c_iter % self.archive_performance_frequency) == 0:
                if self._check_convergence():
                    # logging.info(f"Convergence achieved in {c_iter} iterations.")
                    self.iter_convergence = c_iter
                    break

    def _correct_zero_cell_issue(self):
        """Handles zero-cell issues by redistributing probabilities."""
        if self.seed.shape[0] != self.seed_all.shape[0]:
            self.seed_all["prob"] = (self.seed["frequency"] / self.seed["frequency"].sum())
            null_rows = self.seed_all["prob"].isnull()
            self.seed_all["prob_all"] = (self.seed_all["frequency"] / self.seed_all["frequency"].sum())
            self.seed_all.loc[null_rows, "prob"] = self.seed_all.loc[null_rows, "prob_all"]
            borrowed_sum = self.seed_all.loc[null_rows, "prob"].sum()
            adjustment = 1 - borrowed_sum
            self.seed_all.loc[~null_rows, "prob"] *= adjustment

            return self.seed_all["prob"].copy().values
        else:
            return self.seed["frequency"].copy().values

    def _adjust_cell_frequencies(self):
        """Adjusts cell frequencies based on marginal constraints."""
        for var in self.variable_names:
            for cat in self.variables_cats[var]:
                row_subset = self.idx[(var, cat)]
                try:
                    marginal = self.marginals.loc[(var, cat)]
                except KeyError:
                    marginal = self.marginals.loc[(str(var), str(cat))]

                # Guard against zeros/near-zeros to avoid exploding adjustments.
                marginal = max(float(marginal), float(self.zero_marginal_correction), 1e-6)
                current_sum = float(self.frequencies[row_subset].sum())
                if current_sum <= 0.0 or not np.isfinite(current_sum):
                    # This should be rare if _correct_zero_cell_issue() worked, but can still
                    # happen when the seed is extremely sparse.
                    self.frequencies[row_subset] = max(np.finfo(np.float64).tiny, 1e-6)
                    current_sum = float(self.frequencies[row_subset].sum())

                adjustment = marginal / current_sum
                self.frequencies[row_subset] *= adjustment

                # Ensure strictly positive cells to prevent later division-by-zero.
                self.frequencies[self.frequencies == 0] = max(np.finfo(np.float64).tiny, 1e-6)

    def _check_convergence(self):
        """Checks if the IPF process has converged."""
        average_diff = self._calculate_average_deviation()
        self.average_diff_iters.append(average_diff)
        # Use an absolute tolerance on the deviation itself (not just the delta between
        # successive deviations). Using only delta can stop early when the algorithm
        # stalls far from a good fit.
        if average_diff < self.ipf_tolerance:
            return True
        return False

    def _print_marginals(self):
        """Logs the original and adjusted marginals for verification."""
        for var in self.variable_names:
            for cat in self.variables_cats[var]:
                row_subset = self.idx[(var, cat)]
                adjusted_frequency = self.frequencies[row_subset].sum()
                try:
                    original_frequency = self.marginals.loc[(var, cat)]
                except KeyError:
                    original_frequency = self.marginals.loc[(str(var), str(cat))]
                logging.info(f"({var}, {str(cat)}): Original={original_frequency}, Adjusted={adjusted_frequency}")

    def _calculate_average_deviation(self):
        """Computes the average deviation to assess convergence."""
        diff_sum = 0.0

        # NOTE: At the end of an IPF iteration, the *last* variable that was adjusted
        # will match its marginals (nearly) exactly. Many IPF implementations therefore
        # track convergence on all-but-last variables.
        vars_to_check = self.variable_names[:-1] if len(self.variable_names) > 1 else self.variable_names
        cats_count = sum(len(self.variables_cats[v]) for v in vars_to_check) or 1

        for var in vars_to_check:
            for cat in self.variables_cats[var]:
                row_subset = self.idx[(var, cat)]
                adjusted_frequency = self.frequencies[row_subset].sum()
                try:
                    original_frequency = self.marginals.loc[(var, cat)]
                except KeyError:
                    original_frequency = self.marginals.loc[(str(var), str(cat))]

                original_frequency = max(float(original_frequency), float(self.zero_marginal_correction), 1e-6)
                diff_sum += abs(float(adjusted_frequency) - original_frequency) / original_frequency

        return diff_sum / cats_count

class Run_IPF:
    DEFAULT_IPF_PARAMETERS = {
        "tolerance": 0.0001,
        "iterations": 250,
        "zero_marginal_correction": 0.00001,
        "rounding_procedure": "bucket",
        "archive_performance_frequency": 1
    }

    def __init__(self, entities, housing_entities, column_names_config, scenario_config, db):
        self.entities = entities
        self.housing_entities = housing_entities
        self.column_names_config = column_names_config
        self.scenario_config = scenario_config
        self.db = db

        if "ipf" not in scenario_config._data["parameters"] or self.scenario_config._data["parameters"]["ipf"] is None:
            logging.warning("Key 'ipf' not found in configuration. Using default IPF parameters.")
            self.scenario_config._data["parameters"]["ipf"] = Run_IPF.DEFAULT_IPF_PARAMETERS

        self.ipf_config = self.scenario_config.parameters.ipf
        logging.debug(f"IPF Configuration: {self.ipf_config}")

        for key, default_value in self.DEFAULT_IPF_PARAMETERS.items():
            if getattr(self.ipf_config, key) in [None, ""]:
                setattr(self.ipf_config, key, default_value)
                logging.warning(f"Setting default value for '{key}' as it was missing or empty.")

        self.ipf_rounding = self.ipf_config.rounding_procedure
        self.sample_geo_name = self.column_names_config.sample_geo

    def run_ipf(self):
        region_marginals = self.db.region_marginals
        region_controls_config = self.scenario_config.control_variables.region
        region_ids = self.db.region_ids
        region_to_sample = self.db.geo["region_to_sample"]
        logging.info("Region level IPF:")

        (self.region_constraints,
         self.region_constraints_dict,
         self.region_iters_convergence_dict,
         self.region_average_diffs_dict) = self._run_ipf_for_resolution(
            region_marginals, region_controls_config, region_ids, region_to_sample)

        self.region_columns_dict = self._get_columns_constraints_dict(self.region_constraints_dict)

        geo_controls_config = getattr(self.scenario_config.control_variables, "geo", None)

        if geo_controls_config:
            logging.info("Geo level IPF:")
            geo_marginals = self.db.geo_marginals
            geo_ids = self.db.geo_ids
            geo_to_sample = self.db.geo["geo_to_sample"]

            (self.geo_constraints,
             self.geo_constraints_dict,
             self.geo_iters_convergence_dict,
             self.geo_average_diffs_dict) = self._run_ipf_for_resolution(
                geo_marginals, geo_controls_config, geo_ids, geo_to_sample)

            self.geo_columns_dict = self._get_columns_constraints_dict(self.geo_constraints_dict)

            if self.ipf_rounding == "bucket":
                self.geo_frequencies = self._get_frequencies_for_resolution(
                    geo_ids, self.geo_constraints_dict, "bucket")
        else:
            logging.info("Skipping geo IPF: No geo controls found.")
            self.geo_constraints = None
            self.geo_constraints_dict = None
            self.geo_iters_convergence_dict = None
            self.geo_average_diffs_dict = None
            self.geo_columns_dict = None
            self.geo_frequencies = None

    def _create_ds_for_resolution_entity(self, sample, entity, variable_names,
                                         variables_count, variables_cats,
                                         sample_geo_name):
        """Creates dataset for IPF resolution entity."""
        ipf_ds_geo = IPF_DS(sample, entity, variable_names,
                            variables_count, variables_cats,
                            sample_geo_name)
        seed_geo = ipf_ds_geo.get_seed()

        ipf_ds_all = IPF_DS(sample, entity, variable_names,
                            variables_count, variables_cats)
        seed_all = ipf_ds_all.get_seed()
        row_idx = ipf_ds_all.get_row_idx(seed_all)

        return seed_geo, seed_all, row_idx

    def _run_ipf_for_resolution(self, marginals_at_resolution,
                                control_variables_config,
                                geo_ids, geo_corr_to_sample):
        """Runs IPF for all resolution levels."""
        constraints_list = []
        constraints_dict = {}
        iters_convergence_dict = {}
        average_diffs_dict = {}

        for entity in self.entities:
            logging.info(f"Running IPF for {entity}")
            sample = self.db.sample[entity]
            marginals = marginals_at_resolution[entity]
            variable_names = control_variables_config[entity].return_list()
            # Be robust to nested YAML lists (e.g. [[hinc, hsize]]), which would
            # otherwise propagate Config(list) objects into pandas indexing.
            _flat_vars = []
            for _v in variable_names:
                if isinstance(_v, Config):
                    _flat_vars.extend(_v.return_list())
                elif isinstance(_v, (list, tuple)):
                    _flat_vars.extend(list(_v))
                else:
                    _flat_vars.append(_v)
            variable_names = _flat_vars

            if not variable_names:
                continue

            variables_cats = self.db.return_variables_cats(entity, variable_names)
            variables_count = len(variable_names)
            variables_cats_count = sum(len(cats) for cats in variables_cats.values())

            seed_geo, seed_all, row_idx = self._create_ds_for_resolution_entity(
                sample, entity, variable_names, variables_count, variables_cats, self.sample_geo_name)

            constraints, iters_convergence, average_diffs = self._run_ipf_all_geos(
                entity, seed_geo, seed_all, row_idx, marginals,
                variable_names, variables_count, variables_cats,
                variables_cats_count, geo_ids, geo_corr_to_sample)

            constraints_dict[entity] = constraints
            iters_convergence_dict[entity] = iters_convergence
            average_diffs_dict[entity] = average_diffs
            constraints_list.append(constraints)

        constraints_resolution = self._get_stacked_constraints(constraints_list)
        return constraints_resolution, constraints_dict, iters_convergence_dict, average_diffs_dict

    def _run_ipf_all_geos(self, entity, seed_geo, seed_all, row_idx, marginals,
                          variable_names, variables_count, variables_cats,
                          variables_cats_count, geo_ids, geo_corr_to_sample):
        """Runs IPF for all geographical units."""

        results_dict = {}
        iters_convergence_dict = {}
        average_diffs_dict = {}

        for geo_id in geo_ids:
            sample_geo_id = geo_corr_to_sample.loc[geo_id, self.sample_geo_name]

            # NOTE: `seed_geo` is indexed by [sample_geo, entity, var1, var2, ...].
            # For a given geo_id, the mapping may point to either:
            #   * a single sample_geo (scalar),
            #   * multiple sample_geos (Series), or
            #   * -1 meaning "use all samples".
            #
            # The older code used `sum(level=...)`, which was removed in recent
            # pandas versions. We now use groupby(level=...) instead.

            if isinstance(sample_geo_id, pd.Series):
                sample_geo_ids = sample_geo_id.dropna().tolist()
                if len(sample_geo_ids) == 0:
                    raise ValueError(f"No sample_geo ids found for geo_id={geo_id}")

                seed_subset = seed_geo.loc[sample_geo_ids]
                # Collapse the sample_geo level (level 0) so index matches seed_all.
                seed_for_geo_id = seed_subset.groupby(level=seed_subset.index.names[1:]).sum()

            else:
                # Scalar mapping.
                if pd.isna(sample_geo_id):
                    raise ValueError(f"sample_geo_id is NaN for geo_id={geo_id}")

                sample_geo_scalar = int(sample_geo_id)
                if sample_geo_scalar == -1:
                    seed_for_geo_id = seed_all.copy()
                else:
                    # Select a single sample_geo and drop the sample_geo level.
                    try:
                        seed_for_geo_id = seed_geo.xs(sample_geo_scalar, level=self.sample_geo_name, drop_level=True)
                    except KeyError as e:
                        raise KeyError(
                            f"sample_geo_id={sample_geo_scalar} for geo_id={geo_id} not found in seed index."
                        ) from e

            marginals_geo = marginals.loc[geo_id]
            ipf_obj_geo = IPF(seed_all, seed_for_geo_id, row_idx, marginals_geo,
                              self.ipf_config, variable_names, variables_cats, variables_cats_count)
            ipf_obj_geo.run_ipf()

            #
            results_dict[geo_id] = ipf_obj_geo.frequencies
            iters_convergence_dict[geo_id] = ipf_obj_geo.iter_convergence
            average_diffs_dict[geo_id] = ipf_obj_geo.average_diff_iters[-1]

            if (ipf_obj_geo.frequencies == 0).any():
                raise RuntimeError("IPF cell values of zero are returned. Needs troubleshooting.")

        #
        ipf_results = pd.DataFrame(results_dict, index=seed_all.index)
        ipf_iters_convergence = pd.DataFrame(iters_convergence_dict, index=["iterations"])
        ipf_average_diffs = pd.DataFrame(average_diffs_dict, index=["average_percent_deviation"])

        return ipf_results, ipf_iters_convergence.T, ipf_average_diffs.T

    def _get_stacked_constraints(self, constraints_list):
        """Stacks constraints across multiple entities."""
        if not constraints_list:
            return None
        elif len(constraints_list) == 1:
            return constraints_list[0].T

        stacked_constraints = constraints_list[0].T
        for constraint in constraints_list[1:]:
            try:
                stacked_constraints = pd.concat([stacked_constraints, constraint.T], axis=1, join='outer')
            except pd.errors.MergeError as e:
                logging.error(f"MergeError: {e}")
                raise
        stacked_constraints.sort_index(axis=1, inplace=True)
        return stacked_constraints

    def _get_columns_constraints_dict(self, constraints_dict):
        """Extracts column constraints as a dictionary."""
        return {entity: constraints.index.values.tolist() for entity, constraints in constraints_dict.items()}

    def _get_frequencies_for_resolution(self, geo_ids, constraints_dict, procedure="bucket"):
        """Applies frequency rounding for integerizing multiway frequency distributions."""
        frequencies_list = []

        for entity in self.housing_entities:
            logging.info(f"Rounding frequencies for Entity: {entity}")
            frequencies = constraints_dict[entity].copy()

            for geo_id in geo_ids:
                frequency_geo = frequencies.loc[:, geo_id].values
                adjusted_frequency_geo = []
                accumulated_difference = 0

                for frequency in frequency_geo:
                    frequency_int = np.floor(frequency)
                    frequency_dec = frequency - frequency_int
                    accumulated_difference += frequency_dec
                    adjustment = accumulated_difference.round()
                    adjusted_frequency_geo.append(frequency_int + adjustment)
                    accumulated_difference -= adjustment

                frequencies.loc[:, geo_id] = adjusted_frequency_geo

            frequencies_list.append(frequencies)

        return self._get_stacked_constraints(frequencies_list)
