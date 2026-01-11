import pandas as pd
import numpy as np
import logging
from scipy import stats


class Draw_Population:
    """Handles the process of drawing synthetic populations based on weighted sampling."""

    def __init__(self, scenario_config, geo_ids, geo_row_idx, geo_frequencies,
                 geo_constraints, geo_stacked, region_sample_weights):
        self.scenario_config = scenario_config
        self.geo_ids = geo_ids
        self.geo_row_idx = geo_row_idx
        self.geo_frequencies = geo_frequencies
        self.geo_constraints = geo_constraints
        self.geo_stacked = geo_stacked
        self.region_sample_weights = region_sample_weights

        # Draw parameters (robust defaults if not configured).
        draws_cfg = getattr(self.scenario_config.parameters, "draws", None)
        if draws_cfg is None:
            logging.warning("'parameters.draws' missing; using defaults.")
            self.iterations = 25
            self.seed = 0
            self.pvalue_tolerance = 0.9999
        else:
            self.iterations = getattr(draws_cfg, "iterations", 25) or 25
            self.seed = getattr(draws_cfg, "seed", 0) or 0
            self.pvalue_tolerance = getattr(draws_cfg, "pvalue_tolerance", 0.9999) or 0.9999

        self.geo_id_rows_syn_dict = {}
        self.performance_columns = ["p_value", "iterations", "chi_sq_stat"]
        self.draws_performance = pd.DataFrame(index=self.geo_ids, columns=self.performance_columns)

    def draw_population(self):
        """Executes the drawing of households for each geographical ID."""
        np.random.seed(self.seed)
        logging.info("Starting synthetic population drawing.")

        for geo_id in self.geo_ids:
            logging.info(f"Processing geo ID: {geo_id}")
            geo_sample_weights = self.region_sample_weights.loc[:, geo_id]
            geo_cumulative_weights = self._return_cumulative_probability(geo_sample_weights)
            geo_id_frequencies = self.geo_frequencies.loc[geo_id, :]
            geo_id_constraints = self.geo_constraints.loc[geo_id, :]

            if geo_id_frequencies.sum() == 0:
                logging.warning(f"Skipping geo_id {geo_id}: Zero frequency sum.")
                continue

            p_value_max, geo_id_rows_syn_max, iter_max, stat_max = -1, None, None, None

            for iter in range(self.iterations):
                geo_id_rows_syn = self._pick_households(geo_id_frequencies, geo_cumulative_weights)
                stat, p_value = self._measure_match(geo_id_rows_syn, geo_id_constraints)

                if p_value > self.pvalue_tolerance:
                    # logging.info(f"Converged at iteration {iter} for geo ID {geo_id}.")
                    p_value_max, geo_id_rows_syn_max, iter_max, stat_max = p_value, geo_id_rows_syn, iter, stat
                    break
                elif p_value > p_value_max:
                    p_value_max, geo_id_rows_syn_max, iter_max, stat_max = p_value, geo_id_rows_syn, iter, stat

            self.draws_performance.loc[geo_id, self.performance_columns] = (p_value_max, iter_max, stat_max)
            self.geo_id_rows_syn_dict[geo_id] = geo_id_rows_syn_max

    def _return_cumulative_probability(self, geo_sample_weights):
        """Computes cumulative probabilities for weighted sampling."""
        geo_cumulative_weights = {}

        for column in self.geo_frequencies.columns:
            rows = self.geo_row_idx[column]
            weights = geo_sample_weights.take(rows).astype(float)
            # Replace NaNs with 0 (can happen if weights weren't fully initialized).
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            total = float(weights.sum())

            if total <= 0.0 or not np.isfinite(total):
                # If all weights are zero for this cell, fall back to uniform sampling
                # over the eligible rows. This prevents NaNs in the cumulative cdf.
                n = int(len(weights))
                if n == 0:
                    geo_cumulative_weights[column] = np.array([], dtype=float)
                else:
                    geo_cumulative_weights[column] = (np.arange(1, n + 1, dtype=float) / float(n))
            else:
                geo_cumulative_weights[column] = (weights / total).cumsum()

        return geo_cumulative_weights

    def _pick_households(self, geo_id_frequencies, geo_cumulative_weights):
        """Selects households based on cumulative probabilities."""
        last = 0
        rand_numbers = np.random.random(int(geo_id_frequencies.sum()))
        list_rows_syn_subpop = []

        for column in self.geo_frequencies.columns:
            rows = self.geo_row_idx[column]
            column_frequency = int(geo_id_frequencies[column])
            if column_frequency <= 0:
                continue
            if rows.size == 0:
                continue
            if geo_cumulative_weights[column].size == 0:
                continue
            column_bins = np.searchsorted(geo_cumulative_weights[column], rand_numbers[last:last + column_frequency],
                                          side="right")
            last += column_frequency
            list_rows_syn_subpop.append(rows.take(column_bins))

        if not list_rows_syn_subpop:
            return np.array([], dtype=int)
        return np.sort(np.concatenate(list_rows_syn_subpop))

    def _measure_match(self, geo_id_rows_syn, geo_id_constraints):
        """Evaluates the match between synthetic and expected constraints using chi-square test."""
        geo_id_constraints.name = "constraint"
        geo_id_synthetic = self.geo_stacked.take(geo_id_rows_syn).sum()
        geo_id_synthetic = pd.DataFrame(geo_id_synthetic, columns=["synthetic_count"])
        geo_id_synthetic = geo_id_synthetic.join(geo_id_constraints, how="inner")

        observed, expected = geo_id_synthetic["synthetic_count"], geo_id_synthetic["constraint"]
        observed_sum, expected_sum = observed.sum(), expected.sum()

        if abs(observed_sum - expected_sum) / expected_sum > 1e-08:
            expected *= observed_sum / expected_sum

        return stats.chisquare(observed, expected)
