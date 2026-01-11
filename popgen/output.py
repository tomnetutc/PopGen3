import os
import time
import pandas as pd
import numpy as np
import logging
from scipy import stats

from collections.abc import Mapping

from .config import Config, wrap_config_value


class Syn_Population:
    """Handles synthetic population generation and output."""

    def __init__(self, location, db, column_names_config, scenario_config, run_ipf_obj, run_ipu_obj,
                 draw_population_obj, entities, housing_entities, person_entities):
        self.location = location
        self.db = db
        self.column_names_config = column_names_config
        self.scenario_config = scenario_config

        # Cache outputs configuration once. The Config wrapper logs warnings for
        # missing keys; doing this once prevents repeated warnings when users
        # intentionally omit optional `outputs:` sections.
        self.outputs_config = self._safe_get_outputs_config()

        self.run_ipf_obj = run_ipf_obj
        self.geo_constraints = run_ipf_obj.geo_constraints
        self.geo_frequencies = run_ipf_obj.geo_frequencies
        self.region_constraints = run_ipf_obj.region_constraints

        self.run_ipu_obj = run_ipu_obj
        self.geo_row_idx = run_ipu_obj.geo_row_idx
        self.geo_stacked = run_ipu_obj.geo_stacked
        self.region_sample_weights = run_ipu_obj.region_sample_weights

        self.draw_population_obj = draw_population_obj
        self.entities = entities
        self.housing_entities = housing_entities
        self.person_entities = person_entities

        self.geo_name = self.column_names_config.geo
        self.region_name = self.column_names_config.region
        self.hid_name = self.column_names_config.hid
        self.pid_name = self.column_names_config.pid
        self.unique_id_in_geo_name = "unique_id_in_geo"

        self.pop_syn = None
        self.pop_syn_data = {}

        self.pop_syn_geo_id_columns = [self.geo_name, self.unique_id_in_geo_name]
        self.pop_syn_all_id_columns = [self.geo_name, self.hid_name, self.unique_id_in_geo_name]
        self.pop_syn_housing_matching_id_columns = [self.geo_name, self.hid_name]
        self.pop_syn_person_matching_id_columns = [self.geo_name, self.hid_name, self.pid_name]

        self.pop_rows_syn_dict = {}
        self.housing_syn_dict = {}
        self.person_syn_dict = {}
        self.controls = {}
        self.geo_controls = {}
        self.region_controls = {}

        self._create_preliminaries()
        logging.debug(f"Scenario config type: {type(scenario_config)}")

    def _safe_get_outputs_config(self):
        """Return scenario `outputs` config if present, otherwise None.

        We prefer direct access to the underlying `_data` dict to avoid
        repeatedly triggering Config.return_value() warnings for missing keys.
        """
        try:
            data = getattr(self.scenario_config, "_data", None)
            if isinstance(data, Mapping) and ("outputs" in data) and (data["outputs"] is not None):
                return wrap_config_value(data["outputs"])
            return None
        except Exception:
            # Fallback to getattr (may warn, but keeps behaviour compatible).
            return getattr(self.scenario_config, "outputs", None)

    @staticmethod
    def _safe_get_child_config(parent_cfg, key: str):
        """Safely fetch a child config without triggering warnings for missing keys."""
        if parent_cfg is None:
            return None
        try:
            data = getattr(parent_cfg, "_data", None)
            if isinstance(data, Mapping):
                if key in data and data[key] is not None:
                    return wrap_config_value(data[key])
                # If key is absent OR explicitly null, treat as missing.
                return None
        except Exception:
            pass
        # Last resort: normal attribute access (may warn via Config.return_value).
        return getattr(parent_cfg, key, None)

    def _create_preliminaries(self):
        """Initialize preliminary data and configurations."""
        self._create_ds()
        self._create_meta_data()
        self._prepare_output_directory()

    def _create_ds(self):
        """Create stacked samples for housing and person entities."""
        self.housing_stacked_sample = self._get_stacked_sample(self.housing_entities)
        # Use the configured household id column name (do NOT hard-code 'hid').
        self.housing_stacked_sample.set_index(self.hid_name, inplace=True, drop=True)

        self.person_stacked_sample = self._get_stacked_sample(self.person_entities)
        self.person_stacked_sample.set_index(self.hid_name, inplace=True, drop=True)

    def _create_meta_data(self):
        """Create metadata for controls and entity types."""
        region_controls_config = self.scenario_config.control_variables.region
        geo_controls_config = self.scenario_config.control_variables.geo

        controls_config_list = [geo_controls_config, region_controls_config]
        for entity in self.entities:
            self.controls[entity] = self._return_controls_for_entity(controls_config_list, entity)

        controls_config_list = [geo_controls_config]
        for entity in self.entities:
            self.geo_controls[entity] = self._return_controls_for_entity(controls_config_list, entity)

        controls_config_list = [region_controls_config]
        for entity in self.entities:
            self.region_controls[entity] = self._return_controls_for_entity(controls_config_list, entity)

        self.entity_types_dict = {entity: "housing" for entity in self.housing_entities}
        self.entity_types_dict.update({entity: "person" for entity in self.person_entities})

        self.entity_types = ["housing", "person"]

    def _create_prepare_output_directory(self):
        """Prepare the output directory for storing results."""
        current_time_str = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        foldername = f"{current_time_str} {self.scenario_config.description}"
        self.outputlocation = os.path.join(self.location, foldername)
        if not os.path.exists(self.outputlocation):
            os.makedirs(self.outputlocation)

        self.filetype_sep_dict = {"csv": ","}

    def _return_controls_for_entity(self, controls_config_list, entity):
        """Return control variables for a given entity."""
        controls = []
        for controls_config in controls_config_list:
            controls += controls_config[entity].return_list()
        return controls

    def _prepare_output_directory(self):
        """Prepare the output directory for storing results."""
        current_time_str = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        foldername = f"{current_time_str} {self.scenario_config.description}"
        self.outputlocation = os.path.join(self.location, foldername)
        os.makedirs(self.outputlocation, exist_ok=True)
        self.filetype_sep_dict = {"csv": ","}

    def _get_stacked_sample(self, entities):
        """Get stacked sample data for the given entities."""
        sample_list = [self.db.sample[entity] for entity in entities]
        stacked_sample = pd.concat(sample_list).fillna(0)
        stacked_sample.sort_index(inplace=True)
        return stacked_sample

    def add_records(self):
        """Add records for synthetic population."""
        for geo_id, geo_id_rows_syn in self.draw_population_obj.geo_id_rows_syn_dict.items():
            geo_id_pop_syn = self._get_stacked_geo_for_geo_id(geo_id, geo_id_rows_syn)
            self.pop_rows_syn_dict[geo_id] = geo_id_pop_syn
            self.pop_rows_syn_dict[geo_id][self.unique_id_in_geo_name] = range(1, geo_id_rows_syn.shape[0] + 1)
        logging.info("Records added to synthetic population.")

    def _get_stacked_geo_for_geo_id(self, geo_id, geo_id_rows_syn):
        """Get stacked geo data for a specific geo ID."""
        geo_id_pop_syn = self.geo_stacked.take(geo_id_rows_syn).copy()
        geo_id_pop_syn[self.geo_name] = geo_id
        return geo_id_pop_syn

    def prepare_data(self):
        """Prepare data for synthetic population."""
        self._stack_records()
        self._create_synthetic_population()
        self._create_index()
        logging.info("Data preparation completed.")

    def _stack_records(self):
        """Stack records for the synthetic population."""
        start_time = time.time()
        self.pop_syn = pd.concat(self.pop_rows_syn_dict.values(), copy=False)
        logging.info(f"Time elapsed for stacking population: {time.time() - start_time:.4f} seconds")

    def _create_synthetic_population(self):
        """Create synthetic population data."""
        start_time = time.time()
        self.pop_syn_data["housing"] = self.pop_syn.loc[:, self.pop_syn_geo_id_columns].join(
            self.housing_stacked_sample)
        self.pop_syn_data["person"] = self.pop_syn.loc[:, self.pop_syn_geo_id_columns].join(self.person_stacked_sample)
        logging.info(f"Housing population size: {self.pop_syn_data['housing'].shape}")
        logging.info(f"Person population size: {self.pop_syn_data['person'].shape}")

    def _create_index(self):
        """Create indexes for synthetic population data."""
        for entity_type in ["housing", "person"]:
            self.pop_syn_data[entity_type].reset_index(inplace=True)
            self.pop_syn_data[entity_type].set_index(
                self.pop_syn_housing_matching_id_columns if entity_type == "housing" else self.pop_syn_person_matching_id_columns,
                inplace=True,
                drop=False
            )
            self.pop_syn_data[entity_type].sort_index(inplace=True)

    def export_outputs(self):
        """Export all output data."""
        start_time = time.time()

        # Users often omit the optional `outputs:` section to keep YAML files
        # minimal (especially when prototyping). Historically, PopGen would
        # still write the key artifacts (weights + synthetic population) and
        # skip optional reports. We preserve that behaviour here.
        if self.outputs_config is None:
            logging.info("No 'outputs' section configured. Using default output settings (weights + synthetic population).")
            self._export_weights()
            self._export_synthetic_population()
            self._pretty_print_scenario_configuration_file_to_output()
            logging.info(f"Time elapsed for generating outputs: {time.time() - start_time:.4f} seconds")
            return

        self._export_multiway_tables()
        self._export_summary()
        self._export_performance_data()
        self._export_weights()
        self._export_synthetic_population()
        self._pretty_print_scenario_configuration_file_to_output()
        logging.info(f"Time elapsed for generating outputs: {time.time() - start_time:.4f} seconds")

    def _pretty_print_scenario_configuration_file_to_output(self):
        """Pretty print the scenario configuration file to the output location."""
        filepath = os.path.join(self.outputlocation, f"{self.scenario_config.description}.yaml")

        if self.scenario_config is None:
            raise ValueError("ERROR: `self.scenario_config` is None before calling `write_to_file`!")

        if not callable(getattr(self.scenario_config, "write_to_file", None)):
            raise TypeError(f"ERROR: `write_to_file` is not callable! It is: {self.scenario_config.write_to_file}")

        self.scenario_config.write_to_file(filepath)
        # logging.info(f"Scenario configuration written to {filepath}")

    def _export_performance_data(self):
        """Export performance data."""
        outputs = self.outputs_config
        if outputs is None:
            logging.info("Skipping performance export: no 'outputs' section configured.")
            return

        values_to_export = self._safe_get_child_config(outputs, "performance")
        if values_to_export is None:
            logging.info("Skipping performance export: 'outputs.performance' not configured.")
            return

        # Normalize to a python list.
        if isinstance(values_to_export, Config):
            values_to_export = values_to_export.return_list()
        elif isinstance(values_to_export, str):
            values_to_export = [values_to_export]

        if "ipf" in values_to_export:
            self._export_all_df_in_dict(self.run_ipf_obj.geo_iters_convergence_dict, "ipf_geo_iters_convergence_")
            self._export_all_df_in_dict(self.run_ipf_obj.geo_average_diffs_dict, "ipf_geo_average_diffs_")
            self._export_all_df_in_dict(self.run_ipf_obj.region_iters_convergence_dict, "ipf_region_iters_convergence_")
            self._export_all_df_in_dict(self.run_ipf_obj.region_average_diffs_dict, "ipf_region_average_diffs_")
        if "reweighting" in values_to_export:
            self._export_df(self.run_ipu_obj.average_diffs, "reweighting_average_diffs")
        if "drawing" in values_to_export:
            self._export_df(self.draw_population_obj.draws_performance, "draws")
        logging.info("Performance data export completed.")

    def _export_weights(self):
        """Export weights data."""
        # Defaults: always export weights unless explicitly disabled.
        export = True
        collate = False

        outputs = self.outputs_config
        if outputs is not None:
            weights_cfg = self._safe_get_child_config(outputs, "weights")
            if weights_cfg is not None:
                export = bool(getattr(weights_cfg, "export", True))
                collate = bool(getattr(weights_cfg, "collate_across_geos", False))

        if not export:
            logging.info("Skipping weights export (disabled in configuration).")
            return

        df = self.run_ipu_obj.region_sample_weights
        # Ensure DataFrame for consistent CSV output.
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        if collate:
            df = df.sum(axis=1).to_frame(name="weight")

        filepath = os.path.join(self.outputlocation, "weights.csv")
        df.to_csv(filepath, float_format='%.10f', index=True, index_label=self.hid_name)
        logging.info(f"Exported weights to {filepath}")

    def _export_df(self, df, filename):
        """Export a DataFrame to a CSV file."""
        filepath = os.path.join(self.outputlocation, f"{filename}.csv")
        df.to_csv(filepath)
        # logging.info(f"Exported {filename} to {filepath}")

    def _export_all_df_in_dict(self, dict_of_dfs, fileprefix):
        """Export all DataFrames in a dictionary to CSV files."""
        for key, value in dict_of_dfs.items():
            filepath = os.path.join(self.outputlocation, f"{fileprefix}{key}.csv")
            value.to_csv(filepath)
            # logging.info(f"Exported {fileprefix}{key} to {filepath}")

    def _export_multiway_tables(self):
        """Export multiway tables."""
        outputs = self.outputs_config
        if outputs is None:
            logging.info("Skipping multiway export: no 'outputs' section configured.")
            return

        multiway_config = self._safe_get_child_config(outputs, "multiway")
        if multiway_config is None:
            logging.info("No multiway tables configured. Skipping multiway export.")
            return

        multiway_tables = self._return_multiway_tables(multiway_config)
        for (filename, filetype), table in multiway_tables.items():
            filepath = os.path.join(self.outputlocation, filename)
            table.to_csv(filepath, sep=self.filetype_sep_dict[filetype])
            logging.info(f"Exported multiway table {filename} to {filepath}")

    def _return_multiway_tables(self, multiway_config):
        """Return multiway tables based on the configuration."""
        multiway_tables = {}
        for table_config in multiway_config:
            start_time = time.time()
            variables, filename, filetype, entity = (table_config.variables.return_list(), table_config.filename, table_config.filetype, table_config.entity)
            entity_type = self.entity_types_dict[entity]
            multiway_table_entity = self._return_aggregate_by_geo(variables, entity_type, entity)
            multiway_tables[(filename, filetype)] = multiway_table_entity
            # logging.info(f"Time elapsed for each table is: {time.time() - start_time:.4f} seconds")
        return multiway_tables

    def _export_synthetic_population(self):
        """Export synthetic population data."""
        start_time = time.time()

        outputs = self.outputs_config
        if outputs is None:
            # No `outputs` section - default to writing the two core tables.
            self._export_synthetic_population_default()
            return

        synthetic_population_config = self._safe_get_child_config(outputs, "synthetic_population")
        if synthetic_population_config is None:
            # `outputs` exists but user didn't configure... for synthetic pop.
            logging.info("No 'outputs.synthetic_population' configured. Using default synthetic population filenames.")
            self._export_synthetic_population_default()
            return

        # We sort before writing to make output deterministic. Internally we keep geographic IDs
        # as both index levels (for joins) and columns (for groupby). Pandas raises an ambiguity
        # error if a label exists in both places and we call sort_values(by=...).
        sort_columns = self.pop_syn_all_id_columns

        for entity_type in self.entity_types:
            entity_config = getattr(synthetic_population_config, entity_type, None)
            if entity_config is None:
                logging.warning(
                    f"Skipping export for '{entity_type}' because it is not configured in 'synthetic_population'."
                )
                continue

            filename, filetype = entity_config.filename, entity_config.filetype
            filepath = os.path.join(self.outputlocation, filename)

            df = self.pop_syn_data[entity_type]

            # Avoid: ValueError: '<col>' is both an index level and a column label, which is ambiguous.
            if any((c in df.index.names) and (c in df.columns) for c in sort_columns):
                df = df.reset_index(drop=True)
            else:
                df = df.copy()

            # Only sort by columns that exist.
            sort_by = [c for c in sort_columns if c in df.columns]
            if sort_by:
                df.sort_values(by=sort_by, inplace=True, kind="mergesort")  # stable sort for reproducibility

            df.reset_index(drop=True, inplace=True)
            df.index.name = f"unique_{entity_type}_id"
            df.to_csv(filepath, sep=self.filetype_sep_dict[filetype])
            logging.info(f"Exported synthetic population for '{entity_type}' to {filepath}")

        logging.info(f"Time to write synthetic population files: {time.time() - start_time:.4f} seconds")


    def _export_synthetic_population_default(self):
        """Export synthetic population using default filenames.

        This is used when the YAML omits the `outputs:` section, which is common
        in minimal configs. The default filenames match the historical PopGen
        outputs:
          - housing_synthetic.csv
          - person_synthetic.csv
        """
        if not self.pop_syn_data:
            logging.warning("Skipping synthetic population export: synthetic population data not prepared.")
            return

        default_names = {
            "housing": ("housing_synthetic.csv", "csv"),
            "person": ("person_synthetic.csv", "csv"),
        }

        sort_columns = self.pop_syn_all_id_columns

        for entity_type in self.entity_types:
            if entity_type not in self.pop_syn_data:
                continue

            filename, filetype = default_names.get(entity_type, (f"{entity_type}_synthetic.csv", "csv"))
            filepath = os.path.join(self.outputlocation, filename)

            df = self.pop_syn_data[entity_type]

            # Avoid ambiguity if geo/hid exist as both index levels and columns.
            if any((c in df.index.names) and (c in df.columns) for c in sort_columns):
                df = df.reset_index(drop=True)
            else:
                df = df.copy()

            sort_by = [c for c in sort_columns if c in df.columns]
            if sort_by:
                df.sort_values(by=sort_by, inplace=True, kind="mergesort")

            df.reset_index(drop=True, inplace=True)
            df.index.name = f"unique_{entity_type}_id"

            sep = self.filetype_sep_dict.get(filetype, ",")
            df.to_csv(filepath, sep=sep)
            logging.info(f"Exported synthetic population for '{entity_type}' to {filepath}")


    def _return_aggregate_by_geo(self, variables, entity_type, entity):
        """Return aggregate data by geo."""
        if isinstance(variables, str):
            variables = [variables]
        groupby_columns = ["entity", self.geo_name] + variables

        if self.geo_name not in self.pop_syn_data[entity_type].columns and self.region_name not in self.pop_syn_data[
            entity_type].columns:
            logging.warning("Geographic ID column does not exist in the DataFrame.")

        # Do not mutate the main synthetic population table in-place.
        df = self.pop_syn_data[entity_type].reset_index(drop=True)
        multiway_table = df.groupby(groupby_columns).size()

        condition = True
        for col in variables:
            if col in multiway_table.index.names:
                condition &= (multiway_table.index.get_level_values(col) != 0)
        multiway_table = multiway_table[condition]
        multiway_table = multiway_table.reset_index('entity', drop=True)

        # After dropping 'entity', the index is [geo, var1, var2, ...].
        # We want a table indexed by geo with columns representing all variable levels.
        idx_names = list(multiway_table.index.names)
        levels_to_unstack = list(range(1, len(idx_names)))  # unstack all variable levels
        multiway_table_entity = (
            multiway_table.unstack(level=levels_to_unstack)
            if levels_to_unstack
            else multiway_table.to_frame(name="count")
        )
        return multiway_table_entity

    def _export_summary(self):
        """Export summary data."""
        outputs = self.outputs_config
        if outputs is None:
            logging.info("Skipping summary export: no 'outputs' section configured.")
            return

        summary_cfg = self._safe_get_child_config(outputs, "summary")
        if summary_cfg is None:
            logging.info("Skipping summary export: no 'outputs.summary' configured.")
            return

        start_time = time.time()
        marginal_geo = self._return_marginal_geo()

        summary_config_region = self._safe_get_child_config(summary_cfg, "region")
        summary_config_geo = self._safe_get_child_config(summary_cfg, "geo")

        if summary_config_geo is not None:
            filepath = os.path.join(self.outputlocation, summary_config_geo.filename)
            marginal_geo.to_csv(filepath, sep=self.filetype_sep_dict[summary_config_geo.filetype])
            # logging.info(f"Exported geo summary to {filepath}")
        else:
            logging.warning("Skipping geo summary export because 'geo' is not configured.")

        if summary_config_region is not None:
            marginal_region = self._return_marginal_region(marginal_geo)
            filepath = os.path.join(self.outputlocation, summary_config_region.filename)
            marginal_region.to_csv(filepath, sep=self.filetype_sep_dict[summary_config_region.filetype])
            # logging.info(f"Exported region summary to {filepath}")
        else:
            logging.warning("Skipping region summary export because 'region' is not configured.")

        # logging.info(f"Summary creation completed took: {time.time() - start_time:.4f} seconds")
        logging.info(f"Summary creation completed.")

    def _return_marginal_region(self, marginal_geo):
        """Return marginal region data."""
        region_to_geo = self.db.geo["region_to_geo"].set_index(self.geo_name)
        marginal_region = pd.concat([region_to_geo, marginal_geo], axis=1, join='inner')
        marginal_region = marginal_region.groupby(self.region_name).sum()
        marginal_region.columns = pd.MultiIndex.from_tuples(marginal_region.columns)
        return marginal_region

    def _return_marginal_geo(self):
        """Return marginal geo data."""
        marginal_list = []
        for entity in self.entities:
            entity_type = self.entity_types_dict[entity]
            for variable in self.controls[entity]:
                # Ensure at least one geographic ID exists
                if 'geo' not in self.pop_syn_data[entity_type].columns and 'region' not in self.pop_syn_data[
                    entity_type].columns:
                    print("⚠️ GEOGRAPHIC ID may not exist in the DataFrame.")

                variable_marginal = self._return_aggregate_by_geo(variable, entity_type, entity)
                marginal_list.append(variable_marginal)

        # **Fix: Properly handle stacking to avoid duplicated MultiIndex levels**
        marginal_geo = self._stack_marginal(marginal_list)

        # **Check for MultiIndex duplication**
        if isinstance(marginal_geo.columns, pd.MultiIndex):
            if marginal_geo.columns.nlevels == 2:
                if marginal_geo.columns.get_level_values(0).equals(marginal_geo.columns.get_level_values(1)):
                    marginal_geo.columns = marginal_geo.columns.get_level_values(0)  # Remove duplicate levels

        if marginal_geo.columns.duplicated().any():
            # print("⚠️ Found duplicated column names in marginal_geo, removing duplicates.")
            marginal_geo = marginal_geo.loc[:, ~marginal_geo.columns.duplicated()]

        return marginal_geo

    def _stack_marginal(self, marginal_list):
        """Stack marginal data."""
        marginal_T_list = []
        for index, marginal in enumerate(marginal_list):
            if isinstance(marginal.columns, pd.MultiIndex):
                if marginal.columns.nlevels == 2 and marginal.columns.get_level_values(0).equals(
                        marginal.columns.get_level_values(1)):
                    marginal.columns = marginal.columns.get_level_values(0)
                else:
                    raise ValueError(f"Inconsistent MultiIndex in marginal {index}. Columns: {marginal.columns}")
            marginal = marginal.T.copy()
            marginal["name"] = marginal.index.name
            marginal_T_list.append(marginal)

        stacked_marginal = pd.concat(marginal_T_list)
        stacked_marginal.index.name = "categories"
        stacked_marginal.reset_index(inplace=True)
        stacked_marginal.set_index(["name", "categories"], inplace=True)
        stacked_marginal.sort_index(inplace=True)
        return stacked_marginal.T

    def _report_summary(self, geo_id_rows_syn, geo_id_frequencies, geo_id_constraints, over_columns=None):
        """Report summary of synthetic population."""
        geo_id_synthetic = self.geo_stacked.take(geo_id_rows_syn).sum()
        geo_id_synthetic = pd.DataFrame(geo_id_synthetic, columns=["synthetic_count"])
        geo_id_synthetic["frequency"] = geo_id_frequencies
        geo_id_synthetic["constraint"] = geo_id_constraints
        geo_id_synthetic["diff_constraint"] = geo_id_synthetic["synthetic_count"] - geo_id_synthetic["constraint"]
        geo_id_synthetic["abs_diff_constraint"] = geo_id_synthetic["diff_constraint"].abs()
        geo_id_synthetic["diff_frequency"] = geo_id_synthetic["synthetic_count"] - geo_id_synthetic["frequency"]
        geo_id_synthetic["abs_diff_frequency"] = geo_id_synthetic["diff_frequency"].abs()

        stat, p_value = stats.chisquare(geo_id_synthetic["synthetic_count"], geo_id_synthetic["constraint"])
        aad_in_frequencies = geo_id_synthetic["abs_diff_frequency"].mean()
        aad_in_constraints = geo_id_synthetic["abs_diff_constraint"].mean()
        sad_in_constraints = geo_id_synthetic["abs_diff_constraint"].sum()
        sd_in_constraints = geo_id_synthetic["diff_constraint"].sum()

        logging.info(f"Chi-square test: stat={stat:.4f}, p-value={p_value:.4f}")
        logging.info(f"AAD in frequencies: {aad_in_frequencies:.4f}, AAD in constraints: {aad_in_constraints:.4f}")
        logging.info(f"SAD in constraints: {sad_in_constraints}, SD in constraints: {sd_in_constraints}")

    @staticmethod
    def get_sample_restructure(entity, sample, variable_names, hid_name):
        # print(sample)
        sample.reset_index(inplace=True, drop=True)
        sample["entity"] = entity
        # print(sample)
        # print("Number of columns in 'sample' after adding 'entity':", sample.shape[1])
        # sample.to_csv("sample_output.csv", index=True)

        groupby_columns = [hid_name, "entity"] + variable_names
        # print(groupby_columns)
        columns_count = len(groupby_columns)
        # print(columns_count)

        sample_restruct = (sample.groupby(groupby_columns)
                           .size()
                           .unstack(level=list(range(1, columns_count)))
                           .fillna(0)
                           )
        # print(sample_restruct)
        # sample_restruct.to_csv("sample_restructure.csv", index=False)
        # sys.exit()
        return sample_restruct

    @staticmethod
    def get_row_idx(sample_restruct):
        row_idx = {}
        contrib = {}
        for column in sample_restruct.columns.values.tolist():
            rows = np.where(sample_restruct[column] > 0)[0]
            row_idx[column] = rows
            contrib[column] = np.array(
                sample_restruct[column].values, order="C", dtype=int)
        return (row_idx, contrib)

    @staticmethod
    def get_stacked_sample_restruct(sample_restruct_list):
        if len(sample_restruct_list) == 0:
            return None
        elif len(sample_restruct_list) == 1:
            return sample_restruct_list[0]

        # Initialize stacked sample with the first element
        stacked_sample = sample_restruct_list[0]

        for sample_restruct in sample_restruct_list[1:]:
            # Merge using pd.concat and fill missing values with 0
            stacked_sample = pd.concat([stacked_sample, sample_restruct], axis=1, join='outer').fillna(0)

        # Sort row indices
        stacked_sample.sort_index(inplace=True)

        # Sort columns alphabetically
        stacked_sample.sort_index(axis=1, inplace=True)

        # Save to CSV file
        # stacked_sample.to_csv("stacked_sample_restruct.csv", index=True)

        return stacked_sample


if __name__ == "__main__":
    pass
    # location = "popgen"
    # db = pd.read_csv("popgen/data.csv")
    # column_names_config = ColumnNamesConfig("popgen/column_names.yaml")
    # scenario_config = ScenarioConfig("popgen/scenario.yaml")
    # run_ipf_obj = RunIPF(location, db, column_names_config, scenario_config)
    # run_ipu_obj = RunIPU(location, db, column_names_config, scenario_config, run_ipf_obj)
    # draw_population_obj = DrawPopulation(location, db, column_names_config, scenario_config, run_ipu_obj)
    # entities = ["household", "person"]
    # housing_entities = ["household"]
    # person_entities = ["person"]
    # syn_population = Syn_Population(location, db, column_names_config, scenario_config, run_ipf_obj, run_ipu_obj,
    #                                 draw_population_obj, entities, housing_entities, person_entities)
    # syn_population.add_records()
    # syn_population.prepare_data()
    # syn_population.export_outputs()
    # print("Synthetic population generation and output completed.")