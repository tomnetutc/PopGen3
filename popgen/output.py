import os
import time
import pandas as pd
import numpy as np
# from .config import Config

class Syn_Population:
    def __init__(self, location, db, column_names_config, scenario_config, run_ipf_obj, run_ipu_obj, draw_population_obj, entities, housing_entities, person_entities):
        self.location = location
        self.db = db
        self.column_names_config = column_names_config
        self.scenario_config = scenario_config

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

    def _create_preliminaries(self):
        """Initialize preliminary data and configurations."""
        self._create_ds()
        self._create_meta_data()
        self._create_prepare_output_directory()

    def _create_ds(self):
        """Create stacked samples for housing and person entities."""
        self.housing_stacked_sample = self._get_stacked_sample(self.housing_entities)
        self.housing_stacked_sample.set_index('hid', inplace=True)

        self.person_stacked_sample = self._get_stacked_sample(self.person_entities)
        self.person_stacked_sample.set_index('hid', inplace=True)

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

    def _get_stacked_sample(self, entities):
        """Get stacked sample data for the given entities."""
        sample_list = [self.db.sample[entity] for entity in entities]
        stacked_sample = pd.concat(sample_list).fillna(0)
        stacked_sample.sort_index(inplace=True)
        return stacked_sample

    def add_records(self):
        """Add records for synthetic population."""
        geo_id_rows_syn_dict = self.draw_population_obj.geo_id_rows_syn_dict
        for geo_id, geo_id_rows_syn in geo_id_rows_syn_dict.items():
            geo_id_pop_syn = self._get_stacked_geo_for_geo_id(geo_id, geo_id_rows_syn)
            self.pop_rows_syn_dict[geo_id] = geo_id_pop_syn
            self.pop_rows_syn_dict[geo_id][self.unique_id_in_geo_name] = range(1, geo_id_rows_syn.shape[0] + 1)

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

    def _stack_records(self):
        """Stack records for the synthetic population."""
        start_time = time.time()
        self.pop_syn = pd.concat(self.pop_rows_syn_dict.values(), copy=False)
        print(f"Time elapsed for stacking population is: {time.time() - start_time:.4f} seconds")

    def _create_synthetic_population(self):
        """Create synthetic population data."""
        start_time = time.time()

        self.pop_syn_data["housing"] = self.pop_syn.loc[:, self.pop_syn_geo_id_columns].join(self.housing_stacked_sample)
        self.pop_syn_data["person"] = self.pop_syn.loc[:, self.pop_syn_geo_id_columns].join(self.person_stacked_sample)

        print(f"Size of the housing population table: {self.pop_syn_data['housing'].shape}")
        print(f"Size of the person population table: {self.pop_syn_data['person'].shape}")

    def _create_index(self):
        """Create indexes for synthetic population data."""
        self.pop_syn_data["housing"].reset_index(inplace=True)
        self.pop_syn_data["housing"].set_index(self.pop_syn_housing_matching_id_columns, inplace=True, drop=False)
        self.pop_syn_data["housing"].sort_index(inplace=True)

        self.pop_syn_data["person"].reset_index(inplace=True)
        self.pop_syn_data["person"].set_index(self.pop_syn_person_matching_id_columns, inplace=True, drop=False)
        self.pop_syn_data["person"].sort_index(inplace=True)

    def export_outputs(self):
        """Export all output data."""
        start_time = time.time()
        self._export_performance_data()
        self._export_multiway_tables()
        self._export_summary()
        self._export_weights()
        self._export_synthetic_population()
        self._pretty_print_scenario_configuration_file_to_output()
        print(f"Time elapsed for generating outputs is: {time.time() - start_time:.4f} seconds")

    def _pretty_print_scenario_configuration_file_to_output(self):
        """Pretty print the scenario configuration file to the output location."""
        filepath = os.path.join(self.outputlocation, f"{self.scenario_config.description}.yaml")
        self.scenario_config.write_to_file(filepath)

    def _export_performance_data(self):
        """Export performance data."""
        values_to_export = self.scenario_config.outputs.performance
        if "ipf" in values_to_export:
            self._export_all_df_in_dict(self.run_ipf_obj.geo_iters_convergence_dict, "ipf_geo_iters_convergence_")
            self._export_all_df_in_dict(self.run_ipf_obj.geo_average_diffs_dict, "ipf_geo_average_diffs_")
            self._export_all_df_in_dict(self.run_ipf_obj.region_iters_convergence_dict, "ipf_region_iters_convergence_")
            self._export_all_df_in_dict(self.run_ipf_obj.region_average_diffs_dict, "ipf_region_average_diffs_")
        if "reweighting" in values_to_export:
            self._export_df(self.run_ipu_obj.average_diffs, "reweighting_average_diffs")
        if "drawing" in values_to_export:
            self._export_df(self.draw_population_obj.draws_performance, "draws")

    def _export_weights(self):
        """Export weights data."""
        export_weights_config = self.scenario_config.outputs.weights
        if export_weights_config.export:
            df = pd.DataFrame(self.run_ipu_obj.region_sample_weights)
            if export_weights_config.collate_across_geos:
                df = df.sum(axis=1)
            filepath = os.path.join(self.outputlocation, "weights.csv")
            df.to_csv(filepath, float_format='%.10f')

    def _export_df(self, df, filename):
        """Export a DataFrame to a CSV file."""
        filepath = os.path.join(self.outputlocation, f"{filename}.csv")
        df.to_csv(filepath)

    def _export_all_df_in_dict(self, dict_of_dfs, fileprefix):
        """Export all DataFrames in a dictionary to CSV files."""
        for key, value in dict_of_dfs.items():
            filepath = os.path.join(self.outputlocation, f"{fileprefix}{key}.csv")
            value.to_csv(filepath)

    def _export_multiway_tables(self):
        """Export multiway tables."""
        multiway_tables = self._return_multiway_tables()
        for (filename, filetype), table in multiway_tables.items():
            filepath = os.path.join(self.outputlocation, filename)
            table.to_csv(filepath, sep=self.filetype_sep_dict[filetype])

    def _return_multiway_tables(self):
        """Return multiway tables based on the configuration."""
        multiway_tables = {}
        for table_config in self.scenario_config.outputs.multiway:
            start_time = time.time()
            variables, filename, filetype, entity = (table_config.variables.return_list(), table_config.filename, table_config.filetype, table_config.entity)
            entity_type = self.entity_types_dict[entity]
            multiway_table_entity = self._return_aggregate_by_geo(variables, entity_type, entity)
            multiway_tables[(filename, filetype)] = multiway_table_entity
            print(f"Time elapsed for each table is: {time.time() - start_time:.4f} seconds")
        return multiway_tables

    def _export_synthetic_population(self):
        """Export synthetic population data."""
        start_time = time.time()
        synthetic_population_config = self.scenario_config.outputs.synthetic_population
        sort_columns = self.pop_syn_all_id_columns
        for entity_type in self.entity_types:
            filename, filetype = synthetic_population_config[entity_type].filename, synthetic_population_config[entity_type].filetype
            filepath = os.path.join(self.outputlocation, filename)
            self.pop_syn_data[entity_type].sort_values(by=sort_columns, inplace=True)
            self.pop_syn_data[entity_type].reset_index(drop=True, inplace=True)
            self.pop_syn_data[entity_type].index.name = f"unique_{entity_type}_id"
            self.pop_syn_data[entity_type].to_csv(filepath, sep=self.filetype_sep_dict[filetype])
        print(f"Time to write syn pop files is: {time.time() - start_time:.4f} seconds")

    def _return_aggregate_by_geo(self, variables, entity_type, entity):
        """Return aggregate data by geo."""
        if isinstance(variables, str):
            variables = [variables]
        groupby_columns = ["entity", self.geo_name] + variables
        columns_count = len(groupby_columns)

        if 'geo' not in self.pop_syn_data[entity_type].columns:
            print("Column 'geo' does not exist in the DataFrame.")

        self.pop_syn_data[entity_type].reset_index(drop=True, inplace=True)
        multiway_table = self.pop_syn_data[entity_type].groupby(groupby_columns).size()

        columns_to_check = [col for col in groupby_columns if col not in ['entity', 'geo']]
        condition = True
        for col in columns_to_check:
            index_pos = multiway_table.index.names.index(col)
            condition &= (multiway_table.index.get_level_values(index_pos) != 0)
        multiway_table = multiway_table[condition]
        multiway_table = multiway_table.reset_index('entity', drop=True)
        multiway_table_entity = multiway_table.unstack(level=[1, columns_count - 2])
        return multiway_table_entity

    def _export_summary(self):
        """Export summary data."""
        start_time = time.time()
        summary_config = self.scenario_config.outputs.summary
        marginal_geo = self._return_marginal_geo()
        geo_filename, geo_filetype = summary_config.geo.filename, summary_config.geo.filetype
        filepath = os.path.join(self.outputlocation, geo_filename)
        marginal_geo.to_csv(filepath, sep=self.filetype_sep_dict[geo_filetype])

        marginal_region = self._return_marginal_region(marginal_geo)
        region_filename, region_filetype = summary_config.region.filename, summary_config.region.filetype
        filepath = os.path.join(self.outputlocation, region_filename)
        marginal_region.to_csv(filepath, sep=self.filetype_sep_dict[region_filetype])
        print(f"Summary creation took: {time.time() - start_time:.4f} seconds")

    def _return_marginal_region(self, marginal_geo):
        """Return marginal region data."""
        region_to_geo = self.db.geo["region_to_geo"]
        region_to_geo.reset_index(inplace=True, drop=True)
        region_to_geo.set_index(self.geo_name, inplace=True)
        marginal_region = pd.concat([region_to_geo, marginal_geo], axis=1, join='inner')
        marginal_region = marginal_region.reset_index()
        marginal_region.set_index(self.region_name, inplace=True)
        marginal_region = marginal_region[marginal_geo.columns]
        marginal_region = marginal_region.reset_index().groupby(self.region_name).sum()
        marginal_region.columns = pd.MultiIndex.from_tuples(marginal_region.columns)
        return marginal_region

    def _return_marginal_geo(self):
        """Return marginal geo data."""
        marginal_list = []
        for entity in self.entities:
            entity_type = self.entity_types_dict[entity]
            for variable in self.controls[entity]:
                if 'geo' not in self.pop_syn_data[entity_type].columns:
                    print(f"'geo' column missing in DataFrame for {entity_type}")
                variable_marginal = self._return_aggregate_by_geo(variable, entity_type, entity)
                marginal_list.append(variable_marginal)
        marginal_geo = self._stack_marginal(marginal_list)
        return marginal_geo

    def _stack_marginal(self, marginal_list):
        """Stack marginal data."""
        marginal_T_list = []
        for index, marginal in enumerate(marginal_list):
            if isinstance(marginal.columns, pd.MultiIndex):
                if marginal.columns.nlevels == 2 and marginal.columns.get_level_values(0).equals(marginal.columns.get_level_values(1)):
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

        print(f"{stat:.4f}, {p_value}, {aad_in_frequencies}, {aad_in_constraints}, {sad_in_constraints}, {sd_in_constraints}")


if __name__ == "__main__":
    # Example instantiation
    # Replace with actual parameters and remove comments to run
    # location = "path/to/location"
    # db = database_object
    # column_names_config = column_names_config_object
    # scenario_config = scenario_config_object
    # run_ipf_obj = run_ipf_object
    # run_ipu_obj = run_ipu_object
    # draw_population_obj = draw_population_object
    # entities = list_of_entities
    # housing_entities = list_of_housing_entities
    # person_entities = list_of_person_entities

    # syn_population = Syn_Population(location, db, column_names_config, scenario_config, run_ipf_obj, run_ipu_obj, draw_population_obj, entities, housing_entities, person_entities)
    # syn_population.add_records()
    # syn_population.prepare_data()
    # syn_population.export_outputs()
    pass
