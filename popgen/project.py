import logging
import os
import time
import yaml

from .config import Config
from .data import DB
from .ipf import Run_IPF
from .reweighting import Run_Reweighting, Reweighting_DS
from .draw import Draw_Population
from .output import Syn_Population

class Project(object):
    """This is the primary class to setup and run PopGen projects."""
    def __init__(self, config_loc):
        self.config_loc = config_loc

    def load_project(self):
        self._load_config()
        self._populate_project_properties()
        self._load_data()

    def _load_config(self):
        with open(self.config_loc, "r") as config_f:
            config_dict = yaml.safe_load(config_f)
        logging.info(f"Loaded configuration: {config_dict}")  # Add this line for debugging
        self._config = Config(config_dict)
        self.column_names_config = self._config.project.inputs.column_names
        self.entities = self._config.project.inputs.entities
        self.housing_entities = self._config.project.inputs.housing_entities
        self.person_entities = self._config.project.inputs.person_entities

    def _populate_project_properties(self):
        self.name = self._config.project.name
        self.location = os.path.abspath(self._config.project.location)

    def _load_data(self):
        self.db = DB(self._config)
        self.db.load_data()

    def run_scenarios(self):
        scenarios_config = self._config.project.scenario
        synthesize = getattr(self._config.project, 'synthesize', True)  # Moved here
        for scenario_config in scenarios_config:
            logging.info(f"Running Scenario: {scenario_config.description}")
            scenario_obj = Scenario(self.location,
                                    self.entities, self.housing_entities,
                                    self.person_entities,
                                    self.column_names_config,
                                    scenario_config, self.db,
                                    synthesize)  # Pass synthesize to Scenario
            scenario_obj.run_scenario()

class Scenario(object):
    def __init__(self, location, entities, housing_entities, person_entities,
                 column_names_config, scenario_config, db, synthesize):
        self.location = location
        self.entities = entities
        self.housing_entities = housing_entities
        self.person_entities = person_entities
        self.column_names_config = column_names_config
        self.scenario_config = scenario_config
        self.db = db
        self.synthesize = synthesize  # Store synthesize value
        self.t = time.time()

    def run_scenario(self):
        self._get_geo_ids()
        self._run_ipf()
        self._run_weighting()

        logging.info(f"Synthesize value: {self.synthesize}")  # Add this line for debugging
        if self.synthesize is not False:  # Check for both None and True
            self._draw_sample()
            self._report_results()
        else:
            self._output_weights_only()

    def _get_geo_ids(self):
        self.db.enumerate_geo_ids_for_scenario(self.scenario_config)

    def _run_ipf(self):
        self.run_ipf_obj = Run_IPF(self.entities,
                                   self.housing_entities,
                                   self.column_names_config,
                                   self.scenario_config, self.db)
        self.run_ipf_obj.run_ipf()
        logging.info(f"IPF completed in: {time.time() - self.t:.4f}")

    def _run_weighting(self):
        self.run_reweighting_obj = Run_Reweighting(
            self.entities, self.column_names_config,
            self.scenario_config, self.db)
        self.run_reweighting_obj.create_ds()
        self.run_reweighting_obj.run_reweighting(
            self.run_ipf_obj.region_constraints,
            self.run_ipf_obj.geo_constraints)
        logging.info(f"Reweighting completed in: {time.time() - self.t:.4f}")

    def _draw_sample(self):
        self.draw_population_obj = Draw_Population(
            self.scenario_config, self.db.geo_ids,
            self.run_reweighting_obj.geo_row_idx,
            self.run_ipf_obj.geo_frequencies,
            self.run_ipf_obj.geo_constraints,
            self.run_reweighting_obj.geo_stacked,
            self.run_reweighting_obj.region_sample_weights)
        self.draw_population_obj.draw_population()
        logging.info(f"Drawing completed in: {time.time() - self.t:.4f}")

    def _report_results(self):
        self.syn_pop_obj = Syn_Population(
            self.location,
            self.db,
            self.column_names_config,
            self.scenario_config,
            self.run_ipf_obj,
            self.run_reweighting_obj,
            self.draw_population_obj,
            self.entities,
            self.housing_entities,
            self.person_entities)

        self.syn_pop_obj.add_records()
        self.syn_pop_obj.prepare_data()
        self.syn_pop_obj.export_outputs()
        logging.info(f"Results completed in: {time.time() - self.t:.4f}")

    def _output_weights_only(self):
        weights_output_path = os.path.join(self.location, 'weights_output.csv')
        self.run_reweighting_obj.region_sample_weights.to_csv(weights_output_path, float_format='%.10f', index=False)
        logging.info(f"Sample weights saved to {weights_output_path}")

def popgen_run(project_config):
    logging.basicConfig(level=logging.INFO)
    t = time.time()
    p_obj = Project(project_config)
    p_obj.load_project()
    p_obj.run_scenarios()
    logging.info(f"Time it took: {time.time() - t:.4f}")

def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'configuration_arizona.yaml')
    os.chdir(os.path.join(os.path.dirname(__file__), '..', 'data'))
    popgen_run(config_path)


if __name__ == "__main__":
    main()