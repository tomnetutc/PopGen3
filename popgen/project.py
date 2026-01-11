import logging
import os
import time
import yaml
import sys
import pandas as pd



from .config import Config
from .data import DB
from .ipf import Run_IPF
from .reweighting import Run_Reweighting
from .draw import Draw_Population
from .output import Syn_Population



class Project:
    """Primary class to set up and run PopGen projects."""

    def __init__(self, config_loc):
        self.config_loc = config_loc
        self._config = None
        self.db = None

    def load_project(self):
        """Loads the project configuration and initializes necessary components."""
        try:
            self._load_config()
            self._populate_project_properties()
            self._load_data()
        except Exception as e:
            logging.error(f"Failed to load project: {e}", exc_info=True)
            raise

    def _load_config(self):
        """Loads configuration from the YAML file."""
        try:
            with open(self.config_loc, "r") as config_f:
                config_dict = yaml.safe_load(config_f)
            logging.info("Configuration loaded successfully.")
            self._config = Config(config_dict)
        except FileNotFoundError:
            logging.critical(f"Configuration file {self.config_loc} not found.")
            raise
        except yaml.YAMLError as e:
            logging.critical("Error parsing YAML configuration file.", exc_info=True)
            raise

    def _populate_project_properties(self):
        """Extracts basic project properties from configuration."""
        self.name = self._config.project.name
        self.location = os.path.abspath(self._config.project.location)
        logging.info(f"Project initialized: {self.name} at {self.location}")

    def _load_data(self):
        """Loads the project database."""
        self.db = DB(self._config)
        self.db.load_data()
        logging.info("Database loaded successfully.")

    def run_scenarios(self):
        """Runs all configured scenarios."""
        scenarios_config = self._config.project.scenario

        # Project-level default (can be overridden per scenario).
        # Use raw access to avoid warning spam when the key is optional.
        project_synthesize = self._config.project.get_raw('synthesize', True)
        if project_synthesize is None:
            project_synthesize = True
        project_synthesize = bool(project_synthesize)

        for scenario_config in scenarios_config:
            if "parameters" not in scenario_config._data or scenario_config._data["parameters"] is None:
                logging.warning("'parameters' missing in scenario. Using default values.")
                scenario_config._data["parameters"] = Config.DEFAULT_PARAMETERS

            logging.info(f"Running scenario: {scenario_config.description}")

            # Whether to run in 2-level (region->geo) mode or single-level mode.
            # Raw access avoids warning spam for optional keys.
            apply_cross_level = bool(scenario_config.get_raw("apply_cross_level", True))
            if not apply_cross_level:
                self.process_geo_region_mappings(scenario_config)

            try:
                # Scenario-level override (optional).
                # Use raw access so missing keys don't produce warnings.
                scenario_synthesize = scenario_config.get_raw('synthesize', None)
                if scenario_synthesize is None:
                    scenario_synthesize = project_synthesize
                scenario_synthesize = bool(scenario_synthesize)

                scenario_obj = Scenario(
                    self.location,
                    self._config.project.inputs.entities,
                    self._config.project.inputs.housing_entities,
                    self._config.project.inputs.person_entities,
                    self._config.project.inputs.column_names,
                    scenario_config,
                    self.db,
                    scenario_synthesize
                )
                scenario_obj.run_scenario()
            except Exception as e:
                logging.error(f"Error running scenario: {scenario_config.description}", exc_info=True)

    def process_geo_region_mappings(self, scenario_config):
        """Enable single-level workflows.

        The upstream PopGen workflow assumes two geography levels:
        a *region* level and a finer *geo* level, with a mapping between them.

        In many real projects you only have **one** geography level available.
        Setting ``apply_cross_level: false`` in a scenario is intended to let you
        run PopGen by treating *region* and *geo* as the same level.

        This function makes that behaviour reliable by:
        - Ensuring BOTH ``control_variables.region`` and ``control_variables.geo`` exist
          by aliasing whichever one is provided.
        - Ensuring BOTH ``region_marginals`` and ``geo_marginals`` exist by aliasing.
        - Creating an **identity** ``region_to_geo`` mapping (region_id == geo_id).
        - Ensuring ``region_to_sample`` and ``geo_to_sample`` exist and are correctly
          indexed (no resetting the index / losing ids).

        The goal is that with only one set of marginals + one geo_to_sample mapping,
        you do NOT need to prepare a redundant second set of inputs.
        """

        # ------------------------------------------------------------------
        # Column names from YAML (with safe defaults)
        # ------------------------------------------------------------------
        col_names = self._config.project.inputs.column_names

        # Fill missing column name keys *quietly*.
        if isinstance(col_names._data, dict):
            col_names._data.setdefault("geo", "geo")
            col_names._data.setdefault("region", col_names._data.get("geo", "region"))
            col_names._data.setdefault("sample_geo", "sample_geo")

        region_col = col_names.get_raw("region", col_names.get_raw("geo", "region"))
        sample_geo_col = col_names.get_raw("sample_geo", "sample_geo")

        # In single-level mode we treat geo as region.
        # This keeps downstream code (which references column_names.geo) consistent.
        try:
            col_names["geo"] = region_col
        except Exception:
            # Config wrapper supports __setitem__, but guard just in case.
            pass
        geo_col = col_names.get_raw("geo", region_col)

        # ---- 1) Alias control variables (region <-> geo) ----
        control_vars = getattr(scenario_config, "control_variables", None)
        if control_vars is not None:
            # Use raw lookups to avoid warning spam in single-level configs where
            # users intentionally omit one side.
            region_cv = control_vars.get("region", None) if hasattr(control_vars, "get") else getattr(control_vars, "region", None)
            geo_cv = control_vars.get("geo", None) if hasattr(control_vars, "get") else getattr(control_vars, "geo", None)

            # If only one side exists, copy it to the other.
            if (region_cv is None or len(region_cv) == 0) and geo_cv is not None:
                control_vars.region = geo_cv
            if (geo_cv is None or len(geo_cv) == 0) and region_cv is not None:
                control_vars.geo = region_cv

        # ---- 2) Alias marginals (region <-> geo) ----
        # Many single-level projects only provide one of these.
        if (not self.db.region_marginals) and self.db.geo_marginals:
            self.db.region_marginals = self.db.geo_marginals
        if (not self.db.geo_marginals) and self.db.region_marginals:
            self.db.geo_marginals = self.db.region_marginals

        # ---- 3) Determine the set of geography ids we should operate on ----
        ids = None
        for key in ("region_to_sample", "region_to_geo", "geo_to_sample"):
            df = self.db.geo.get(key)
            if isinstance(df, pd.DataFrame) and (not df.empty):
                ids = list(pd.Index(df.index).unique())
                break

        if not ids:
            # Try infer from marginals.
            marginals_dict = self.db.region_marginals or self.db.geo_marginals
            if marginals_dict:
                first_df = next(iter(marginals_dict.values()))
                if isinstance(first_df, pd.DataFrame) and (len(first_df.index) > 0):
                    ids = list(pd.Index(first_df.index).unique())

        if not ids:
            raise RuntimeError(
                "apply_cross_level=False but no geography ids could be inferred. "
                "Provide at least one of: geo_to_sample, region_to_sample, region_to_geo, "
                "or a marginals table with a non-empty index."
            )

        # ---- 4) Build identity region_to_geo mapping ----
        # Index: region id, Column: geo id (same values).
        region_to_geo = pd.DataFrame({geo_col: ids}, index=ids)
        region_to_geo.index.name = region_col
        self.db.geo["region_to_geo"] = region_to_geo

        # ---- 5) Ensure sample geography mappings exist and are well-formed ----
        def _ensure_sample_map(df, idx_name: str):
            if not isinstance(df, pd.DataFrame) or df.empty:
                return None
            df = df.copy()
            # If the expected sample_geo column is missing but there's exactly one column,
            # assume that column is the sample_geo id.
            if sample_geo_col not in df.columns:
                if df.shape[1] == 1:
                    df.columns = [sample_geo_col]
                else:
                    # Try a case-insensitive match.
                    matches = [c for c in df.columns if str(c).lower() == str(sample_geo_col).lower()]
                    if matches:
                        df = df.rename(columns={matches[0]: sample_geo_col})
                    else:
                        raise RuntimeError(
                            f"Mapping table is missing required column '{sample_geo_col}'. "
                            f"Found columns: {list(df.columns)}"
                        )
            df.index.name = idx_name
            return df[[sample_geo_col]]

        region_to_sample = _ensure_sample_map(self.db.geo.get("region_to_sample"), region_col)
        geo_to_sample = _ensure_sample_map(self.db.geo.get("geo_to_sample"), geo_col)

        if region_to_sample is None and geo_to_sample is not None:
            region_to_sample = geo_to_sample.copy()
            region_to_sample.index.name = region_col
        if geo_to_sample is None and region_to_sample is not None:
            geo_to_sample = region_to_sample.copy()
            geo_to_sample.index.name = geo_col

        if region_to_sample is None and geo_to_sample is None:
            # Default: -1 means "use all samples" (see IPF logic).
            default_map = pd.DataFrame({sample_geo_col: [-1] * len(ids)}, index=ids)
            default_map.index.name = region_col
            region_to_sample = default_map.copy()
            geo_to_sample = default_map.copy()
            geo_to_sample.index.name = geo_col

        self.db.geo["region_to_sample"] = region_to_sample
        self.db.geo["geo_to_sample"] = geo_to_sample

        # Recompute cached id lists so later defaults are consistent.
        try:
            self.db._enumerate_geo_ids()
        except Exception:
            logging.warning("Failed to re-enumerate geo ids after applying single-level mapping.", exc_info=True)

        # Finally, ensure geo-level controls exist for downstream drawing.
        if control_vars is not None:
            scenario_config.control_variables.geo = scenario_config.control_variables.region



class Scenario:
    """Class to manage and execute a scenario."""

    def __init__(self, location, entities, housing_entities, person_entities,
                 column_names_config, scenario_config, db, synthesize):
        self.location = location
        self.entities = entities
        self.housing_entities = housing_entities
        self.person_entities = person_entities
        self.column_names_config = column_names_config
        self.scenario_config = scenario_config
        self.db = db
        self.synthesize = synthesize
        self.start_time = time.time()

    def run_scenario(self):
        """Executes the scenario."""
        try:
            self._get_geo_ids()
            self._run_ipf()
            self._run_weighting()

            if self.synthesize:
                # Drawing/synthesis requires geo-level constraints and integerized frequencies.
                if self.run_ipf_obj.geo_constraints is None or self.run_ipf_obj.geo_frequencies is None:
                    raise RuntimeError(
                        "Scenario is configured to synthesize a population, but geo-level IPF constraints "
                        "and/or integerized frequencies are missing. Ensure you have geo controls enabled "
                        "(or apply_cross_level=False to treat region as geo), and rounding_procedure='bucket'."
                    )
                self._draw_sample()
                self._report_results()
            else:
                self._output_weights_only()
        except Exception as e:
            logging.error(f"Scenario execution failed: {self.scenario_config.description}", exc_info=True)

    def _get_geo_ids(self):
        """Enumerates geographical IDs for the scenario."""
        self.db.enumerate_geo_ids_for_scenario(self.scenario_config)
        logging.info("Geographical IDs enumerated successfully.")

    def _run_ipf(self):
        """Runs iterative proportional fitting (IPF)."""
        self.run_ipf_obj = Run_IPF(
            self.entities, self.housing_entities,
            self.column_names_config, self.scenario_config, self.db
        )
        self.run_ipf_obj.run_ipf()
        logging.info(f"IPF completed in: {time.time() - self.start_time:.4f} seconds")

    def _run_weighting(self):
        """Runs the reweighting process."""
        self.run_reweighting_obj = Run_Reweighting(
            self.entities, self.column_names_config, self.scenario_config, self.db
        )
        self.run_reweighting_obj.create_ds()
        self.run_reweighting_obj.run_reweighting(
            self.run_ipf_obj.region_constraints,
            self.run_ipf_obj.geo_constraints if self.run_ipf_obj.geo_constraints is not None else None
        )
        logging.info(f"Reweighting completed in: {time.time() - self.start_time:.4f} seconds")

    def _draw_sample(self):
        """Draws the synthetic population sample."""
        self.draw_population_obj = Draw_Population(
            self.scenario_config, self.db.geo_ids,
            self.run_reweighting_obj.geo_row_idx,
            self.run_ipf_obj.geo_frequencies,
            self.run_ipf_obj.geo_constraints,
            self.run_reweighting_obj.geo_stacked,
            self.run_reweighting_obj.region_sample_weights
        )
        self.draw_population_obj.draw_population()
        logging.info(f"Drawing completed in: {time.time() - self.start_time:.4f} seconds")

    def _report_results(self):
        self.syn_pop_obj = Syn_Population(
            self.location, self.db, self.column_names_config,
            self.scenario_config, self.run_ipf_obj,
            self.run_reweighting_obj, self.draw_population_obj,
            self.entities, self.housing_entities, self.person_entities
        )

        self.syn_pop_obj.add_records()
        self.syn_pop_obj.prepare_data()
        logging.debug(f"Scenario config type: {type(self.scenario_config)}")

        self.syn_pop_obj.export_outputs()
        logging.info(f"Results generated in: {time.time() - self.start_time:.4f} seconds")

    def _output_weights_only(self):
        """Writes weights and (optionally) diagnostics without drawing/synthesizing a population."""

        # Create an output folder consistent with the full synthesis workflow.
        current_time_str = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        foldername = f"{current_time_str} {self.scenario_config.description}"
        outputlocation = os.path.join(self.location, foldername)
        os.makedirs(outputlocation, exist_ok=True)

        # Save weights WITH the household id index.
        hid_name = getattr(self.column_names_config, "hid", "hid")
        weights_output_path = os.path.join(outputlocation, "weights.csv")
        self.run_reweighting_obj.region_sample_weights.to_csv(
            weights_output_path,
            float_format='%.10f',
            index=True,
            index_label=hid_name,
        )
        logging.info(f"Sample weights saved to {weights_output_path}")

        # Save reweighting performance (helpful for debugging convergence).
        try:
            perf_path = os.path.join(outputlocation, "reweighting_average_diffs.csv")
            self.run_reweighting_obj.average_diffs.to_csv(perf_path)
        except Exception:
            logging.warning("Failed to write reweighting performance diagnostics.", exc_info=True)

        # Write a copy of the scenario configuration used.
        try:
            self.scenario_config.write_to_file(os.path.join(outputlocation, f"{self.scenario_config.description}.yaml"))
        except Exception:
            logging.warning("Failed to write scenario configuration to output folder.", exc_info=True)


def popgen_run(project_config):
    """Entry point for running PopGen project."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    start_time = time.time()
    p_obj = Project(project_config)
    p_obj.load_project()
    p_obj.run_scenarios()
    logging.info(f"Total execution time: {time.time() - start_time:.4f} seconds")
