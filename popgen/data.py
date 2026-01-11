import os
import pandas as pd
import numpy as np
import logging
from .config import Config, ConfigError


class DB:
    """Handles all PopGen input data files and maintains necessary mappings and datasets."""

    def __init__(self, config):
        self.config = config
        self.sample = {}
        self.geo_marginals = {}
        self.region_marginals = {}
        self.geo = {}
        self.geo_ids = None
        self.region_ids = None
        self.sample_geo_ids = None
        self._inputs_config = self.config.project.inputs
        self.location = os.path.abspath(self.config.project.location)

    def load_data(self):
        """Loads all required data while handling missing data gracefully."""
        logging.info("Loading database data...")

        # Use the non-warning helpers on Config for optional keys.
        location_cfg = getattr(self._inputs_config, "location", None)

        # Load geo correspondence mapping
        geo_corr_mapping_config = location_cfg.get("geo_corr_mapping", None) if isinstance(location_cfg, Config) else None
        self.geo = self.get_data(geo_corr_mapping_config) if geo_corr_mapping_config else {}

        # Load sample data
        sample_config = location_cfg.get("sample", None) if isinstance(location_cfg, Config) else None
        self.sample = self.get_data(sample_config) if sample_config else {}

        # Load marginals
        marginals_config = location_cfg.get("marginals", None) if isinstance(location_cfg, Config) else None
        marginals_geo_cfg = marginals_config.get("geo", None) if isinstance(marginals_config, Config) else None
        marginals_region_cfg = marginals_config.get("region", None) if isinstance(marginals_config, Config) else None

        self.geo_marginals = self.get_data(marginals_geo_cfg, header=[0, 1]) if marginals_geo_cfg else {}
        self.region_marginals = self.get_data(marginals_region_cfg, header=[0, 1]) if marginals_region_cfg else {}

        try:
            self._enumerate_geo_ids()
        except Exception as e:
            logging.warning(f"_enumerate_geo_ids failed due to {e}. Defaulting to empty lists.")
            self.geo_ids = []
            self.region_ids = []

    def get_data(self, config, header=0):
        """Loads data while handling missing files gracefully."""
        config_dict = config.return_dict() if config else {}
        data_dict = {}

        for item, filename in config_dict.items():
            if filename is None:
                logging.warning(f"{item} has no filename specified. Skipping.")
                continue

            full_location = os.path.join(self.location, filename)
            if os.path.exists(full_location):
                try:
                    data_dict[item] = pd.read_csv(full_location, index_col=0, header=header)
                    if data_dict[item].index.name:
                        data_dict[item].loc[:, data_dict[item].index.name] = data_dict[item].index.values
                except Exception as e:
                    logging.warning(f"Failed to load {filename} due to {e}. Skipping {item}.")
            else:
                logging.warning(f"{filename} not found. Skipping {item}.")

        return data_dict

    def _enumerate_geo_ids(self):
        """Ensures proper initialization of geo and region IDs."""
        self.geo_ids_all = []
        self.region_ids_all = []

        try:
            if "geo_to_sample" in self.geo:
                self.geo_ids_all = self.geo["geo_to_sample"].index.tolist()

            if "region_to_geo" in self.geo:
                self.region_ids_all = np.unique(self.geo["region_to_geo"].index.values).tolist()
            elif "region_to_sample" in self.geo:
                self.region_ids_all = np.unique(self.geo["region_to_sample"].index.values).tolist()
        except Exception as e:
            logging.warning(f"Failed to enumerate geo IDs due to {e}. Defaulting to empty lists.")
            self.geo_ids_all = []
            self.region_ids_all = []

    def get_geo_ids_for_region(self, region_id):
        """Retrieves geo IDs corresponding to a given region ID."""
        geo_name = self._inputs_config.column_names.geo
        geo_list = self.geo["region_to_geo"].loc[region_id, geo_name]
        return [int(geo_list)] if isinstance(geo_list, (int, np.integer)) else list(geo_list)

    def enumerate_geo_ids_for_scenario(self, scenario_config):
        """Resolve which region ids and geo ids should be processed for a scenario.

        Key design goals:
        - Be permissive with YAML (many keys are optional).
        - Avoid surprising "run all" behavior when the user explicitly provided an
          ids list (especially for single-level runs).
        - Keep the original PopGen behavior for multi-level (apply_cross_level=True):
          weighting loops over *regions*.
        """

        def _as_list(v):
            if v is None:
                return None
            if isinstance(v, Config):
                return v.return_list()
            if isinstance(v, (list, tuple, set)):
                return list(v)
            return [v]

        def _normalize_id(x):
            # numpy scalar -> python scalar
            try:
                import numpy as _np
                if isinstance(x, _np.generic):
                    x = x.item()
            except Exception:
                pass

            if isinstance(x, str):
                s = x.strip()
                if s.isdigit():
                    try:
                        return int(s)
                    except Exception:
                        return x
            return x

        def _normalize_ids(ids):
            ids_list = _as_list(ids)
            if not ids_list:
                return []
            out = []
            for v in ids_list:
                # flatten one level if user accidentally nested a list
                if isinstance(v, (list, tuple, set)):
                    for vv in v:
                        out.append(_normalize_id(vv))
                else:
                    out.append(_normalize_id(v))
            return out

        def _normalize_bool(v):
            """Best-effort conversion of YAML-ish truthy values to bool/None."""
            if v is None:
                return None
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, np.integer)):
                if int(v) == 1:
                    return True
                if int(v) == 0:
                    return False
                return bool(v)
            if isinstance(v, str):
                s = v.strip().lower()
                if s in {"true", "t", "yes", "y", "1"}:
                    return True
                if s in {"false", "f", "no", "n", "0"}:
                    return False
                return bool(s)
            return bool(v)

        def _read_block(geos_spec: dict, key: str):
            """Return (all_ids, ids_list, block_present)."""
            if not isinstance(geos_spec, dict) or key not in geos_spec:
                return None, [], False
            block = geos_spec.get(key)
            if block is None:
                return None, [], True
            if isinstance(block, Config):
                all_ids = _normalize_bool(block.get_raw("all_ids", None))
                ids_raw = block.get_raw("ids", None)
            elif isinstance(block, dict):
                all_ids = _normalize_bool(block.get("all_ids", None))
                ids_raw = block.get("ids", None)
            else:
                # Unknown structure; treat as absent.
                return None, [], False

            all_ids = _normalize_bool(all_ids)
            ids_list = _normalize_ids(ids_raw) if ids_raw is not None else []
            return all_ids, ids_list, True

        # ------------------------------------------------------------------
        # Read config (no warnings)
        # ------------------------------------------------------------------
        try:
            apply_cross_level = bool(scenario_config.get_raw("apply_cross_level", True))
            geos_spec_raw = scenario_config.get_raw("geos_to_synthesize", None)

            region_ids: list = []

            if isinstance(geos_spec_raw, dict):
                r_all, r_ids, r_present = _read_block(geos_spec_raw, "region")
                g_all, g_ids, g_present = _read_block(geos_spec_raw, "geo")

                if apply_cross_level:
                    # Multi-level: region ids drive weighting and synthesis.
                    # Professional/safe behavior:
                    # - If the user provided an ids list, it ALWAYS takes precedence.
                    # - If all_ids is True AND ids is non-empty, warn (ambiguous) but still honor ids.
                    # - If all_ids is False AND ids is empty, raise (explicit-but-empty selection).
                    if r_ids:
                        if r_all is True:
                            logging.warning(
                                "Ambiguous geos_to_synthesize.region: both ids and all_ids=True were provided. "
                                "Honoring ids and ignoring all_ids to avoid an accidental full run."
                            )
                        region_ids = r_ids
                    elif r_all is True or r_all is None:
                        region_ids = list(self.region_ids_all)
                    elif r_all is False:
                        raise ValueError(
                            "geos_to_synthesize.region.all_ids is False but ids is empty. "
                            "Provide ids or set all_ids: True."
                        )
                    else:
                        region_ids = list(self.region_ids_all)

                    # If the user only provided a geo block in multi-level mode, be explicit.
                    if (not r_present) and g_present and (g_ids or g_all is False):
                        logging.warning(
                            "Multi-level scenario (apply_cross_level=True) ignores geos_to_synthesize.geo. "
                            "Use geos_to_synthesize.region to select regions."
                        )
                else:
                    # Single-level: region == geo. Accept either block, prefer geo (more intuitive).
                    chosen_all = None
                    chosen_ids: list = []
                    chosen_level = None
                    if g_present:
                        chosen_all, chosen_ids = g_all, g_ids
                        chosen_level = "geo"
                    elif r_present:
                        chosen_all, chosen_ids = r_all, r_ids
                        chosen_level = "region"

                    if chosen_ids:
                        if chosen_all is True:
                            logging.warning(
                                f"Ambiguous geos_to_synthesize.{chosen_level}: both ids and all_ids=True were provided. "
                                "Honoring ids and ignoring all_ids to avoid an accidental full run."
                            )
                        region_ids = chosen_ids
                    elif chosen_all is True or chosen_all is None:
                        region_ids = list(self.region_ids_all) if self.region_ids_all else list(self.geo_ids_all)
                    elif chosen_all is False:
                        raise ValueError(
                            "Single-level scenario: geos_to_synthesize.<level>.all_ids is False but ids is empty. "
                            "Provide ids or set all_ids: True."
                        )
                    else:
                        region_ids = list(self.region_ids_all) if self.region_ids_all else list(self.geo_ids_all)
            else:
                # No geos_to_synthesize block -> ALL
                region_ids = list(self.region_ids_all) if self.region_ids_all else list(self.geo_ids_all)

            # Validate user-provided ids (when we have an "all" universe)
            if region_ids and self.region_ids_all:
                universe = set(self.region_ids_all)
                missing = [rid for rid in region_ids if rid not in universe]
                if missing:
                    raise KeyError(
                        "The following region ids were requested in geos_to_synthesize but were not found in the project's region ids: "
                        f"{missing[:20]}" + (" ..." if len(missing) > 20 else "")
                    )

            self.region_ids = region_ids

            # Resolve geo ids.
            self.geo_ids = []
            region_to_geo = self.geo.get("region_to_geo") if isinstance(self.geo, dict) else None
            if region_to_geo is not None and hasattr(region_to_geo, "empty") and (not region_to_geo.empty):
                for region_id in self.region_ids:
                    geo_list = self.get_geo_ids_for_region(region_id)
                    if geo_list:
                        self.geo_ids.extend(list(geo_list))
            else:
                # No mapping: region ids are geo ids.
                logging.info("No region_to_geo mapping. Treating region ids as geo ids.")
                self.geo_ids = list(self.region_ids)

            # Final fallback
            if (not self.geo_ids) and self.geo_ids_all:
                self.geo_ids = list(self.geo_ids_all)
            if (not self.region_ids) and self.region_ids_all:
                self.region_ids = list(self.region_ids_all)

            # Helpful summary (kept short to avoid flooding logs).
            def _preview(ids, max_n: int = 10):
                ids = list(ids) if ids else []
                if len(ids) <= max_n:
                    return ids
                return ids[:max_n] + ["..."]

            logging.info(
                f"Scenario geography selection resolved: "
                f"{len(self.region_ids)} region ids, {len(self.geo_ids)} geo ids."
            )
            logging.info(f"  Region ids (preview): {_preview(self.region_ids)}")
            logging.info(f"  Geo ids (preview):    {_preview(self.geo_ids)}")

        except Exception as e:
            logging.warning(
                f"Failed to enumerate geo/region ids for scenario due to {e}. Defaulting to all."
            )
            self.geo_ids = list(self.geo_ids_all) if self.geo_ids_all else []
            self.region_ids = list(self.region_ids_all) if self.region_ids_all else []

    def return_variables_cats(self, entity, variable_names):
        """Returns unique categories for each variable in an entity dataset."""
        return {var: self.return_variable_cats(entity, var) for var in variable_names}

    def return_variable_cats(self, entity, variable_name):
        """Returns unique values for a specific variable in an entity dataset."""
        return np.unique(self.sample[entity][variable_name].values).tolist()

    def check_data(self):
        """Runs data consistency checks.

        PopGen's original code referenced two checks (sample-vs-marginals
        alignment and general marginal sanity checks) but did not ship concrete
        implementations. We keep these hooks so power-users can override/extend
        them, while providing safe default implementations.
        """
        self.check_sample_marginals_consistency()
        self.check_marginals()

    def check_sample_marginals_consistency(self):
        """Lightweight consistency checks between sample data and marginals.

        This is intentionally conservative: we avoid enforcing strict rules
        because PopGen is used with many data conventions. The goal is to catch
        obvious issues early and provide actionable diagnostics.
        """
        # Nothing to do if data isn't loaded.
        if not getattr(self, "sample", None) or not getattr(self, "geo_marginals", None):
            return

        # If a marginal file is present for an entity but the corresponding
        # sample table is missing, that will fail later in a less clear way.
        for entity, m in self.geo_marginals.items():
            if m is None:
                continue
            if entity not in self.sample:
                logging.warning(
                    "Marginals provided for entity '%s' but no matching sample table was loaded.",
                    entity,
                )

    def check_marginals(self):
        """Basic marginal sanity checks."""
        # Ensure marginal tables have an index.
        for scope_name, scope in ("geo", getattr(self, "geo_marginals", {})), ("region", getattr(self, "region_marginals", {})):
            if not isinstance(scope, dict):
                continue
            for entity, df in scope.items():
                if df is None:
                    continue
                if df.index is None:
                    logging.warning("%s marginals for entity '%s' have no index.", scope_name, entity)
                if df.empty:
                    logging.warning("%s marginals for entity '%s' are empty.", scope_name, entity)

    def check(self):
        """Placeholder for additional data consistency checks."""
        pass
