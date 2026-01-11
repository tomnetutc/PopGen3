"""Configuration helpers.

The PopGen3 project is driven by a YAML configuration file. We wrap the parsed
YAML object in :class:`Config` so nested keys can be accessed via attribute
notation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Sequence, Union

import yaml


class ConfigError(Exception):
    pass


def wrap_config_value(value):
    """Wrap YAML elements as Config objects to allow attribute access.

    Example:
    If YAML defines:
        attribute1:
            attribute2: 'Value'

    Then, x.attribute1.attribute2 can be used to access "Value".
    """
    # NOTE: A key goal of PopGen's Config wrapper is to be resilient to
    # partially-specified YAML. In many places the code does
    # `control_variables[level][entity].return_list()`. For missing keys we
    # therefore want the *indexing* operator ([]) to return an "empty config"
    # rather than raising. Historically this was achieved by returning
    # `Config(None)` which behaved like an empty dict.
    #
    # However, we must ALSO preserve YAML booleans. In Python, `bool` is a
    # subclass of `int`; naive "numeric" checks can accidentally convert
    # `false/true` into `0/1` and later break logic.

    if value is None:
        # Treat missing/null values as an empty config so chained calls like
        # `x["missing"].return_list()` are safe.
        return Config({})

    # Avoid double-wrapping Config objects (can create Config-of-Config nesting).
    if isinstance(value, Config):
        return value
    # Preserve primitive scalars.
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value

    # Wrap lists/tuples/dicts in Config so downstream code can use .return_list()
    # and nested attribute access.
    return Config(value)


class Config:
    """Config class for handling YAML configuration as attribute-accessible objects."""

    DEFAULT_PARAMETERS = {
        "ipf": {
            "tolerance": 0.0001,
            "iterations": 250,
            "zero_marginal_correction": 0.00001,
            "rounding_procedure": "bucket",
            "archive_performance_frequency": 1
        },
        "reweighting": {
            "procedure": "ipu",
            "tolerance": 0.0001,
            "inner_iterations": 1,
            "outer_iterations": 50,
            "archive_performance_frequency": 1
        },
        "draws": {
            "pvalue_tolerance": 0.9999,
            "iterations": 25,
            "seed": 0
        }
    }

    def __init__(self, data: Any):
        # Keep the original type (dict OR list). Only coerce None to empty dict.
        self._data = {} if data is None else data

    def __setitem__(self, key, value):
        self._data[key] = value

    def __setattr__(self, key, value):
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __getattr__(self, key):
        """Attribute-style access for dict-like sections.

        This wrapper is intentionally forgiving for missing YAML keys. However,
        third-party libraries (notably pandas) sometimes probe objects for
        private attributes like `_typ`. If this Config wraps a list, the old
        behaviour could raise a TypeError. We treat such probes as normal
        attribute misses."""
        # Let Python/pandas/internal attribute probes fall back to their defaults.
        if key.startswith('_'):
            raise AttributeError(key)
        # Attribute access only makes sense for dict-like containers.
        if not isinstance(self._data, Mapping):
            raise AttributeError(key)
        value = self.return_value(key)
        return wrap_config_value(value) if value is not None else None


    def __getitem__(self, key):
        value = self.return_value(key)
        return wrap_config_value(value)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over list-like config sections.

        This supports YAML like:

        project:
          scenario:
            - description: "A"
            - description: "B"
        """
        # If the underlying data is a sequence, provide iteration.
        if isinstance(self._data, Sequence) and not isinstance(self._data, (str, bytes, bytearray)):
            for i in range(len(self._data)):
                yield self[i]
        else:
            # For dict-like sections, iterate keys (matching Python dict behaviour).
            for k in self._data:
                yield k

    def __getstate__(self):
        """Ensure `yaml.dump()` works correctly with Config objects."""
        return self.__dict__

    def write_to_file(self, filepath):
        with open(filepath, 'w') as file:
            yaml.dump(self._data, file, default_flow_style=False)

    def return_value(self, key):
        """Retrieve value from config data safely."""
        try:
            return self._data[key]
        except KeyError:
            logging.warning(f"Key '{key}' not found in configuration.")
            return None  # Return None instead of raising an error
        except TypeError:
            # e.g. attempting dict access on a non-dict container
            logging.error(
                f"Invalid configuration structure when accessing key '{key}'. "
                f"Container type: {type(self._data)}"
            )
            raise

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return repr(self._data)

    # ---------------------------------------------------------------------
    # Convenience helpers (do NOT log warnings)
    # ---------------------------------------------------------------------
    def has_key(self, key: str) -> bool:
        """Return True if this config section is dict-like and contains *key*.

        This is a quiet check (no warnings). Use it when optional keys are
        genuinely optional and you don't want to spam logs.
        """
        return isinstance(self._data, Mapping) and key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style get with wrapping, without warnings."""
        if isinstance(self._data, Mapping) and key in self._data:
            return wrap_config_value(self._data[key])
        return default

    def get_raw(self, key: str, default: Any = None) -> Any:
        """Return the raw underlying YAML value (no wrapping, no warnings)."""
        if isinstance(self._data, Mapping) and key in self._data:
            return self._data[key]
        return default

    def return_list(self):
        """Return a list of top-level keys in the configuration."""
        if isinstance(self._data, Mapping):
            return list(self._data.keys())
        if isinstance(self._data, Sequence) and not isinstance(self._data, (str, bytes, bytearray)):
            return list(self._data)
        return [self._data]

    def return_dict(self):
        """Convert nested Config objects to pure dictionaries."""

        def convert(value):
            if isinstance(value, Config):
                return value.return_dict()
            elif isinstance(value, list):
                return [convert(v) for v in value]
            elif isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            return value

        return convert(self._data)

    def write_to_open(self, filepath):
        with open(filepath, 'w') as outfile:
            yaml.dump(self._data, outfile, default_flow_style=False)