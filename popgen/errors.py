"""Exceptions raised by PopGen.

Before 3.0.5 a failing scenario was only logged and ``popgen.run`` returned
normally (the CLI exited with status 0), so a batch of runs could silently miss
outputs.  Failures are now raised after all scenarios have been attempted.
"""


class PopGenError(Exception):
    """Base class for all PopGen errors."""


class ConfigurationError(PopGenError, ValueError):
    """The configuration file asks for something PopGen cannot do."""


class InputValidationError(PopGenError):
    """Input checks found errors and strict validation is enabled."""


class ScenarioError(PopGenError):
    """One or more scenarios failed.  ``failures`` lists (description, exception)."""

    def __init__(self, message, failures=None):
        super().__init__(message)
        self.failures = list(failures or [])
