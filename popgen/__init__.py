"""PopGen3 package.

This package implements a population synthesis pipeline:

* IPF (iterative proportional fitting) to build multiway constraints from
  marginals.
* Reweighting (IPU or entropy) to compute sample household weights.
* Optional drawing of an integerized synthetic population.
"""

from .project import popgen_run
from .errors import ConfigurationError, InputValidationError, PopGenError, ScenarioError

__version__ = '3.0.6'


def run(config_path: str, strict=None) -> None:
    """Run PopGen from Python.

    Raises :class:`popgen.ScenarioError` if any scenario fails (all scenarios
    are attempted first).  ``strict=True`` stops a scenario before IPF when the
    input checks find errors; the default (None) follows ``validation.strict``
    in the YAML, which defaults to False.
    """
    popgen_run(config_path, strict=strict)
