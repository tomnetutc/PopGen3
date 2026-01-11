"""PopGen3 package.

This package implements a population synthesis pipeline:

* IPF (iterative proportional fitting) to build multiway constraints from
  marginals.
* Reweighting (IPU or entropy) to compute sample household weights.
* Optional drawing of an integerized synthetic population.
"""

from .project import popgen_run

__version__ = '3.0.4'


def run(config_path: str) -> None:
    """Run PopGen from Python."""
    popgen_run(config_path)