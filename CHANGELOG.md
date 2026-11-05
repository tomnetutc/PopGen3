# Changelog

## 3.0.5 (2026-11-05)

**Compatibility promise:** for every input that 3.0.4 handled correctly, the
outputs (weights, synthetic population, summaries, multiway tables, performance
files) are byte-for-byte identical to 3.0.4. Verified on the 19 THA Wave 2
weighting runs, the 9 THA Wave 1 weighting/synthesis runs and the four
Arizona example configurations (86 runs on Python 3.9-3.13 with pandas 1.3 to
3.0), and with the test suite on x86-64 and ARM64 (aarch64), up to pandas 3.1.
The one deliberate change of results concerns configurations whose entity
names sort the person entity before the housing entities (see Fixed).

### Fixed
- **pandas 3.0:** 3.0.4 failed in IPF with "assignment destination is read-only"
  (copy-on-write) and, because errors were swallowed, finished without output
  and without raising. Also removed a `pd.concat(copy=...)` argument that
  pandas 3 deprecates and pandas 4 removes.
- **Single-level geography selection:** with `apply_cross_level: false`, ids
  written under `geos_to_synthesize.region` were ignored whenever the geo block
  held the template value `ids: [] / all_ids: True`, and every geography was
  run (configs written for 3.0.1/3.0.2, such as the THA Wave 1 weighting,
  failed this way). An explicit `ids` list now always wins, geo block first.
- An explicit but invalid selection (unknown ids, or `all_ids: false` with no
  ids) now fails the scenario. 3.0.4 caught its own error and silently ran
  every geography.
- **Failures are reported:** if a scenario fails, `popgen.run` raises
  `popgen.ScenarioError` after attempting all scenarios, and the `popgen`
  command exits with status 1. 3.0.4 only logged the error and exited 0.
- Drawing could raise `IndexError` when a random number fell above the last
  cumulative probability because of rounding; such draws are now clipped.
  Draws that worked before are unchanged.
- **IPU sweep order no longer depends on entity names.** 3.0.3/3.0.4 sweep the
  constraint cells in reverse alphabetical order, so the housing cells are
  matched last only because "person" sorts after "household" and
  "groupquarter". With the person entity named e.g. `adult`, 3.0.4 swept the
  person cells last and, in the Arizona example, left household totals 3.8 %
  off on average (11 % max) and group-quarter totals 5 % off - the 3.0.2
  failure again, silently. The cells of `housing_entities` are now always swept
  after the person cells. With the usual names the order is exactly that of
  3.0.4 (all outputs identical); with `adult` the weights and the synthetic
  population are now identical to the run with `person`.

### Added
- Input checks before IPF (`popgen/validation.py`): duplicate household ids,
  persons without a household, missing values in control variables, sample
  categories absent from the marginals and vice versa, control totals that
  disagree across variables, selected geographies without marginals, mapping
  rows or sample records. Findings are logged; the checks never modify data.
  `validation: {strict: true}` (scenario or project), `popgen --strict` or
  `popgen.run(..., strict=True)` stop the scenario on errors.
- `reweighting_fit_summary.csv`: per geography and entity, the fit of the
  weighted sample to the control totals (`margin_*`) and to the IPF multiway
  cells (`cell_*`); a one-line summary is logged per region. Deviations are
  rounded to 1e-10 so that the file does not change between machines (the
  last digits of floating-point sums depend on the BLAS build and CPU).
- `reweighting.early_stop: true` (opt-in): stop the IPU loop once every
  geography's mean cell deviation is below `reweighting.tolerance`. Off by
  default, so results are unchanged.
- A warning when reweighting has no geo-level constraints (the loop then makes
  a single IPU pass - behaviour unchanged).
- `popgen --version`; `popgen.PopGenError`, `ScenarioError`,
  `ConfigurationError`, `InputValidationError`.
- Drawing uses its own `RandomState`, seeded exactly as before: the synthetic
  population is identical, but NumPy's global random state is no longer reset.
- Test suite (`tests/`): golden outputs from 3.0.4 (line-ending independent,
  so it also passes on Windows), accuracy thresholds, reproduction of the IPU
  worked example of Ye et al. (2009, Table 1), entity-name independence,
  geography selection, failure handling, input checks, and an opt-in
  byte-for-byte comparison with the PyPI 3.0.4 wheel. GitHub Actions CI on
  Linux (Python 3.8-3.13, pandas 1.3-3.x) and Windows.

### Packaging
- Metadata moved to `pyproject.toml` (PEP 621); `setup.py` removed.
- `pandas<4`: pandas 3.1 deprecates the `inplace=` argument of `set_index`,
  `sort_values`, `reset_index` and similar methods (PDEP-8), which PopGen still
  uses, and pandas 4 will remove it. The cap keeps a fresh `pip install` from
  pairing 3.0.5 with a pandas it cannot run on - the way 3.0.4 met pandas 3.
  On pandas 3.1 the deprecation warnings are harmless (results identical).
- Project URLs point to github.com/tomnetutc/PopGen3.
- Fixed the Binder link in the README.
- Example run outputs and notebook checkpoints are no longer tracked in git;
  run output folders are ignored.

### Known issues (unchanged; planned for 3.1)
- The scenario YAML written to each output folder contains Python tags
  (`!!python/object:popgen.config.Config`) and cannot be read with
  `yaml.safe_load`.
- Sample files are read with pandas' default missing-value markers, so survey
  answers such as "None" or "NA" become missing (pandas >= 2 also treats
  "None") and appear as 0 in the synthetic population.
- Two-level runs: IPU cannot match household and person totals exactly for
  the Arizona example (person totals about 2.7 % mean / 5 % max deviation);
  more iterations do not reduce it. See `reweighting_fit_summary.csv`.
- Reweighting without geo-level constraints makes a single IPU pass.
- Input DataFrames are modified in place by the pipeline.
- `reweighting.procedure: entropy` is not covered by the regression tests (no
  bundled example uses it); it sweeps the cells in forward order as before.

## History of numerical changes, 3.0.1 - 3.0.4

Reconstructed in November 2026 by running every released wheel on the same inputs.

- **3.0.4** (2026-01-10): IPU sweeps the constraints in reverse order again,
  as 3.0.3 and earlier did, fixing the two-level regression of 3.0.1/3.0.2
  (for entity names that sort the person entity after the housing entities;
  made name-independent in 3.0.5).
  IPF stops when the average deviation itself is below tolerance (previously:
  when it changed by less than tolerance between iterations), and the deviation
  is averaged over the variables actually checked. Single-level mode
  (`apply_cross_level: false`) and weights-only runs (`synthesize: false`)
  reworked; `geos_to_synthesize` validated.
- **3.0.3** (2025-04-29): return to the pre-refactor code line (reverse IPU
  sweep). No single-level support; `outputs.multiway` is required.
- **3.0.1 / 3.0.2** (2025-04-27 / 2025-04-29): refactored code line. IPU swept
  the constraints in forward order; in two-level runs this left household
  totals about 4.5 % off on average (14 % max) and group-quarter totals about
  6 % off in the Arizona example, whereas 3.0.3/3.0.4 match them to 0.02 %.
  Single-level runs were barely affected (weights within 0.7 % on THA Wave 1).
  `synthesize` was read only at project level, so a scenario-level
  `synthesize: true` produced weights only.
