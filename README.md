# PopGen3

## Introduction

**PopGen** is a state-of-the-art open-source synthetic population generator for advanced travel demand modeling. Developed under the guidance of Professor Ram M. Pendyala at Arizona State University, PopGen utilizes both heuristic algorithms (IPU) and entropy-based methods to control and match both household-level and person-level attribute distributions, synthesizing realistic population data.

## Key Features

- Advanced control of household and person variables across multiple geographic resolutions (region and geo).
- Command-line interface for simplified usage and enhanced computational efficiency.
- Fully compatible with Python 3, with updated dependencies.
  
## Previous Version
For insights into the earlier versions of PopGen, please refer to the [legacy PopGen webpage](https://www.mobilityanalytics.org/popgen.html). This resource provides access to previous versions, detailed descriptions of software objectives, and paper references.

## Run PopGen3 Online
- **Binder**: A cloud-based Jupyter notebook with step-by-step tutorials for data input understanding and output visualization.

   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tomnetutc/PopGen3/main?filepath=popgen_learning%2FLearning_materials.ipynb)

- **Colab**: Provides a cloud-based environment for executing PopGen3 code and downloading result packages.

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1j1Stb8IA8OfaoPRh232kId8hqi3dUtur?usp=sharing)


## Contents

### Data Inputs

#### Survey sample data:
  `household_sample.csv`       
  `person_sample.csv`

#### Population marginal data:

- **geo-level marginals:**  
     `person_marginals.csv`  
     `household_marginals.csv`  

- **region-level marginals:**  
  `region_person_marginals.csv`  
  `region_household_marginals.csv`  

#### Multi-Geographic Resolution Mapping Data:

  `region_geo_mapping.csv`  
  `geo_sample_mapping.csv`

Example of Multi-Geographic Resolution Levels:
- <small><i>Region Level:</i> Set as census county subdivision</small>
- <small><i>Geo Level:</i> Set as census tracts</small>
- <small><i>Sample Geo Level:</i> Set as Public Use Microdata Areas (PUMAs)</small>

### Configuration File 

The `configuration.yaml` file contains several key sections for the PopGen setup. 

#### Designed Input Keys

| **Project Wide Setting** | **Input Data Files** | **Scenario Settings** |
| --- | --- | --- |
| - synthesize<br>- name<br>- location | - entities<br>- column_names<br>- location | - description<br>- apply_region_controls<br>- control_variables<br>- parameters<br>- geos_to_synthesize<br>- outputs |

Entity names are free (`person`, `adult`, ...). When household and person controls cannot all be met at once, IPU gives priority to the entities listed under `housing_entities`: their totals are matched exactly and the person totals as closely as possible (Ye et al., 2009). Since 3.0.5 this does not depend on how the entities are named.

Each entity can have any number of control variables, and a household-only configuration can draw a synthetic population (`person_entities` may be empty; only `housing_synthetic.csv` is then written). Before 3.0.6 several such layouts failed (see `CHANGELOG.md`).

### Results Outputs

#### Sample Weighting Results:

`weights.csv`
#### Generated Synthetic Population:
`housing_synthetic.csv`                                
`person_synthetic.csv`

#### Diagnostics (new in 3.0.5):
`reweighting_fit_summary.csv` - for every geography and entity, how closely the weighted sample matches the control totals (`margin_*` columns) and the IPF multiway cells (`cell_*` columns). It is written wherever `reweighting_average_diffs.csv` is (always in weights-only runs; with `performance: [reweighting]` otherwise). Judge the fit by the `margin_*` columns: at region level the cell targets come from a separate IPF on the region seed, which IPU does not finish on, so large region-level `cell_*` values are expected.

### Input checks and errors (new in 3.0.5)
Before IPF, PopGen checks the inputs (duplicate household ids, missing values in control variables, categories present in the sample but not in the marginals and vice versa, control totals that disagree, negative targets and zero targets for categories with sample records, geographies without sample records) and logs what it finds. The checks never change results. To stop a scenario when they find errors, add to the scenario (or project):

```yaml
validation:
  strict: true
```

or run `popgen --strict config.yaml` / `popgen.run("config.yaml", strict=True)`.

A run that is certain to fail at its end stops right away, strict or not: with `synthesize: true`, a sample column named like the geo column would make writing the synthetic population fail, so the scenario stops before IPF and says which column to rename.

If a scenario fails, `popgen.run` now raises `popgen.ScenarioError` (after attempting all scenarios) and the `popgen` command exits with status 1; earlier versions only logged the error.

## Testing
```
pip install -e ".[test]"
pytest                                   # golden-output, accuracy, validation and CLI tests
POPGEN_DIFFERENTIAL=1 pytest             # also compare every output with popgen3 3.0.4 and 3.0.5 from PyPI
```
The golden tests require the outputs to stay byte-for-byte identical to release 3.0.4 (and, for configurations that only work since 3.0.6, to what 3.0.5 produced in a working layout, with the weights compared to a relative tolerance of 1e-9 because their last decimal can differ between machines); see `CHANGELOG.md` for the history of numerical changes between releases.
