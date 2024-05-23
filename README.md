# PopGen3

## Introduction

**PopGen** is a state-of-the-art open-source synthetic population generator for advanced travel demand modeling. Developed under the guidance of Professor Ram M. Pendyala at Arizona State University, PopGen utilizes a heuristic algorithm (IPU) to control and match both household-level and person-level attribute distributions.

## Key Features

- Advanced control of household and person variables across multiple geographic resolutions (region and geo).
- Command-line interface for simplified usage and enhanced computational efficiency.
- Fully compatible with Python 3, with updated dependencies.
  
## Previous Version
For insights into the earlier versions of PopGen, please refer to the [legacy PopGen webpage](https://www.mobilityanalytics.org/popgen.html). This resource provides access to previous versions, detailed descriptions of software objectives, and paper references.

## Run PopGen3 Online
- **Binder**: A cloud-based Jupyter notebook with step-by-step tutorials on data input understanding and output visualization.

   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tomnetutc/PopGen3/main?filepath=popgen_learning%2Fpopgen_learning.ipynb)

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

Mapping files ensure accurate mapping and alignment of different data files with different resolutions. 

Example of survey and marginal data collected and labeled with different geographic resolution levels:

- **Region level**: Set as census county subdivision
- **Geo level**: Set as census tracts
- **Sample Geo level**: Set as Public Use Microdata Areas (PUMAs)



