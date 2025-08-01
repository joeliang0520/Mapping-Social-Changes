# Data and Code Release for the Paper "Mapping Scoial Change: A Unified Framework for Temporal Clustering

The repository contains the data and code used in the paper "Mapping Social Change: A Unified Framework for Temporal Clustering". 

## File Structure:

`tscluster`: Contains the `tscluster` package, which includes all temporal clustering methods used in the paper. This package is also available on PyPI.

`helper.py`: Contains helper functions used in the analysis and visualization of the case studies.

`case_studies.ipynb`: Contains all the instructions to load and call all the codes and analysis for figures, tables, and results in the manuscript and supplementary materials. Formated as a Jupyter Notebook with Python.

To run the notebook, you need to install the required packages listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

You also need a valid API key for the Gurobi MILP solver, which is used for the optimization tasks in the notebook. You can obtain a free academic license from [Gurobi's website](https://www.gurobi.com/academia/academic-program-and-licenses/).

Cleaned data and Raw data used in each of the case studies are provided in the `data` folder:
- State-level Temporal Polarization:
    - `data/spatiotemporal_polarization_raw.csv` for the raw data.
    - `data/spatiotemporal_polarization_cleaned.csv` for the cleaned data.
- Global Cultural Change: World Values Survey:
    - `data/world_value_survey_raw.csv` for the raw data.
    - `data/world_value_survey_cleaned.csv` for the cleaned data.
- Business Development Patterns in Chicago:
    - `data/chicago_business_development_cleaned.csv` for the cleaned data.

Due to the scalability of the Chicago Business Development Patterns case study which takes a long time to run the MILP formulation in provided notebook, we also provided the pre-run cluster labels and cluster centers in the `pre-run results` folder:
- `pre-run results/cluster_center_chicago_business_development.npy` for the pre-run cluster centers
- `pre-run results/cluster_label_chicago_business_development.npy` for the pre-run cluster labels.

We provided codes and instructions to load these pre-run cluster labels and centers in the `case_studies.ipynb` notebook.

`chicago_spatial_map.geojson` and folder `us_state_spatial_map` contains the necessary geographic data to plot the spatial maps in the paper.