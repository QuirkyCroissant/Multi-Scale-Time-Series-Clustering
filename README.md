# Multi Scale Time Series Clustering

> Time-series data is a collection of data points recorded over time, each associated with a specific timestamp. This form of data is prevalent in various fields such as finance, economics, meteorology, healthcare, energy, telecommunications, and transportation. Current algorithms assume that we only have time-series data of the same scaling, but in real-world data time-series often consists of different scalings, e.g. hourly, daily, or weekly weather forecasts. This project will mainly focus on the development of a clustering algorithm that can handle time series with different scalings. 

## Overview

This repository contains code and documentation for my bachelor thesis on clustering time-series data with different scalings. The project focuses on developing clustering algorithms that are robust to the varying temporal resolutions found in real-world data (e.g., hourly, daily, weekly).

## Objectives

- **Develop clustering algorithms:** Create/apply methods that effectively cluster time-series data with different scalings.
- **Evaluate performance:** Test and validate the algorithms on various datasets.
- **(Optional Extension):** Generate graphs from time-series data, apply clustering algorithms to the graphs, and compare the results with the time series clustering using similarity measures.


## Project Structure

```
.
├│ data/                  		# Sample datasets or links to data sources
│   ├── ts_demo_data_clean.csv  # Synthetical demo data (prototype mode)
│   ├── ts_demo_data_corrupted.csv  	# modified faulty data (prototype mode)
│   ├── restored/  				# restored data by multiple means(e.g.: interpolation)
│      ├── ts_demo_data_<method>.csv  	# restored data through interpolation method
│      ├── ... 
├│ docs/                  		# Documentation and thesis drafts
├│ notebooks/					# Jupyter notebooks for exploratory analysis
├│ src/							# Source code (algorithms, utility functions)
│   ├── config.py  				# Stores essential parameters and constants
│   ├── data_corruption.py  	# Module for synthetic dataset corruption
│   ├── data_generation.py  	# Module for synthetic dataset generation
│   ├── data_restoration.py  	# Module for the restoration of data through various means
│   ├── main.py					# Main script with mode selection
│   ├── project_utilities.py	# helper utilities for the project
├│ experiments/             	# Scripts and logs from experimental runs
│   ├── distance_matrices		# exported dissimilarity/distance matrices used for clustering
│   ├── logs					# log files from various experimental operations
│      ├── interpolations
│   ├── plots					# plot diagrams from various experimental operations
│      ├── interpolations
└│ README.md                	# Project overview and instructions
```


## Installation/Usage

1. **Clone the repository:**

    ```bash
		git clone https://github.com/QuirkyCroissant/Multi-Scale-Time-Series-Clustering
	```
	
	
2. **Create and activate a virtual environment:**

    ```bash
		python -m venv env
		source env/bin/activate   
	```
	
3. **Install dependencies:**

    ```bash
		pip install -r requirements.txt
	```

4. **Run the project:**

    Run the application by providing the compiler with either the prototype or the final implementation flags:
    ```bash
		python src/main.py --mode <demo / prod>
	```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact & Acknowledgements

- **Supervisor:** [Ass.-Prof. Dott.ssa Dott.ssa.mag.Yllka Velaj, PhD](mailto:yllka.velaj@univie.ac.at)
- **Student:** Florian Hajek

Thank you to everyone who contributed to this project!