# Change Log

All notable changes to this project will be documented in this file.

---

## v1.0.0 - Upcoming

### Planned
- Addition of evaluation mode for clustering results (Rand Index, Adjusted Rand, NMI, ARI)
- Automation of experimental runs with flexible override of key config variables via CLI (optional)
- Separation of analysis workflows for synthetic datasets (with ground truth) vs. production datasets (without ground truth)
- Refined export of interpolation evaluation (MAPE, MSE) for synthetic experiments
- Distinction between graph-based and classic clustering workflows

---

## [prototype-v3](https://github.com/QuirkyCroissant/Multi-Scale-Time-Series-Clustering/tree/prototype-v3) - 2025-04-28

### Added
- Graph transformation: time series to node mapping using Pearson similarity
- Graph clustering integration: Louvain, Greedy Modularity Maximation, and Label Propagation (networkx)
- Visualization of graph structures and clusters
- Exported graph-based distance matrices and cluster assignment logs
- Distinction between graph-based and classic clustering workflows
- Extensive log export for clustering metadata (distance measure, clustering method, runtime, etc.)

### Changed
- Unified distance measures: Pearson similarity without (1-p), negative correlations capped at 0
- Improved plot generation and directory structure for cleaner experiment separation

---

## [prototype-v2](https://github.com/QuirkyCroissant/Multi-Scale-Time-Series-Clustering/tree/prototype-v2) - 2025-04-10

### Added
- Finalized full support for multi-series time series clustering
- Synthetic data generator for realistic household profiles
- Clustering pipeline updated for multiple subjects (multi-TS support)
- Distance matrix computation adjusted for pairwise comparison across series
- Visualizations updated to show cluster medoids and colored TS plots

### Changed
- Refactored restoration pipeline to handle multiple time series
- Restructured corruption functions for daily-level deletion logic
- Improved project structure and added per-method export folders for clarity

---

## [prototype-v1](https://github.com/QuirkyCroissant/Multi-Scale-Time-Series-Clustering/tree/prototype-v1) - 2025-03-25

### Added
- Complete pipeline for single time series segmentation (per-day windowing)
- Support for data generation, corruption, restoration, clustering
- Dissimilarity measures (DTW, FastDTW)
- K-Medoids clustering integration
- Logging and plotting utilities
