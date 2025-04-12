# Change Log

All notable changes to this project will be documented in this file.

---

## prototype-v3 - Upcoming

### Added
- Graph transformation: time series to node mapping using Pearson correlation dissimilarity
- Visualization module for graph structures with threshold-based edge sparsification
- Export of graph-based dissimilarity matrix

### Planned
- Graph clustering module (e.g., community detection)
- Comparison of graph-based clusters with DTW/FastDTW-based results

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
