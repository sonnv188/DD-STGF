# DD-STGF: Dual-Domain Spatiotemporal Graph Learning

DD-STGF is a deep learning framework for next-day, multi-category crime count forecasting and spatial hotspot prediction under extreme data sparsity. It combines geographical dependencies among spatial cells with adaptive relationships among crime categories to learn complementary representations of historical crime patterns.

## Data

The `data/` directory contains two versions of the processed crime dataset:

| Directory | Description |
|---|---|
| `events_clean/` | Event-level crime records cleaned and standardized from the dataset released by Zhang et al. (2025). These records serve as the basis for spatial and temporal aggregation. |
| `daily_grid_counts/` | Daily crime counts obtained by assigning the cleaned events to **3 km × 3 km** spatial grid cells and aggregating them by **date, grid cell, and crime category**. These data are used for model training and evaluation. |
