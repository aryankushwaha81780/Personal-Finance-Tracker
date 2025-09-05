# Personal Finance Tracker (NumPy) — MVP

Author: Aryan Kushwaha 
<br>
Repository: [https://github.com/aryankushwaha81780/Personal-Finance-Tracker.git]


This repository contains an MVP Personal Finance Tracker implemented using **NumPy**. It ingests monthly expense data (per-category), produces analytical summaries (per-category totals, monthly totals, averages and variability) and generates short-term forecasts using lightweight numerical techniques (moving average and linear trend). The goal is a concise, reproducible analytics pipeline built on core numeric primitives — suitable for beginners to demonstrate practical NumPy proficiency.

## Problem
Individuals and micro-businesses often lack a lightweight, reproducible way to:
- Aggregate monthly expenses across categories,
- Understand where money is concentrated,
- Detect spikes and seasonality,
- Forecast short-term spending to aid budgeting.

This project provides a minimal, extendable solution that addresses those problems using only NumPy (and Matplotlib for visualization).

## What this repo contains
- `source.py` — core script with:
  - CSV ingestion helper
  - Aggregation utilities (`np.sum`, `np.mean`, `np.std`)
  - Moving-average and linear-trend forecasting
  - Optional plotting functions
- `sample_data.csv` — example input file (format: `Month,Category1,Category2,...`)
- `README.md` — this document

## Key learning outcomes
From implementing and using this project you will:
- Master fundamental NumPy operations for real data (slicing, axis-based aggregation).
- Implement simple forecasting algorithms using NumPy primitives:
  - Rolling moving average (iterative)
  - Linear trend extrapolation with `np.polyfit`
- Understand practical engineering considerations: input validation, forecasting evaluation (MAPE/RMSE), and minimal visualization.
- Build a reproducible analytic pipeline suitable for extension to dashboards or machine learning models.

## How to use
1. Install dependencies:
  ```pip install numpy pandas matplotlib```

2. Run MVP with synthetic data:
  ```python finance_tracker.py```

3. To use your CSV:
- Provide `sample_data.csv` with header `Month, Cat1, Cat2,...`
- Modify the script to call `load_csv("sample_data.csv")` in the main block.

## Data schema
- Header: `Month, CategoryA, CategoryB, ...`
- Rows: `YYYY-MM, <numeric>, <numeric>, ...`
- Example: 2024-01, 8000, 15000, 3000, 4000

## Where it can be used (applications)
- Personal budgeting and household finance tracking.
- Small-business expense monitoring and short-term cash flow planning.
- Preprocessing for financial ML models (feature engineering using NumPy).
- Educational demos for teaching numeric programming and time-series basics.

