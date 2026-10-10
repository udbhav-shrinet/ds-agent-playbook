"""Recipe 01: Automated Exploratory Data Analysis & Statistical Diagnostics.

Run with: uv run python recipes/01_automated_eda.py
"""

import polars as pl
from ds_agent_playbook.profiler import ProfilerEngine

def run_eda_recipe():
    print("--- Running Recipe 01: Automated EDA ---")
    data_path = "data/sample_transactions.csv"
    
    profiler = ProfilerEngine.from_csv(data_path)
    summary = profiler.profile()

    print(f"Dataset Dimensions: {summary.row_count} rows x {summary.col_count} columns")
    print(f"Duplicate Rows Identified: {summary.duplicate_rows}")
    
    print("\nColumn Anomalies Detected:")
    for col in summary.columns:
        if col.anomaly_flags:
            print(f"  - [{col.name}] ({col.dtype}): {', '.join(col.anomaly_flags)}")

if __name__ == "__main__":
    run_eda_recipe()
