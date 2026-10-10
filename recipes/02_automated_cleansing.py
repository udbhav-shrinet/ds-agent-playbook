"""Recipe 02: Deterministic Data Cleansing with DuckDB & Polars.

Run with: uv run python recipes/02_automated_cleansing.py
"""

import polars as pl
from ds_agent_playbook.profiler import ProfilerEngine
from ds_agent_playbook.cleaner import CleanerAgent

def run_cleansing_recipe():
    print("--- Running Recipe 02: Deterministic Cleansing ---")
    data_path = "data/sample_transactions.csv"
    
    df = pl.read_csv(data_path)
    profiler = ProfilerEngine(df)
    summary = profiler.profile()

    cleaner = CleanerAgent(summary)
    plan = cleaner.devise_plan()

    print(f"Formulated {len(plan)} automated cleansing actions:")
    for step, act in enumerate(plan, 1):
        print(f"  Step {step}: [{act.action_type}] on '{act.column}' -> {act.rationale}")

    cleaned_df = cleaner.apply_plan(df, plan)
    print(f"\nBefore: {df.height} rows, {df.width} cols")
    print(f"After:  {cleaned_df.height} rows, {cleaned_df.width} cols")

if __name__ == "__main__":
    run_cleansing_recipe()
