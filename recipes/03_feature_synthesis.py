"""Recipe 03: Feature Hypothesis Generation & Materialization.

Run with: uv run python recipes/03_feature_synthesis.py
"""

import polars as pl
from ds_agent_playbook.features import FeatureSynthesizer

def run_feature_recipe():
    print("--- Running Recipe 03: Feature Synthesis ---")
    data_path = "data/sample_transactions.csv"
    
    df = pl.read_csv(data_path)
    synth = FeatureSynthesizer(df)
    hypotheses = synth.propose_features()

    print(f"Generated {len(hypotheses)} domain hypotheses:")
    for h in hypotheses:
        print(f"  * Feature: {h.feature_name}")
        print(f"    Formula:    {h.transformation}")
        print(f"    Hypothesis: {h.hypothesis}\n")

    materialized_df = synth.materialize_interactions(df)
    print(f"Original columns: {len(df.columns)}")
    print(f"Augmented columns: {len(materialized_df.columns)}")

if __name__ == "__main__":
    run_feature_recipe()
