import pytest
import polars as pl
from ds_agent_playbook.profiler import ProfilerEngine
from ds_agent_playbook.cleaner import CleanerAgent
from ds_agent_playbook.features import FeatureSynthesizer

def test_profiler_and_cleaner_pipeline():
    # Setup test dataframe with intentional anomalies
    data = {
        "id": ["1", "2", "3", "4", "5"],
        "val": [10.0, None, 30.0, 40.0, 50.0],
        "constant": [1, 1, 1, 1, 1],
        "category": ["A", "A", "B", None, "A"],
    }
    df = pl.DataFrame(data)

    profiler = ProfilerEngine(df)
    summary = profiler.profile()

    assert summary.row_count == 5
    assert summary.col_count == 4

    cleaner = CleanerAgent(summary)
    plan = cleaner.devise_plan()

    assert any(a.action_type == "drop_column" and a.column == "constant" for a in plan)
    assert any(a.action_type == "impute" and a.column == "val" for a in plan)

    cleaned_df = cleaner.apply_plan(df, plan)
    assert "constant" not in cleaned_df.columns
    assert cleaned_df["val"].null_count() == 0

def test_feature_synthesizer():
    data = {
        "price": [100.0, 200.0, 300.0],
        "units": [2, 4, 5],
        "category": ["X", "Y", "X"],
    }
    df = pl.DataFrame(data)
    synth = FeatureSynthesizer(df)
    hypotheses = synth.propose_features()
    
    assert len(hypotheses) > 0
    ratio_names = [h.feature_name for h in hypotheses if "ratio" in h.feature_name]
    assert "price_to_units_ratio" in ratio_names
