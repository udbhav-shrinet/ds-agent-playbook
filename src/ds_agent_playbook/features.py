from typing import List, Optional
import polars as pl
from ds_agent_playbook.schemas import FeatureCandidate

class FeatureSynthesizer:
    """Automated feature hypothesis generation and vector extraction."""

    def __init__(self, df: pl.DataFrame, target_col: Optional[str] = None):
        self.df = df
        self.target_col = target_col

    def propose_features(self) -> List[FeatureCandidate]:
        candidates: List[FeatureCandidate] = []
        cols = [c for c in self.df.columns if c != self.target_col]

        num_cols = [c for c in cols if self.df[c].dtype.is_numeric()]
        cat_cols = [c for c in cols if self.df[c].dtype in [pl.Utf8, pl.Categorical]]

        # Interaction / Ratio hypotheses for numerical features
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                # Ratio feature
                candidates.append(
                    FeatureCandidate(
                        feature_name=f"{c1}_to_{c2}_ratio",
                        source_columns=[c1, c2],
                        transformation=f"{c1} / ({c2} + 1e-6)",
                        hypothesis=f"Normalized relative magnitude between {c1} and {c2}.",
                        expected_signal_boost="High for scale-invariant relationships",
                    )
                )

        # Categorical aggregation hypotheses
        for cat in cat_cols:
            for num in num_cols:
                candidates.append(
                    FeatureCandidate(
                        feature_name=f"{num}_mean_by_{cat}",
                        source_columns=[cat, num],
                        transformation=f"GROUP BY {cat} -> MEAN({num})",
                        hypothesis=f"Target group behavior differences captured across {cat}.",
                        expected_signal_boost="High for hierarchical variance",
                    )
                )

        return candidates[:8]  # Top candidate hypotheses

    def materialize_interactions(self, df: pl.DataFrame) -> pl.DataFrame:
        candidates = self.propose_features()
        res = df.clone()
        for cand in candidates:
            if "ratio" in cand.feature_name and len(cand.source_columns) == 2:
                c1, c2 = cand.source_columns
                res = res.with_columns(
                    (pl.col(c1) / (pl.col(c2) + 1e-6)).alias(cand.feature_name)
                )
        return res
