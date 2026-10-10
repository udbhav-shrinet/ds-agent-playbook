import polars as pl
from typing import List
from ds_agent_playbook.schemas import ColumnProfile, DatasetSummary

class ProfilerEngine:
    """Fast dataset profiling engine backed by Polars."""

    def __init__(self, df: pl.DataFrame):
        self.df = df

    @classmethod
    def from_csv(cls, path: str) -> "ProfilerEngine":
        df = pl.read_csv(path)
        return cls(df)

    @classmethod
    def from_parquet(cls, path: str) -> "ProfilerEngine":
        df = pl.read_parquet(path)
        return cls(df)

    def profile(self) -> DatasetSummary:
        n_rows = self.df.height
        n_cols = self.df.width
        mem_mb = round(self.df.estimated_size() / (1024 * 1024), 2)
        n_dups = n_rows - self.df.unique().height

        profiles: List[ColumnProfile] = []

        for col_name in self.df.columns:
            s = self.df[col_name]
            nulls = int(s.null_count())
            null_pct = round((nulls / n_rows) * 100, 2) if n_rows > 0 else 0.0
            uniques = int(s.n_unique())

            anomalies = []
            if null_pct > 25.0:
                anomalies.append(f"High missingness ({null_pct}%)")
            if uniques == 1:
                anomalies.append("Constant zero-variance column")
            elif uniques == n_rows and n_rows > 50 and s.dtype == pl.Utf8:
                anomalies.append("Potential high-cardinality ID column")

            mean_val, std_val, min_v, max_v, skew_val = None, None, None, None, None

            if s.dtype.is_numeric():
                mean_val = float(s.mean()) if s.mean() is not None else None
                std_val = float(s.std()) if s.std() is not None else None
                min_v = float(s.min()) if s.min() is not None else None
                max_v = float(s.max()) if s.max() is not None else None

                if s.count() > 5 and std_val and std_val > 0 and mean_val is not None:
                    # Pearson median skewness approximation: 3 * (mean - median) / std
                    median_val = float(s.median()) if s.median() is not None else mean_val
                    skew_val = round(3 * (mean_val - median_val) / std_val, 3)
                    if abs(skew_val) > 1.5:
                        anomalies.append(f"High skewness ({skew_val})")

            profiles.append(
                ColumnProfile(
                    name=col_name,
                    dtype=str(s.dtype),
                    null_count=nulls,
                    null_percentage=null_pct,
                    unique_count=uniques,
                    mean=round(mean_val, 3) if mean_val is not None else None,
                    std=round(std_val, 3) if std_val is not None else None,
                    min_val=min_v,
                    max_val=max_v,
                    skewness=skew_val,
                    anomaly_flags=anomalies,
                )
            )

        return DatasetSummary(
            row_count=n_rows,
            col_count=n_cols,
            memory_mb=mem_mb,
            duplicate_rows=n_dups,
            columns=profiles,
        )
