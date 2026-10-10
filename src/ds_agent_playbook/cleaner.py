from typing import List
import polars as pl
from ds_agent_playbook.schemas import CleaningAction, DatasetSummary

class CleanerAgent:
    """Deterministic agent that diagnoses datasets and devises an executable cleaning plan."""

    def __init__(self, summary: DatasetSummary):
        self.summary = summary

    def devise_plan(self) -> List[CleaningAction]:
        actions: List[CleaningAction] = []

        if self.summary.duplicate_rows > 0:
            actions.append(
                CleaningAction(
                    action_type="drop_duplicates",
                    column="*",
                    rationale=f"Remove {self.summary.duplicate_rows} duplicate rows to avoid training bias.",
                )
            )

        for col in self.summary.columns:
            # Drop invariant columns
            if col.unique_count <= 1:
                actions.append(
                    CleaningAction(
                        action_type="drop_column",
                        column=col.name,
                        rationale=f"Zero-variance column with only {col.unique_count} distinct value.",
                    )
                )
                continue

            # Heavy missingness
            if col.null_percentage > 50.0:
                actions.append(
                    CleaningAction(
                        action_type="drop_column",
                        column=col.name,
                        rationale=f"Severe missingness ({col.null_percentage}% missing); imputation adds noise.",
                    )
                )
            elif col.null_percentage > 0:
                # Moderate missingness: impute
                is_num = any(t in col.dtype.lower() for t in ["int", "float"])
                strategy = "median" if is_num else "mode"
                actions.append(
                    CleaningAction(
                        action_type="impute",
                        column=col.name,
                        rationale=f"Impute {col.null_percentage}% missing values using {strategy}.",
                        parameters={"strategy": strategy},
                    )
                )

            # High skewness
            if col.skewness is not None and abs(col.skewness) > 1.5:
                if col.min_val is not None and col.min_val >= 0:
                    actions.append(
                        CleaningAction(
                            action_type="log1p_transform",
                            column=col.name,
                            rationale=f"Mitigate positive skewness ({col.skewness}) via log1p transformation.",
                        )
                    )

        return actions

    def apply_plan(self, df: pl.DataFrame, actions: List[CleaningAction]) -> pl.DataFrame:
        cleaned_df = df.clone()
        for act in actions:
            if act.action_type == "drop_duplicates":
                cleaned_df = cleaned_df.unique()
            elif act.action_type == "drop_column":
                if act.column in cleaned_df.columns:
                    cleaned_df = cleaned_df.drop(act.column)
            elif act.action_type == "impute":
                col = act.column
                if col in cleaned_df.columns:
                    strategy = act.parameters.get("strategy", "median")
                    if strategy == "median":
                        med = cleaned_df[col].median()
                        cleaned_df = cleaned_df.with_columns(pl.col(col).fill_null(med))
                    elif strategy == "mode":
                        mode_val = cleaned_df[col].mode().first()
                        cleaned_df = cleaned_df.with_columns(pl.col(col).fill_null(mode_val))
            elif act.action_type == "log1p_transform":
                col = act.column
                if col in cleaned_df.columns:
                    cleaned_df = cleaned_df.with_columns(pl.col(col).log1p().alias(f"{col}_log1p"))

        return cleaned_df
