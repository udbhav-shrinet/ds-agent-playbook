from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    skewness: Optional[float] = None
    anomaly_flags: List[str] = Field(default_factory=list)

class DatasetSummary(BaseModel):
    row_count: int
    col_count: int
    memory_mb: float
    duplicate_rows: int
    columns: List[ColumnProfile]

class CleaningAction(BaseModel):
    action_type: str
    column: str
    rationale: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class FeatureCandidate(BaseModel):
    feature_name: str
    source_columns: List[str]
    transformation: str
    hypothesis: str
    expected_signal_boost: str
