import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import polars as pl
from ds_agent_playbook.profiler import ProfilerEngine
from ds_agent_playbook.cleaner import CleanerAgent
from ds_agent_playbook.features import FeatureSynthesizer

console = Console()

@click.group()
def main():
    """DS Agent Playbook: Autonomous recipes for ML & Data Science pipelines."""
    pass

@main.command()
@click.argument("file_path", type=click.Path(exists=True))
def inspect(file_path: str):
    """Deep structural and statistical audit of a CSV or Parquet file."""
    console.print(Panel(f"[bold cyan]Auditing Dataset:[/] {file_path}", expand=False))
    
    if file_path.endswith(".parquet"):
        profiler = ProfilerEngine.from_parquet(file_path)
    else:
        profiler = ProfilerEngine.from_csv(file_path)

    summary = profiler.profile()

    table = Table(title="Dataset Architecture & Diagnostics")
    table.add_column("Column", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Missing %", style="yellow")
    table.add_column("Distinct", style="green")
    table.add_column("Mean ± Std", style="blue")
    table.add_column("Anomalies / Signals", style="red")

    for col in summary.columns:
        mean_std = f"{col.mean} ± {col.std}" if col.mean is not None else "-"
        anomalies = ", ".join(col.anomaly_flags) if col.anomaly_flags else "[dim]Normal[/]"
        table.add_row(
            col.name,
            col.dtype,
            f"{col.null_percentage}%",
            str(col.unique_count),
            mean_std,
            anomalies,
        )

    console.print(table)
    console.print(
        f"[bold]Rows:[/] {summary.row_count} | [bold]Cols:[/] {summary.col_count} | "
        f"[bold]Duplicates:[/] {summary.duplicate_rows} | [bold]Memory:[/] {summary.memory_mb} MB\n"
    )

@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Path to write the cleaned Parquet file")
def clean(file_path: str, output: str):
    """Devise an automated cleansing plan and optionally apply it."""
    console.print(Panel(f"[bold yellow]Generating Cleansing Actions:[/] {file_path}", expand=False))
    
    df = pl.read_parquet(file_path) if file_path.endswith(".parquet") else pl.read_csv(file_path)
    profiler = ProfilerEngine(df)
    summary = profiler.profile()

    cleaner = CleanerAgent(summary)
    plan = cleaner.devise_plan()

    table = Table(title="Agentic Cleansing Plan")
    table.add_column("Action", style="green")
    table.add_column("Column", style="cyan")
    table.add_column("Rationale", style="white")

    for act in plan:
        table.add_row(act.action_type, act.column, act.rationale)

    console.print(table)

    if output:
        cleaned_df = cleaner.apply_plan(df, plan)
        if output.endswith(".csv"):
            cleaned_df.write_csv(output)
        else:
            cleaned_df.write_parquet(output)
        console.print(f"\n[bold green]Cleaned dataset written to:[/] {output} ({cleaned_df.height} rows, {cleaned_df.width} columns)")

@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--target", "-t", default=None, help="Target label column")
def synthesize(file_path: str, target: str):
    """Autonomously propose feature interactions and ratio transformations."""
    console.print(Panel(f"[bold magenta]Synthesizing Feature Hypotheses:[/] {file_path}", expand=False))
    
    df = pl.read_parquet(file_path) if file_path.endswith(".parquet") else pl.read_csv(file_path)
    synth = FeatureSynthesizer(df, target_col=target)
    candidates = synth.propose_features()

    table = Table(title="Generated Feature Candidates")
    table.add_column("Feature", style="bold magenta")
    table.add_column("Transformation", style="cyan")
    table.add_column("Hypothesis", style="white")

    for c in candidates:
        table.add_row(c.feature_name, c.transformation, c.hypothesis)

    console.print(table)

if __name__ == "__main__":
    main()
