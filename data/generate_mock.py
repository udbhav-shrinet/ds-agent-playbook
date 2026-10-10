import polars as pl
import random
from pathlib import Path

def generate_sample_dataset(output_path: str = "data/sample_transactions.csv"):
    Path("data").mkdir(exist_ok=True)
    random.seed(42)
    n = 250

    customers = [f"CUST_{i:04d}" for i in range(1, 40)]
    categories = ["Electronics", "Fashion", "Groceries", "Home", "Automotive"]

    data = {
        "transaction_id": [f"TX_{i:05d}" for i in range(n)],
        "customer_id": [random.choice(customers) for _ in range(n)],
        "category": [random.choice(categories) if random.random() > 0.05 else None for _ in range(n)],
        "amount": [round(random.expovariate(1/120.0), 2) for _ in range(n)],  # skewed distribution
        "discount_pct": [random.choice([0.0, 0.05, 0.1, 0.2, None]) for _ in range(n)],
        "quantity": [random.randint(1, 8) for _ in range(n)],
        "constant_status": ["COMPLETED" for _ in range(n)],  # Zero variance
        "heavy_missing_notes": [None if random.random() > 0.15 else "Review requested" for _ in range(n)], # 85% missing
    }

    df = pl.DataFrame(data)
    # Add some duplicate rows deliberately to test cleaner
    duplicates = df.head(5)
    df = pl.concat([df, duplicates])

    df.write_csv(output_path)
    print(f"Generated {df.height} rows with deliberate anomalies to {output_path}")

if __name__ == "__main__":
    generate_sample_dataset()
