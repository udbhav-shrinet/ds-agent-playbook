# ds-agent-playbook ⚡

> **Autonomous, zero-bloat recipes for Data Science pipelines**: Automated exploratory data analysis (EDA), deterministic data cleansing, feature hypothesis synthesis, and statistical anomaly detection powered by Polars & DuckDB.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Powered by Polars & DuckDB](https://img.shields.io/badge/Engine-Polars%20%26%20DuckDB-orange.svg)](https://pola.rs/)

---

## 💡 Why ds-agent-playbook?

Most "AI Data Science" frameworks are heavyweight, slow wrappers around opaque prompt chains that cost dollars per prompt and hallucinate column names.

**ds-agent-playbook** provides **modular, deterministic agentic recipes** designed for production data pipelines:
- **Instant Profiling**: Zero-dependency statistical profiling with skewness, variance, and anomaly detection.
- **Auditable Cleansing Plans**: The agent inspects column distributions and generates an explicit, inspectable action plan before modifying data.
- **Autonomous Feature Synthesis**: Formulates candidate features, ratio interactions, and grouping heuristics automatically.
- **Blazing Fast**: Engineered on top of Polars and DuckDB for microsecond execution times.

---

## 🛠️ Architecture

```
Raw CSV / Parquet
       │
       ▼
┌─────────────────────────┐
│   ProfilerEngine        │ ──▶ Statistical Profiling & Anomaly Discovery
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│   CleanerAgent          │ ──▶ Generates & Executes Deterministic Plan
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│   FeatureSynthesizer    │ ──▶ Domain Ratio & Aggregation Hypotheses
└─────────────────────────┘
       │
       ▼
 Clean Parquet / Model Matrix
```

---

## ⚡ Quickstart

### 1. Clone & Setup with `uv`

```bash
git clone https://github.com/udbhav-shrinet/ds-agent-playbook.git
cd ds-agent-playbook
uv sync
```

### 2. Inspect a Dataset

```bash
uv run ds-agent inspect data/sample_transactions.csv
```

### 3. Generate & Apply Cleansing Plan

```bash
uv run ds-agent clean data/sample_transactions.csv --output data/cleaned.parquet
```

### 4. Synthesize New Feature Hypotheses

```bash
uv run ds-agent synthesize data/cleaned.parquet
```

---

## 📖 Recipes

The `recipes/` directory contains standalone, copy-pasteable Python scripts you can integrate directly into your own scripts or Jupyter notebooks:

1. [`recipes/01_automated_eda.py`](recipes/01_automated_eda.py): Instant dataset audit and anomaly discovery.
2. [`recipes/02_automated_cleansing.py`](recipes/02_automated_cleansing.py): Formulate and apply deterministic data cleaning actions.
3. [`recipes/03_feature_synthesis.py`](recipes/03_feature_synthesis.py): Propose and materialize feature interaction vectors.

---

## 🧪 Testing

```bash
uv run pytest tests/
```

---

## 🤝 Contributing

Contributions are welcome! Please check out [CONTRIBUTING.md](CONTRIBUTING.md) for local setup and guidelines. Feel free to submit an issue or a PR proposing new recipes in `recipes/`.

---

## 👤 Author

**Udbhav Shrinet**  
- GitHub: [@udbhav-shrinet](https://github.com/udbhav-shrinet)

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
