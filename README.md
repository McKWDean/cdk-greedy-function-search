# CDK configuration analysis

Enterprise-level configuration breadth, Jaccard similarity, segmentation (k-means), and cohort coverage tools for CDK setup data.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy openpyxl
```

Place required inputs at the repo root (see `segmentation/pipeline.py` module docstring): `Enterprise_Setup_Map.csv`, `Cnumber_to_Enterprise_Map.xlsx`, BOB / forms / spooler / users exports, 3PA workbooks, etc. These paths are **not** committed to git.

## Main entry points

| Command | Purpose |
|--------|---------|
| `python -m segmentation.pipeline` | Build enterprise features, k-means clusters, deck CSVs |
| `python -m segmentation.jaccard_similarity_clustering` | Jaccard / hierarchical clustering on dimension sets |
| `python -m segmentation.acct_three_cohort_configuration_breadth` | ACCT T&L / M-Seg1 / Seg1 breadth vs population (`--vertical service` for Service sheets) |
| `python -m segmentation.accounting_segment1_analysis` | Segment 1 coverage + greedy 80% test set |
| `python -m segmentation.enterprise_dimension_export` | Enterprise × dimension Excel export |
| `python -m segmentation.exact_setup_analysis` | Exact duplicate configuration groups |
| `python -m segmentation.configuration_overlap_report` | Overlap executive summary + charts |

Aggregation rules and column definitions are documented in `segmentation/pipeline.py` (`AGGREGATION_CHOICES`).

## License

Internal / use per your organization’s policy.
