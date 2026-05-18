# Datasets

This is the canonical home for all workload data used by the repo.

Layout:

- `raw/`: unprocessed source corpora such as ShareGPT dumps
- `processed/`: runnable real-world slices and prepared benchmark subsets
- `synthetic/`: generated synthetic workloads and manifests

Important note:

- `data/` still exists, but only as a backwards-compatibility alias layer.
- New documentation should point to `datasets/...`, not `data/...`.
