# Studies

This is the canonical home for experiment organization.

Layout:

- `specs/`: experiment definitions, hypotheses, and procedures
- `results/`: curated experiment bundles with inputs, metrics, graphs, and preserved raw slices
- `runs/`: transient or heavyweight run artifacts, logs, and scratch experiment trees

Intent:

- `specs/` answers what an experiment is
- `results/` answers what we kept from that experiment
- `runs/` answers what was produced during execution

Backwards-compatibility aliases still exist for non-experiment namespaces:

- `runs -> studies/runs`
