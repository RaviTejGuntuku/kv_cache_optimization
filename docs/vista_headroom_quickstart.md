# Vista Headroom Quickstart

This note explains the minimum steps needed to move the empirical headroom pilot to a TACC Vista allocation and run it from `$SCRATCH`.

## 1. Sync from your Mac to Vista

Run this locally from the repo root on your Mac:

```bash
benchmarking/setup/sync_to_vista.sh <tacc_username>
```

Optional second argument if you want a custom destination:

```bash
benchmarking/setup/sync_to_vista.sh <tacc_username> /scratch/<project_or_user_path>/kv_cache_research
```

The sync is intentionally curated:

- includes the empirical headroom code, setup scripts, and specs
- excludes:
  - `.git`
  - local virtualenvs
  - large local result dumps
  - processed datasets
  - `external/vllm` and `external/lmcache`
  - the local `sglang` tree

Those excluded pieces are either large or unnecessary on Vista. The remote setup scripts will recreate what is needed there.

## 2. Log into Vista

```bash
ssh <tacc_username>@vista.tacc.utexas.edu
```

Then move to the synced repo:

```bash
cd \$SCRATCH/kv_cache_research
```

If you used a custom destination, `cd` there instead.

## 3. Bootstrap the repo on Vista

Refresh the baseline repos:

```bash
benchmarking/setup/setup_empirical_headroom_baselines.sh
```

Create the Python environment:

```bash
module load gcc/13.2.0 cuda/12.8 python3/3.11.8
benchmarking/setup/setup_empirical_headroom_env.sh
source .venv-empirical-headroom/bin/activate
```

If `python3 --version` is below `3.10`, load a newer Python module before creating the environment.

The default environment script assumes a CUDA 12.8 stack and installs matching PyTorch wheels before building `lmcache`.
It also upgrades the active build backend (`setuptools`, `packaging`, `ninja`) before the `lmcache` build step.

Generate the pilot workloads:

```bash
python benchmarking/workload_generators/generate_empirical_headroom_workloads.py
```

## 4. Get an interactive GPU node

Use `idev` to get a short Grace-Hopper development session:

```bash
idev -p gh-dev -N 1 -n 1 -t 1:00:00
```

Once on the compute node, reactivate the environment:

```bash
cd \$SCRATCH/kv_cache_research
source .venv-empirical-headroom/bin/activate
```

## 5. Run a small pilot

Prefix-only pilot:

```bash
SYSTEM=vllm_apc \
MODEL=meta-llama/Llama-3.1-8B-Instruct \
BUNDLE_ROOT=datasets/processed/empirical_headroom/shared_prefix_64x16 \
REQUEST_LIMIT=16 \
benchmarking/setup/run_empirical_headroom_pilot.sh
```

Broad-reuse pilot:

```bash
SYSTEM=lmcache_cacheblend \
MODEL=meta-llama/Llama-3.1-8B-Instruct \
BUNDLE_ROOT=datasets/processed/empirical_headroom/mixed_reuse_1024req \
REQUEST_LIMIT=8 \
MAX_COUNTERFACTUALS_PER_REQUEST=2 \
benchmarking/setup/run_empirical_headroom_pilot.sh
```

## 6. What files to inspect

Oracle 0:

- `studies/results/empirical_headroom_pilot/oracle0/.../summary.json`

Marginal counterfactuals:

- `studies/results/empirical_headroom_pilot/marginal_counterfactuals/.../summary.json`
- `studies/results/empirical_headroom_pilot/marginal_counterfactuals/.../marginal_counterfactuals.jsonl`

Missed-opportunity accounting:

- `studies/results/empirical_headroom_pilot/missed_opportunity_accounting/summary.json`
- `studies/results/empirical_headroom_pilot/missed_opportunity_accounting/cause_breakdown.csv`
- `studies/results/empirical_headroom_pilot/missed_opportunity_accounting/missed_opportunities.jsonl`

## 7. Nsight traces

If `nsys` is installed on Vista, add profiling by setting the environment variables used by the runners, for example:

```bash
PROFILE_ORACLE0_EVERY=8
PROFILE_BASELINE_EVERY=8
PROFILE_COUNTERFACTUAL_EVERY=16
```

The sampled traces will appear under the run’s `nsys/` directory.
