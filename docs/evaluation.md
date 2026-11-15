# Evaluation and retained evidence

## Questions

1. Which original schemas are faithfully enforced, rejected, or accepted with weakened constraints?
2. How often does generation finish with a schema-valid output under matched limits?
3. Does structural enforcement preserve semantic task quality?
4. Does plan-aware serving improve latency, throughput, or model work relative to strong baselines?

## Baselines and fairness

Compare against XGrammar, llguidance, JSONformer, and relevant Guidance/SGLang execution paths. Include original schemas and semantically equivalent supported rewrites. Review [SchemaFill](https://arxiv.org/abs/2610.07086) for slot-parallel speculative generation and [DOMINO](https://arxiv.org/abs/2403.06988) for tokenizer-aligned constrained decoding.

Separate coverage improvements from speedups on shared coverage. Match models, token budgets, hardware, concurrency, and output requirements. Report compilation and warm execution separately. Include ordinary batching and deterministic fast-forwarding before attributing a gain to the proposed scheduler. An infilling model must be assessed for semantic quality and context differences, not compared as if its generation distribution were unchanged.

## Measurements

| Dimension | Measurements |
|---|---|
| Coverage | Compiler acceptance; faithful enforcement; explicit rejection; source-only versus empirically tested findings |
| Correctness | JSON parsing, independent schema validation, adversarial invalid-instance probes, false acceptance and false rejection |
| Completion | Complete valid outputs, valid unfinished prefixes, dead ends, budget exhaustion, and persistent loops |
| Semantics | Task-specific accuracy, factuality where measurable, and correctness of selected alternatives |
| Serving | Valid completed records/second, latency percentiles, model calls, sampled versus deterministic tokens, constraint overhead, batching utilization |
| Cost | Compile time, compiled-plan size, runtime state, retained uniqueness data, and total model processing |

## Saved baseline evidence

- [Frozen 44-task catalog](../experiments/specs/structured_generation_headroom/suite44/tasks.json): 20 controls and 24 stress schemas, with selection metadata and example files.
- [CPU preflight audit](../experiments/results/20261110T044100Z__structured_generation_manual_preflight_release/summary.json): compiler outcomes and saved instance-recognition probes. Neighboring per-case JSON records retain detailed results.
- [Full-study summary](../experiments/results/20261112T203524Z__structured_generation_headroom_full/analysis/summary.json): the original study summary, retained without relabeling its conditions.
- [Completion replay](../experiments/results/20261112T203524Z__structured_generation_headroom_full/analysis/completion_dive/report.txt): analysis of unfinished constrained prefixes.
- [Original backreference rejection](../experiments/results/20261109__xgrammar_schema_audit/benchmark-stress-38__raw.json) and [finite rewrite](../experiments/results/20261109__xgrammar_schema_audit/benchmark-stress-38__finite-backreference.json).

The full study recorded XGrammar compiler acceptance for 44 schemas and llguidance acceptance for 32. Acceptance is not faithful coverage. The 12 llguidance rejections include restrictions and reference-resolution issues; they are not 12 independent expressive gaps. Completion replay found 38 examined constrained budget-exhausted outputs were accepted prefixes with nonempty continuation masks and no allowed EOS. This demonstrates incomplete generation in those cases, not that every constrained decoder lacks progress guarantees.

The audit pins XGrammar 0.2.8 and llguidance 1.9.1. JSONformer coverage is source-based at commit `bfad031876ace84ec0a7853718a1c0828ea1653a`; no JSONformer model benchmark is claimed. The newer illustrative examples were validated independently, not all rerun through the decoders. The coverage document and table distinguish these evidence types.

These are selected saved artifacts, not a complete reproducibility bundle. Some retained records contain original local paths and references to raw files not published here. Historical directory names and condition metadata are preserved to prevent misrepresenting the measurements. No results in this repository measure the proposed architecture yet.

## Novelty boundary

JSONformer already fills fixed structure; Guidance already mixes generation and control flow; SGLang already optimizes structured programs and deterministic spans; SchemaFill already exploits slot-level parallelism. A defensible contribution must specify the new compiler/plan/runtime mechanism, prove its scoped guarantees, and demonstrate benefits beyond these techniques. Broader keyword acceptance alone does not establish soundness or a serving improvement.
