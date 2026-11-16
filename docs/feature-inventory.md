# Feature inventory and implementation decisions

The Word document preserves its original five pages and appends this decision guide. Categories: V = value; O = outline; C = cross-part; M = composition/reference; Srl = serialization. Costs concern restricted monitor implementations, not GPU inference or full token-mask construction. Support is pinned to XGrammar 0.2.8, llguidance 1.9.1, and JSONformer bfad031. Proposed support is not implemented.

| Feature and category | XGrammar | llguidance | JSONformer | Compile time | Generator runtime | Proposed scope |
|---|---|---|---|---|---|---|
| [F1 Declared required keys](feature-examples.md#feature-1) — O | ✓ | ✓ | ✓ | Easy: index declared keys and required set; O(S) for this flat object. | Emit known keys; track k presence bits. Stack cost O(h). | Core |
| [F2 Optional field omission](feature-examples.md#feature-2) — O | ✓ | ✓ | ✗ | Easy: compile optional-field decisions and dependency closure. | Decide omission once; ensure required fields remain reachable. | Core |
| [F3 Required but undeclared key](feature-examples.md#feature-3) — O | ✗ | ✓ | ✗ | Easy: materialize required names from the extra-key schema; detect forbidden names. | Supply id even when absent from properties; known-key lookup. | Core |
| [F4 Typed extra keys](feature-examples.md#feature-4) — O+V | ✓ | ✓ | ✗ | Easy: compile one additional-value schema; close known-key exceptions. | Generate a key and typed value; exact key uniqueness needs a scope set. | Core |
| [F5 All key names constrained](feature-examples.md#feature-5) — V(key) | Map names checked; named keys can bypass. | ✗ | ✗ | Moderate: compile regular key predicate; precheck every declared name. DFA may expand. | Scan every decoded name in O(length), including declared names. | Core |
| [F6 Pattern keys plus extra keys](feature-examples.md#feature-6) — O+V | Generic extra branch can bypass matching rule. | ✓ | ✗ | Moderate: partition matching keys from extras; preserve named-key intersections. | Classify keys, then enforce all applicable value rules. | Core |
| [F7 Overlapping key patterns](feature-examples.md#feature-7) — V(key)+C | ✗ | ✗ | ✗ | Moderate: intersect compatible regular value rules. Product size at most product(p_i). | Classify key then run intersected monitor. Do not OR overlapping rules. | Add |
| [F8 Named key also matches pattern](feature-examples.md#feature-8) — V(key)+C | ✗ | ✓ | ✗ | Moderate: pre-intersect each named field with every matching pattern. | Use one compiled local monitor for code. No general solving per token. | Add |
| [F9 Property count with optional fields](feature-examples.md#feature-9) — O | ✓ | Restricted; general optional-field case rejected. | ✗ | Easy for known fields: check feasible min/max and required/optional counts. | Counter O(log(M+1)) bits; remaining slots constrain field choices. | Core |
| [F10 Presence dependencies](feature-examples.md#feature-10) — C→O | ✗ | ✗ | ✗ | Easy for positive presence edges: O(k+e) closure; precompute required sets. | Activate companion fields; update seen/required bitsets. | Add |
| [F11 Close an extended object](feature-examples.md#feature-11) — M→O | Partial keyword handling; not this composed closure. | ✗ | ✗ | Moderate only for statically flattenable closure. General annotation elimination can expand exponentially [A]. | Emit only proven evaluated keys. General branch-dependent tracking is deferred. | Restricted |
| [F12 Any key order with required identity](feature-examples.md#feature-12) — Srl | Ordered by default; any_order loses identity checks. | ✗ | ✗ | Easy for fixed order. Explicit arbitrary-order DFA can require 2^k states. | Fixed-order output. Optional any-order mode needs k-bit identity tracking, not just a count. | Canonical |
| [F13 Unique dynamic key names](feature-examples.md#feature-13) — Srl+C | ✗ | ✗ | Fixed names unique; dynamic maps absent. | Easy: allocate name-set scope; select decoded-name equality policy. | Expected O(B) hashing plus exact collision checks; O(B) retained names. Prefix exclusion additional. | Core |
| [F14 Conditional required fields](feature-examples.md#feature-14) — C→O | ✗ | ✗ | ✗ | Easy for a finite tag tested before dependent fields. Interacting condition combinations can expand. | Resolve tag once, activate its fields, reserve space for their completion. | Add |
| [F15 Tagged alternative layouts](feature-examples.md#feature-15) — M→O | ✓ | ✓ | ✗ | Easy for explicit disjoint tags: compile branches in O(total branch size). | Choose tag, execute one layout. General overlapping oneOf excluded. | Core |
| [F16 Combine object requirements](feature-examples.md#feature-16) — M | ✗ | Compatible intersections supported; general limits. | ✗ | Moderate: merge compatible known fields; preserve each branch closure. General satisfiability is hard [B]. | Execute merged plan. Reject combinations needing general online intersection. | Restricted |
| [F17 Mutually exclusive credentials](feature-examples.md#feature-17) — C→O | ✗ | ✗ | ✗ | Easy for pairwise presence exclusion; compile conflict sets. Arbitrary not excluded. | After password, disable token; after token, disable password. | Add |
| [F18 Pattern plus string length](feature-examples.md#feature-18) — V | ✗ | ✓ | ✗ | Moderate: combine regular pattern with length monitor; at most p(M+2) DFA product states. | O(1) transition per decoded character in a precompiled DFA; count Unicode characters. | Core |
| [F19 Finite matching markers](feature-examples.md#feature-19) — V | Original rejected; finite expansion compiled. | ✗ | ✗ | Easy for finite captures only: expand r fixed markers. Arbitrary backreferences excluded. | Remember finite marker ID or run expanded automaton. | Restricted |
| [F20 Regular lookahead constraint](feature-examples.md#feature-20) — V | ✗ | ✗ | ✗ | Moderate for regular lookahead reducible to intersection/complement; possible DFA blow-up. | Precompiled regular monitor; no general regex backtracking during generation. | Restricted |
| [F21 Enums constants and nullable values](feature-examples.md#feature-21) — V | ✓ | ✓ | ✗ | Easy for compatible scalar enum/const/null: build literal trie in O(total literal length). | Follow literal trie or select a permitted scalar; decoded-value semantics matter. | Core |
| [F23 Array length including empty](feature-examples.md#feature-23) — O | ✓ | ✓ | ✗ | Easy: check bounds and compile items, including the empty branch. | Counter O(log(M+1)) bits; reserve required items and closing syntax. | Core |
| [F24 Heterogeneous tuples](feature-examples.md#feature-24) — O+V | ✓ | ✓ | ✗ | Easy: compile each prefix position and the allowed tail. | Position dispatch plus stack; disallow tail when items=false. | Core |
| [F25 Unique array values](feature-examples.md#feature-25) — C | ✗ | ✗ | ✗ | Moderate: restrict to strings/finite enums; prove enough remaining values under other bounds. | Expected O(B) hashing with exact equality; O(B) state. Prefix-safe exclusion is a separate cost. | Add |
| [F26 Matching array item counts](feature-examples.md#feature-26) — C+O | ✗ | ✗ | ✗ | Moderate: compile simple item predicate and count feasibility. Overlapping complex predicates deferred. | Classifier plus saturated counter; reserve enough matching future slots. | Add |
| [F27 Close a reused tuple](feature-examples.md#feature-27) — M→O | Partial local handling; not this ref closure. | ✗ | ✗ | Moderate for a statically resolved tuple: calculate evaluated positions. General annotation elimination costly [A]. | Dispatch allowed positions and forbid the tail; no general evaluation history. | Restricted |
| [F28 Local recursive references](feature-examples.md#feature-28) — M→O | ✓ | ✓ | ✗ | Moderate: build static ref graph O(S+E), enforce depth and existence of a finite base case. | Stack O(h) plus per-scope constraint state; depth limit must retain a completion. | Core bounded |
| [F29 Bundled URI references](feature-examples.md#feature-29) — M | ✗ | Referencing build supports registry / retriever; no default fetch. | ✗ | Moderate: resolve URI IDs from supplied resources before serving; graph work O(S+E). | No network lookup; jump to a resolved plan node. | Core bundled |
| [F30 Dynamic recursive references](feature-examples.md#feature-30) — M | ✗ | ✗ | ✗ | Hard unrestricted: dynamic-reference validation is PSPACE-complete in schema/combined complexity [T]. | No cheap general runtime promised; fixed-schema validation PTIME does not settle prefix generation. | Defer |
| [F31 Integer type without arithmetic](feature-examples.md#feature-31) — V | ✓ | ✓ | ✗ | Easy: compile integer lexical grammar; preserve JSON number semantics. | Scan numeric spelling. Type support is distinct from bounds or divisibility. | Core |
| [F32 Boolean type](feature-examples.md#feature-32) — V | ✓ | ✓ | ✓ | Easy: two literal alternatives. | Select true/false; constant-size monitor. | Core |
| [F33 Fixed scalar constant](feature-examples.md#feature-33) — V | ✓ | ✓ | ✗ | Easy: check compatible type and encode literal, O(literal length). | Emit constant; it still contributes to model context where required. | Core |
| [F34 Simple anyOf union](feature-examples.md#feature-34) — M+V | ✓ | ✓ | ✗ | Easy for disjoint type branches; O(sum of branch sizes). General siblings/intersections need separate checking. | Select type and run its branch; preserve allowed model choices if branches overlap. | Core restricted |
| [F35 Regular string pattern alone](feature-examples.md#feature-35) — V | ✓ | ✓ | ✗ | Compile regex to NFA, then optional DFA. s-state NFA may require 2^s DFA states. | DFA transition O(1) per decoded character; NFA execution has a different cost. | Core |
| [F36 String length alone](feature-examples.md#feature-36) — V | ✓ | ✓ | ✗ | Easy: validate bounds; compile character counter or length-aware grammar. | O(log(M+1)) counter bits; count decoded Unicode characters, not bytes or tokens. | Core |
| [F37 Homogeneous array item types](feature-examples.md#feature-37) — O+V | ✓ | ✓ | ✓ | Easy: compile item schema once, with explicit external generation length bound. | Reuse item plan per position; stack O(h), plus collection counters. | Core bounded |
| [F38 Presence-only dependentSchemas](feature-examples.md#feature-38) — C→O | ✗ | ✗ | ✗ | Easy: normalize this presence-only form to dependency edges. Arbitrary subschemas excluded. | Same bitset/closure mechanism as row 10; no value-dependent schema solver. | Add restricted |
| [F39 Selected format assertions](feature-examples.md#feature-39) — V / policy | Finite format subset; lexical checks. | Finite format subset; lexical checks. | ✗ | Default format is annotation-only. Full semantic format checking differs from a lexical regex. | No full-format guarantee. Example invalidity assumes opt-in FormatChecker, not default vocabulary. | Defer assertions |
| [F40 Numeric assertions inventory](feature-examples.md#feature-40) — V | Bounds; number multipleOf ignored. | ✓ for this bounded example | ✗ | Supported union includes ordinary numeric assertions, but arithmetic is outside our requested initial profile. | No arithmetic enforcement proposed. This row records baseline coverage only. | Exclude |

## Shared original-schema enforcement gaps

These source-grounded examples separate faithful enforcement of the original schema, not the ability to coincidentally emit a valid instance. All valid/invalid pairs pass an independent oracle check; removing the highlighted separating constraint makes each negative witness valid. No new decoder run or proposed-parser result is claimed. Equivalent rewrites may be supported.

| Case | Feature | Failure mode |
|---|---|---|
| [S1 Active scientific record](feature-examples.md#S1) | F14 | X ignores if/then; strict L rejects. Finite tagged rewrite may work in both. |
| [S2 Profile requires source](feature-examples.md#S2) | F10 | X ignores dependency; strict L rejects. Compile presence closure. |
| [S3 One credential method](feature-examples.md#S3) | F17 | X ignores not; strict L rejects. Disjoint explicit layouts provide a rewrite. |
| [S4 Distinct role values](feature-examples.md#S4) | F25 | X ignores uniqueness; strict L rejects. Finite domain makes exhaustion decidable at compile time. |
| [S5 At least one administrator](feature-examples.md#S5) | F26 | X does not count matches; strict L rejects. Reserve one matching slot. |
| [S6 At most one administrator](feature-examples.md#S6) | F26 | X ignores matching-count cap; strict L rejects. Saturated counter plus remaining slots. |
| [S7 Key matches two value rules](feature-examples.md#S7) | F7 | X alternatives do not enforce intersection; L rejects overlapping key patterns. |
| [S8 Dependency expressed as a subschema](feature-examples.md#S8) | F38 | X ignores dependentSchemas; strict L rejects. Normalize this restricted form to S2. |

## Proposed support contract

Core: primitive types, scalar enum/const/null, regular strings with length bounds, fixed/optional/required object fields, typed extras, object counts, arrays, tuples, and bounded collection lengths.

Add: finite conditional fields, presence dependencies and presence-only dependentSchemas, pairwise exclusions, regular key-name constraints with compatible overlaps, string/finite-enum uniqueness, and bounded contains counts for simple classifiers.

Admit with compilation checks: disjoint tagged oneOf, simple anyOf, compatible allOf, bundled static references, bounded recursion with a base case, statically flattenable evaluated-property/item closure, finite-marker rewrites, and regular lookahead normalization.

Defer: unrestricted dynamic references, general negation/intersection, arbitrary JSON-value uniqueness, full format assertions, numeric assertions, arbitrary backreferences, and unbounded generation. Canonical ordering and decoded-key uniqueness are serialization policies. All accepted combinations must preserve a valid completion under explicit generation limits.

Theory: [B — schema satisfiability](https://arxiv.org/abs/1701.02221), [T — modern validation](https://arxiv.org/abs/2307.10034), [A — annotation elimination](https://arxiv.org/abs/2503.11288).
