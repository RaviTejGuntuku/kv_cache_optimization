# Proposed JSON Schema Draft 2020-12 generation profile

This is a proposed support contract, not implemented support or a new standard. Acceptance is determined by keyword combinations, supported predicate forms, and explicit generation bounds. The full language is [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12).

| Feature | Proposed scope | Example |
|---|---|---|
| Objects | Typed required/optional fields, closed objects, typed extra properties | [Rows 1–4](examples.md#row-1) |
| Conditional structure | Finite discriminator choices determining required and permitted fields | [Row 14](examples.md#row-14) |
| Field dependencies | Known-field presence dependencies; restrict dependentSchemas to reducible forms | [Row 10](examples.md#row-10) |
| Exclusions | Simple mutual exclusion, not unrestricted negation | [Row 17](examples.md#row-17) |
| Alternatives and composition | Disjoint tagged layouts and compatible object constraints resolved during compilation | [Rows 15–16](examples.md#row-15) |
| Key constraints | Regular name patterns and compatible intersections of matching value rules | [Rows 5–8](examples.md#row-5) |
| Local values | Types, enums, constants, nullable values, regular string patterns and length bounds | [Rows 18–21](examples.md#row-18) |
| Arrays and tuples | Homogeneous arrays, positional tuples, finite lengths | [Rows 23–24](examples.md#row-23) |
| Matching items | Bounded counts for finite enums or simple supported predicates | [Row 26](examples.md#row-26) |
| Uniqueness | Exact uniqueness of strings or finite-enum items; scoped state | [Row 25](examples.md#row-25) |
| Reuse | Bundled static references; recursive structures with explicit depth limits | [Rows 28–29](examples.md#row-28) |

## Initial exclusions

- Numeric bounds, divisibility, and arbitrary-precision arithmetic constraints.
- General negation, unrestricted overlapping alternatives, arbitrary intersections, or arbitrary value-dependent schemas.
- Dynamic references and unrestricted recursion.
- General evaluated-property/item tracking across arbitrary composition; accept only forms whose closure can be established by the compiler.
- Arbitrary backreference regexes and general JSON-value uniqueness in the initial profile.
- Assertions about business truth, external resource existence, or correctness of the chosen agent action.

A small finite-marker backreference may be explicitly expanded into regular alternatives. Recognized annotation keywords may be retained as metadata; unsupported assertion semantics must be rejected rather than treated as annotations. Format assertion policy must be explicit; `format` is not universally an assertion under the default 2020-12 vocabulary.

## Relationship to CFGs

XGrammar and llguidance do not implement a formally identified, complete “context-free subset of JSON Schema.” Their converters have implementation-specific coverage. Some rejected constructs, including finite conditional layouts, admit context-free or even finite-state representations. Conversely, unrestricted equality and uniqueness over arbitrary strings require remembered content beyond a plain CFG.

The proposed contribution is efficient execution of supported structure and scoped state. It is not a blanket claim to be the first system beyond CFGs. Finite-domain uniqueness and bounded structure can sometimes be compiled into automata, but expansion can be costly.
