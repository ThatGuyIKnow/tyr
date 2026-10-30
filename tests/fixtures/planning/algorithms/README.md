# A* statistics fixtures

The `astar.json` fixtures record goal-on-pop A* with strict `f > incumbent`
pruning and completion of the final f-layer. The 591 recorded configurations
and existing skip maps are preserved from the earlier fixtures. The two
`astar-first-goal.json` files retain those original bytes:

- ground SHA-256: `d1640c3a943a783e2f420e2b1c8e60c2d1e7688d20249c7a31119d313751860a`
- lifted SHA-256: `dcb5a7ed825a5d8c465aff1a93ad8b52111ca469885693111cfc2413e35ab30e`

The migration audit found 357 changed outcomes, including the two newly bounded
Rover configurations described below. The Woodworking `hadd_unit`
plan changed from ten actions to nine in both task representations; VAL
accepted both the old and new plans, and unit search cost equals action count.

The Rover `blind_general` configurations reach a very large zero-cost final
layer. Both remain recorded with a deterministic 1,000-state budget and an
`OUT_OF_STATES` status, no plan, and exact counters. This bound is enforced by
both C++ fixture tests and Python replay/generation; it is not a completeness
or performance result for Rover.
