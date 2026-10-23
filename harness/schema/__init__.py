"""Type schemas for the harness.

- Concept: one schema module per harness module, so complex types live apart
  from the behavior that uses them.
- Design rule: schemas hold types only — no logic, no I/O.
"""
